import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio

from data_loader import *
import model
import utils
import pickle

class GCIC(object):
    def __init__(self, config, logger, running_cnt):

        self.running_cnt = running_cnt

        self.alpha_train = config.alpha_train
        self.beta_train = config.beta_train
        self.alpha_query = config.alpha_query
        self.beta_query = config.beta_query
        self.alpha_retrieval = config.alpha_retrieval
        self.beta_retrieval = config.beta_retrieval
        self.dataset = config.dataset

        complete_data, train_missed_data, query_missed_data, retrieval_missed_data = load_data(self.dataset, self.alpha_train, self.beta_train, self.alpha_query, self.beta_query, self.alpha_retrieval, self.beta_retrieval)

        # data to GPU
        train_missed_data['I_dual_img'] = torch.Tensor(train_missed_data['I_dual_img']).cuda()
        train_missed_data['I_dual_txt'] = torch.Tensor(train_missed_data['I_dual_txt']).cuda()
        train_missed_data['I_dual_label'] = torch.Tensor(train_missed_data['I_dual_label']).cuda()
        train_missed_data['I_oimg'] = torch.Tensor(train_missed_data['I_oimg']).cuda()
        train_missed_data['I_oimg_label'] = torch.Tensor(train_missed_data['I_oimg_label']).cuda()
        train_missed_data['I_otxt'] = torch.Tensor(train_missed_data['I_otxt']).cuda()
        train_missed_data['I_otxt_label'] = torch.Tensor(train_missed_data['I_otxt_label']).cuda()
        complete_data['I_tr'] = torch.Tensor(complete_data['I_tr']).cuda()
        complete_data['T_tr'] = torch.Tensor(complete_data['T_tr']).cuda()
        complete_data['L_tr'] = torch.Tensor(complete_data['L_tr']).cuda()

        self.config = config
        self.logger = logger
        self.EPOCHS = config.epochs
        self.WU_EPOCHS = config.warmup_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.lr_warm = config.lr_warm
        self.nbit = config.nbit
        self.image_hidden_dim = config.image_hidden_dim
        self.text_hidden_dim = config.text_hidden_dim
        self.fusion_dim = config.fusion_dim
        self.run_times = config.run_times
        self.completion_hidden_dim = config.completion_hidden_dim
        self.map_dim = config.map_dim
        self.memory_size = config.memory_size
        self.memory_dim = config.memory_dim

        ########################################### data ###########################################
        self.train_data = [complete_data['I_tr'], complete_data['T_tr']]
        self.train_labels = complete_data['L_tr']

        # self.retrieval_data = [complete_data['I_db'], complete_data['T_db']]
        # self.retrieval_labels = complete_data['L_db'].numpy()

        # train missed data
        self.train_dual_data = [train_missed_data['I_dual_img'], train_missed_data['I_dual_txt']]
        self.train_dual_labels = train_missed_data['I_dual_label']
        self.train_only_imgs = train_missed_data['I_oimg']
        self.train_only_imgs_labels = train_missed_data['I_oimg_label']
        self.train_only_txts = train_missed_data['I_otxt']
        self.train_only_txts_labels = train_missed_data['I_otxt_label']

        # query missed data
        self.query_dual_data = [query_missed_data['I_dual_img'], query_missed_data['I_dual_txt']]
        self.query_only_imgs = query_missed_data['I_oimg'] 
        self.query_only_txts = query_missed_data['I_otxt']  
        self.query_labels = torch.cat((query_missed_data['I_dual_label'], query_missed_data['I_oimg_label'], query_missed_data['I_otxt_label'])).numpy()
        
        # retrieval missed data
        self.retrieval_dual_data = [retrieval_missed_data['I_dual_img'], retrieval_missed_data['I_dual_txt']]
        self.retrieval_only_imgs = retrieval_missed_data['I_oimg'] 
        self.retrieval_only_txts = retrieval_missed_data['I_otxt']  
        self.retrieval_labels = torch.cat((retrieval_missed_data['I_dual_label'], retrieval_missed_data['I_oimg_label'], retrieval_missed_data['I_otxt_label'])).numpy()

        self.train_nums = self.train_labels.shape[0]
        self.train_dual_nums = self.train_dual_data[0].size(0)
        self.train_only_imgs_nums = self.train_only_imgs.size(0)
        self.train_only_txts_nums = self.train_only_txts.size(0)
        assert self.train_nums == (self.train_dual_nums + self.train_only_imgs_nums + self.train_only_txts_nums)

        self.batch_dual_size = math.ceil(self.batch_size * (1 - self.alpha_train))
        self.batch_img_size = math.floor(self.batch_size * self.alpha_train * self.beta_train)
        self.batch_txt_size = self.batch_size - self.batch_dual_size - self.batch_img_size
        assert self.batch_txt_size >= 0

        self.img_dim = self.train_data[0].size(1)
        self.txt_dim = self.train_data[1].size(1)
        self.num_classes = self.train_labels.shape[1]
        self.logger.info('Dataset-%s has %d classes!' % (self.dataset, self.num_classes))
        self.logger.info('img_dim: %s, txt_dim: %s' % (self.img_dim, self.txt_dim))

        self.image_hidden_dim.insert(0, self.img_dim)
        self.image_hidden_dim.append(self.fusion_dim)
        self.text_hidden_dim.insert(0, self.txt_dim)
        self.text_hidden_dim.append(self.fusion_dim)

        self.path = 'model/checkpoint' + '_' + self.dataset + '_' + str(self.nbit) + '.pkl'

        #################################### model define ##########################################
        # modal completion
        self.txt_completion = model.TextCompletionModel(self.img_dim, self.map_dim, self.completion_hidden_dim, self.memory_dim, self.txt_dim, self.memory_size)
        self.img_completion = model.ImageCompletionModel(self.txt_dim, self.map_dim, self.completion_hidden_dim, self.memory_dim, self.img_dim, self.memory_size)
        # specific hash function
        self.img_mlp_enc = model.MLP(units=self.image_hidden_dim)
        self.txt_mlp_enc = model.MLP(units=self.text_hidden_dim)

        self.fusion_model = model.Fusion(fusion_dim=self.fusion_dim, nbit=self.nbit)

        # checkpoint = torch.load(self.path)
        # self.txt_completion.load_state_dict(checkpoint['txt_completion'])
        # self.img_completion.load_state_dict(checkpoint['img_completion'])
        # self.img_mlp_enc.load_state_dict(checkpoint['img_mlp_enc'])
        # self.txt_mlp_enc.load_state_dict(checkpoint['txt_mlp_enc'])
        # self.fusion_model.load_state_dict(checkpoint['fusion_model'])

        if torch.cuda.is_available():
            self.img_mlp_enc.cuda(), self.txt_mlp_enc.cuda()
            self.txt_completion.cuda(), self.img_completion.cuda()
            self.fusion_model.cuda()

        ################################# criterion define #########################################
        self.reconstruction_criterion = nn.MSELoss()

        ################################# optimizer define #########################################
        self.optimizer = torch.optim.Adam(
            [{"params": self.img_mlp_enc.parameters(), "lr": self.lr},
             {"params": self.txt_mlp_enc.parameters(), "lr": self.lr},
             {"params": self.fusion_model.parameters(), "lr": self.lr}])

        self.completion_optimizer = torch.optim.Adam(
            [{"params": self.img_completion.parameters(), "lr": self.lr_warm},
             {"params": self.txt_completion.parameters(), "lr": self.lr_warm}])

        
        ################################# hyper-parameter define ####################################
        self.param_intra = config.param_intra
        self.param_inter = config.param_inter
        self.param_sim = config.param_sim
        self.param_sign = config.param_sign

        self.batch_count = int(math.ceil(self.train_nums / self.batch_size))
    
    def warmup(self):
        self.img_completion.train(), self.txt_completion.train()

        self.train_loader = data.DataLoader(TrainCoupledData(self.train_data[0], self.train_data[1], self.train_labels),
                                                             batch_size=self.batch_size, shuffle=True) 

        for epoch in range(self.WU_EPOCHS):
            for batch_idx, (img_forward, txt_forward, label) in enumerate(self.train_loader):
                img_forward = img_forward.cuda()
                txt_forward = txt_forward.cuda()
                label = label.cuda()
                self.completion_optimizer.zero_grad()

                label_sim = utils.GEN_S_GPU(label, label)

                img_recons = self.img_completion(txt_forward)
                txt_recons = self.txt_completion(img_forward)

                img_forward_norm = F.normalize(img_forward)
                img_recons_norm = F.normalize(img_recons)
                txt_forward_norm = F.normalize(txt_forward)
                txt_recons_norm = F.normalize(txt_recons)
                img_forward_recons_sim = img_recons_norm.mm(img_forward_norm.t())
                img_recons_sim = img_recons_norm.mm(img_recons_norm.t())
                txt_forward_recons_sim = txt_recons_norm.mm(txt_forward_norm.t())
                txt_recons_sim = txt_recons_norm.mm(txt_recons_norm.t())
                img_forward_recons_sim = (img_forward_recons_sim - torch.min(img_forward_recons_sim)) / (torch.max(img_forward_recons_sim) - torch.min(img_forward_recons_sim))
                img_recons_sim = (img_recons_sim - torch.min(img_recons_sim)) / (torch.max(img_recons_sim) - torch.min(img_recons_sim))
                txt_forward_recons_sim = (txt_forward_recons_sim - torch.min(txt_forward_recons_sim)) / (torch.max(txt_forward_recons_sim) - torch.min(txt_forward_recons_sim))
                txt_recons_sim = (txt_recons_sim - torch.min(txt_recons_sim)) / (torch.max(txt_recons_sim) - torch.min(txt_recons_sim))


                LOSS1 = self.reconstruction_criterion(img_recons, img_forward) + \
                       self.reconstruction_criterion(txt_recons, txt_forward)
                LOSS2 = self.reconstruction_criterion(label_sim, img_forward_recons_sim) + self.reconstruction_criterion(label_sim, img_recons_sim)
                LOSS3 = self.reconstruction_criterion(label_sim, txt_forward_recons_sim) + self.reconstruction_criterion(label_sim, txt_recons_sim)
                
                LOSS = LOSS1 + LOSS2 + LOSS3

                LOSS.backward()
                self.completion_optimizer.step()

                if batch_idx == 0:
                    self.logger.info('[%4d/%4d] (Warm-up) Loss: %.4f' % (epoch + 1, self.WU_EPOCHS, LOSS.item()))

    def train(self):
        self.img_mlp_enc.train(), self.txt_mlp_enc.train()
        self.img_completion.train(), self.txt_completion.train()
        self.fusion_model.train()
        maps = [0]
        losses = [0.0]
        best_map = 0.
        best_n = 0
        best_query_code = None
        best_retrieval_code = None
        best_query_labels = None
        best_retrieval_labels = None
        for epoch in range(self.EPOCHS):
            epoch_loss = 0.
            iterations = 0
            dual_idx = torch.randperm(self.train_dual_nums).cuda()
            oimg_idx = torch.randperm(self.train_only_imgs_nums).cuda()
            otxt_idx = torch.randperm(self.train_only_txts_nums).cuda()

            for batch_idx in range(self.batch_count):
                small_dual_idx = dual_idx[batch_idx * self.batch_dual_size: (batch_idx + 1) * self.batch_dual_size]
                small_oimg_idx = oimg_idx[batch_idx * self.batch_img_size: (batch_idx + 1) * self.batch_img_size]
                small_otxt_idx = otxt_idx[batch_idx * self.batch_txt_size: (batch_idx + 1) * self.batch_txt_size]

                train_dual_img = self.train_dual_data[0][small_dual_idx, :]
                train_dual_txt = self.train_dual_data[1][small_dual_idx, :]
                train_dual_labels = self.train_dual_labels[small_dual_idx, :]

                train_only_img = self.train_only_imgs[small_oimg_idx, :]
                train_only_img_labels = self.train_only_imgs_labels[small_oimg_idx, :]
   
                train_only_txt = self.train_only_txts[small_otxt_idx, :]
                train_only_txt_labels = self.train_only_txts_labels[small_otxt_idx, :]

                loss = self.trainstep(train_dual_img, train_dual_txt, train_dual_labels, 
                                      train_only_img, train_only_img_labels, train_only_txt, train_only_txt_labels)
                epoch_loss = epoch_loss + loss
                iterations += 1
                if (batch_idx + 1) == self.batch_count:
                    self.logger.info('[%4d/%4d] Loss: %.4f' % (epoch + 1, self.EPOCHS, loss))
            if (epoch + 1) % 2 == 0:
                best_n += 1
                # self.test()
                result_map, query_code, retrieval_code, query_labels, retrieval_labels = self.test()
                maps.append(result_map)
                losses.append(epoch_loss / iterations)
                if result_map > best_map:
                    best_map = result_map
                    # best_query_code = query_code
                    # best_retrieval_code = retrieval_code
                    # best_query_labels = query_labels
                    # best_retrieval_labels = retrieval_labels
                    best_n = 0
                if best_n > 30:
                    break
        self.logger.info('Best Map: %f' % (best_map))
        # with open('../../baseline/map/map_ours_' + self.dataset + '_' + str(self.nbit) + 'bits.pkl', 'wb') as f:
        #     pickle.dump({'map': best_map}, f)
        # with open('./Hashcode/hashcode_ours_' + self.dataset + '_' + str(self.nbit) + 'bits.pkl', 'wb') as f:
        #     pickle.dump({'qB': best_query_code, 'rB': best_retrieval_code, 'qL': best_query_labels, 'rL': best_retrieval_labels}, f)
        # recall, precision = utils.pr_curve(best_query_code, best_retrieval_code, best_query_labels, best_retrieval_labels)
        # with open('../../baseline/pr/pr_ours_' + self.dataset + '_' + str(self.nbit) + 'bits.pkl', 'wb') as f:
        #     pickle.dump({'recall': recall, 'precision': precision}, f)
        # precision = utils.precision_topn(best_query_code, best_retrieval_code, best_query_labels, best_retrieval_labels, 1000)
        # with open('../../baseline/p/precision_ours_' + self.dataset + '_' + str(self.nbit) + 'bits.pkl', 'wb') as f:
        #     pickle.dump({'precision': precision}, f)
        # with open('/home/ljy/baseline/res/our_map_loss_' + self.dataset + '.pkl', 'wb') as f:
        #     pickle.dump({'map': maps, 'loss': losses}, f)        
        # torch.save({'img_completion': self.img_completion.state_dict(), 
        #             'txt_completion': self.txt_completion.state_dict(), 
        #             'img_mlp_enc': self.img_mlp_enc.state_dict(), 
        #             'txt_mlp_enc': self.txt_mlp_enc.state_dict(), 
        #             'fusion_model': self.fusion_model.state_dict()}, self.path)
    
    def trainstep(self, train_dual_img, train_dual_txt, train_dual_labels,
                  train_only_img, train_only_img_labels, train_only_txt, train_only_txt_labels):

        self.optimizer.zero_grad()

        dual_cnt = train_dual_labels.size(0) 
                                                                             
        img_forward = torch.cat([train_dual_img, train_only_img])
        txt_forward = torch.cat([train_dual_txt, train_only_txt])
        img_labels = torch.cat([train_dual_labels, train_only_img_labels])
        txt_labels = torch.cat([train_dual_labels, train_only_txt_labels])
        labels = torch.cat([train_dual_labels, train_only_img_labels, train_only_txt_labels])

        # construct similarity matrix
        label_sim = utils.GEN_S_GPU(labels, labels)
        img_label_sim = utils.GEN_S_GPU(img_labels, img_labels)
        txt_label_sim = utils.GEN_S_GPU(txt_labels, txt_labels)

        #### Forward
        # specific hash code
        img_feat = self.img_mlp_enc(img_forward)
        txt_feat = self.txt_mlp_enc(txt_forward)

        # generator + specific hash code
        img_recons = self.img_completion(txt_forward)
        txt_recons = self.txt_completion(img_forward)
        img_recons_norm = F.normalize(img_recons)
        txt_recons_norm = F.normalize(txt_recons)
        img_recons_sim = img_recons_norm.mm(img_recons_norm.t())
        txt_recons_sim = txt_recons_norm.mm(txt_recons_norm.t())

        img_recons_feat = self.img_mlp_enc(img_recons)
        txt_recons_feat = self.txt_mlp_enc(txt_recons)

        dual_repre = self.fusion_model(img_feat[:dual_cnt], txt_feat[:dual_cnt])
        oimg_repre = self.fusion_model(img_feat[dual_cnt:], txt_recons_feat[dual_cnt:])
        otxt_repre = self.fusion_model(img_recons_feat[dual_cnt:], txt_feat[dual_cnt:])
        
        total_repre = torch.cat([dual_repre, oimg_repre, otxt_repre])
        total_repre_norm = F.normalize(total_repre)
        B = torch.sign(total_repre)

        img_final_feat = torch.cat([img_feat[:dual_cnt], img_feat[dual_cnt:], img_recons_feat[dual_cnt:]])
        txt_final_feat = torch.cat([txt_feat[:dual_cnt], txt_recons_feat[dual_cnt:], txt_feat[dual_cnt:]])
        img_final_feat_norm = F.normalize(img_final_feat)
        txt_final_feat_norm = F.normalize(txt_final_feat)
        img_txt_sim = img_final_feat_norm.mm(txt_final_feat_norm.t())

        ##### loss function
        LOSS_sign = self.reconstruction_criterion(total_repre, B)

        LOSS_sim = self.reconstruction_criterion(total_repre_norm.mm(total_repre_norm.T), label_sim)

        LOSS_intra = self.reconstruction_criterion(img_label_sim, img_recons_sim) + self.reconstruction_criterion(txt_label_sim, txt_recons_sim)

        LOSS_inter = self.reconstruction_criterion(img_txt_sim, label_sim)

        LOSS = LOSS_sign * self.param_sign + LOSS_sim * self.param_sim + LOSS_intra * self.param_intra + LOSS_inter * self.param_inter

        LOSS.backward()
        self.optimizer.step()
        
        return LOSS.item()

    def test(self):
        self.logger.info('[TEST STAGE]')
        self.img_mlp_enc.eval(), self.txt_mlp_enc.eval()
        self.img_completion.eval(), self.txt_completion.eval()
        self.fusion_model.eval()
        
        self.logger.info('Retrieval Begin.')
        # retrieval set
        retrievalP = []

        with torch.no_grad():
            dual_img_feat = self.img_mlp_enc(self.retrieval_dual_data[0].cuda())
            dual_txt_feat = self.txt_mlp_enc(self.retrieval_dual_data[1].cuda())
            dualH = self.fusion_model(dual_img_feat, dual_txt_feat)
        retrievalP.append(dualH.data.cpu().numpy())

        with torch.no_grad():
            oimg_feat = self.img_mlp_enc(self.retrieval_only_imgs.cuda()) 
            oimg_Gtxt = self.txt_completion(self.retrieval_only_imgs.cuda())  
            oimg_Gtxt = self.txt_mlp_enc(oimg_Gtxt)  
            oimgH = self.fusion_model(oimg_feat, oimg_Gtxt)
        retrievalP.append(oimgH.data.cpu().numpy())

        with torch.no_grad():
            otxt_Gimg = self.img_completion(self.retrieval_only_txts.cuda())
            otxt_Gimg = self.img_mlp_enc(otxt_Gimg)
            otxt_feat = self.txt_mlp_enc(self.retrieval_only_txts.cuda()) 
            otxtH = self.fusion_model(otxt_Gimg, otxt_feat)
        retrievalP.append(otxtH.data.cpu().numpy())

        retrievalH = np.concatenate(retrievalP)
        self.retrieval_code = np.sign(retrievalH)
        self.logger.info('Retrieval End.')

        self.logger.info('Query Begin.')
        # query set
        queryP = []
        with torch.no_grad():
            dual_img_feat = self.img_mlp_enc(self.query_dual_data[0].cuda())
            dual_txt_feat = self.txt_mlp_enc(self.query_dual_data[1].cuda())
            dualH = self.fusion_model(dual_img_feat, dual_txt_feat)
        queryP.append(dualH.data.cpu().numpy())

        with torch.no_grad():
            oimg_feat = self.img_mlp_enc(self.query_only_imgs.cuda()) 
            oimg_Gtxt = self.txt_completion(self.query_only_imgs.cuda())  
            oimg_Gtxt = self.txt_mlp_enc(oimg_Gtxt)  
            oimgH = self.fusion_model(oimg_feat, oimg_Gtxt)
        queryP.append(oimgH.data.cpu().numpy())

        with torch.no_grad():
            otxt_Gimg = self.img_completion(self.query_only_txts.cuda())
            otxt_Gimg = self.img_mlp_enc(otxt_Gimg)
            otxt_feat = self.txt_mlp_enc(self.query_only_txts.cuda()) 
            otxtH = self.fusion_model(otxt_Gimg, otxt_feat)
        queryP.append(otxtH.data.cpu().numpy())

        queryH = np.concatenate(queryP)
        self.query_code = np.sign(queryH)
        self.logger.info('Query End.')
        
        assert self.retrieval_code.shape[0] == self.retrieval_labels.shape[0]
        assert self.query_code.shape[0] == self.query_labels.shape[0]
        
        map = utils.calc_map(self.query_code, self.retrieval_code, self.query_labels, self.retrieval_labels)
        self.logger.info('[%4d/%4d] Map: %f' % (self.running_cnt, self.run_times, map))
        self.logger.info("-----------------------------------------")
        return map, self.query_code, self.retrieval_code, self.query_labels, self.retrieval_labels
