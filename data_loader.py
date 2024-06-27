import math
import os

import h5py
import numpy as np
import torch
import torch.utils.data as data

ROOT = ROOT = '/home/ljy/multi_modal_hashing/dataset/'
if not os.path.exists(ROOT):
    raise Exception('The ROOT path is error.')

paths = {
    'flickr': ROOT + 'mir_cnn_twt.mat',
    'nuswide': ROOT + 'nus_cnn_twt.mat',
    'coco': ROOT + 'coco_cnn_twt.mat'
}

def load_data(DATANAME, alpha_train=0.0, beta_train=0.5, alpha_query=0.0, beta_query=0.5, alpha_retrieval=0.0, beta_retrieval=0.5):
    data = h5py.File(paths[DATANAME], 'r')

    I_tr = torch.Tensor(data['I_tr'][:].T)
    T_tr = torch.Tensor(data['T_tr'][:].T)
    L_tr = torch.FloatTensor(data['L_tr'][:].T)

    I_db = torch.Tensor(data['I_db'][:].T)
    T_db = torch.Tensor(data['T_db'][:].T)
    L_db = torch.FloatTensor(data['L_db'][:].T)

    I_te = torch.Tensor(data['I_te'][:].T)
    T_te = torch.Tensor(data['T_te'][:].T)
    L_te = torch.FloatTensor(data['L_te'][:].T)
    
    complete_data = {'I_tr': I_tr, 'T_tr': T_tr, 'L_tr': L_tr, 
                     'I_db': I_db, 'T_db': T_db, 'L_db': L_db, 
                     'I_te': I_te, 'T_te': T_te, 'L_te': L_te}

    # construct missed data
    train_missed_data = construct_missed_data(I_tr, T_tr, L_tr, alpha=alpha_train, beta=beta_train)
    query_missed_data = construct_missed_data(I_te, T_te, L_te, alpha=alpha_query, beta=beta_query)
    retrieval_missed_data = construct_missed_data(I_db, T_db, L_db, alpha=alpha_retrieval, beta=beta_retrieval)

    return (complete_data, train_missed_data, query_missed_data, retrieval_missed_data)

def construct_missed_data(I_tr, T_tr, L_tr, alpha=0.0, beta=0.5):
    number = I_tr.size(0) 
    # dual_size = math.ceil(number * (1 - alpha))
    only_image_size = math.floor(number * alpha * beta)
    only_text_size = math.floor(number * alpha * (1 - beta))
    dual_size = number - only_image_size - only_text_size
    # only_text_size = number - dual_size - only_image_size
    print('Dual size: %d, Oimg size: %d, Otxt size: %d' % (dual_size, only_image_size, only_text_size))
    
    random_idx = np.random.permutation(number)

    dual_index = random_idx[:dual_size]
    only_image_index = random_idx[dual_size:dual_size+only_image_size]
    only_text_index = random_idx[dual_size+only_image_size:dual_size+only_image_size+only_text_size]

    I_dual_img = I_tr[dual_index, :]
    I_dual_txt = T_tr[dual_index, :]
    I_dual_label = L_tr[dual_index, :]
    
    I_oimg = I_tr[only_image_index, :]
    I_oimg_label = L_tr[only_image_index, :]

    I_otxt = T_tr[only_text_index, :]
    I_otxt_label = L_tr[only_text_index, :]

    _dict = {'I_dual_img': I_dual_img, 'I_dual_txt': I_dual_txt, 'I_dual_label': I_dual_label, 
             'I_oimg': I_oimg, 'I_oimg_label': I_oimg_label, 'I_otxt': I_otxt, 'I_otxt_label': I_otxt_label}
    return _dict

def construct_train_missed_data(I_tr, T_tr, L_tr, alpha=0.0, beta=0.5):
    number = I_tr.size(0) 
    dual_size = math.ceil(number * (1 - alpha))
    only_image_size = math.floor(number * alpha * beta)
    only_text_size = number - dual_size - only_image_size
    augment_image_size = math.ceil(dual_size * 0.5)
    augment_text_size = dual_size - augment_image_size
    print('Dual size: %d, Oimg size: %d, Otxt size: %d, Aimg size: %d, Atxt size: %d' % (dual_size, only_image_size, only_text_size, augment_image_size, augment_text_size))
    
    random_idx = np.random.permutation(number)
    augment_idx = np.random.permutation(dual_size)

    dual_index = random_idx[:dual_size]
    augment_image_index = augment_idx[:augment_image_size]
    augment_text_index = augment_idx[augment_image_size:]
    if only_image_size > only_text_size:
        only_image_index = random_idx[dual_size:dual_size+only_image_size-1]
        only_text_index = random_idx[dual_size+only_image_size:dual_size+only_image_size+only_text_size]
    elif only_image_size < only_text_size:
        only_image_index = random_idx[dual_size:dual_size+only_image_size]
        only_text_index = random_idx[dual_size+only_image_size:dual_size+only_image_size+only_text_size-1]
    else:
        only_image_index = random_idx[dual_size:dual_size+only_image_size]
        only_text_index = random_idx[dual_size+only_image_size:dual_size+only_image_size+only_text_size]

    I_dual_img = I_tr[dual_index, :]
    I_dual_txt = T_tr[dual_index, :]
    I_dual_label = L_tr[dual_index, :]

    I_aimg = I_dual_img[augment_image_index, :]
    I_aimg_label = I_dual_label[augment_image_index, :]

    I_atxt = I_dual_txt[augment_text_index, :]
    I_atxt_label = I_dual_label[augment_text_index, :]
    
    I_oimg = I_tr[only_image_index, :]
    I_oimg_label = L_tr[only_image_index, :]

    I_otxt = T_tr[only_text_index, :]
    I_otxt_label = L_tr[only_text_index, :]

    I_oimg = torch.cat((I_oimg, I_aimg), dim=0)
    I_oimg_label = torch.cat((I_oimg_label, I_aimg_label), dim=0)

    I_otxt = torch.cat((I_otxt, I_atxt), dim=0)
    I_otxt_label = torch.cat((I_otxt_label, I_atxt_label), dim=0)

    _dict = {'I_dual_img': I_dual_img, 'I_dual_txt': I_dual_txt, 'I_dual_label': I_dual_label, 
             'I_oimg': I_oimg, 'I_oimg_label': I_oimg_label, 'I_otxt': I_otxt, 'I_otxt_label': I_otxt_label}
    return _dict

class CoupledData(data.Dataset):
    def __init__(self, img_feature, txt_feature):
        self.img_feature = img_feature
        self.txt_feature = txt_feature
        self.length = self.img_feature.size(0)

    def __getitem__(self, item):
        return self.img_feature[item, :], self.txt_feature[item, :]

    def __len__(self):
        return self.length

    
class TrainCoupledData(data.Dataset):
    def __init__(self, img_feature, txt_feature, label):
        self.img_feature = img_feature
        self.txt_feature = txt_feature
        self.label = label
        self.length = self.img_feature.size(0)

    def __getitem__(self, item):
        return self.img_feature[item, :], self.txt_feature[item, :], self.label[item, :]

    def __len__(self):
        return self.length