import logging
import os
import random
import time
import math

import numpy as np
import scipy.spatial
import scipy.io as sio
import torch
import torch.nn.functional as F

def zero2eps(x):
    x[x == 0] = 1
    return x

def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum # row data sum = 1
    in_affnty = np.transpose(affinity/row_sum) # col data sum = 1 then transpose
    return in_affnty, out_affnty

def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False # False make training process too slow!
    torch.backends.cudnn.deterministic = True

def logger():
    '''
    '\033[0;34m%s\033[0m': blue
    :return:
    '''
    logger = logging.getLogger('PAGN')
    logger.setLevel(logging.DEBUG)

    if not os.path.exists('log/'):
        os.mkdir('log/')

    timeStr = time.strftime("[%m-%d]%H:%M:%S", time.localtime())

    txt_log = logging.FileHandler('log/'+ timeStr +'.log', mode='a')
    txt_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', '%m/%d %H:%M:%S')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    # console + color
    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('\033[0;32m%s\033[0m' % '[%(asctime)s][%(levelname)s] %(message)s', '%m/%d %H:%M:%S')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger

def log_params(logger, config: dict):
    logger.info('--- Configs List---')
    for k in config.keys():
        logger.info('--- {:<18}:{}'.format(k, config[k]))

def GEN_S_GPU(label_1, label_2):
    aff = torch.matmul(label_1, label_2.T)
    affinity_matrix = aff.float()
    affinity_matrix = 1 / (1 + torch.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    return affinity_matrix

def int2bool(flag: int):
    '''

    :param flag: -1: False // 1: True
    :return:
    '''
    return False if flag == -1 else True

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_map(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(qB.shape)
    print(rB.shape)
    print(queryL.shape)
    print(retrievalL.shape)

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = int(np.sum(gnd))
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map

def pr_curve(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    dim = np.shape(rB)
    bit = dim[1]
    all_ = dim[0]
    precision = np.zeros(bit + 1)
    recall = np.zeros(bit + 1)
    num_query = queryL.shape[0]
    num_database = retrievalL.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        all_sum = np.sum(gnd).astype(np.float32)
        # print(all_sum)
        if all_sum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        # print(hamm.shape)
        ind = np.argsort(hamm)
        # print(ind.shape)
        gnd = gnd[ind]
        hamm = hamm[ind]
        hamm = hamm.tolist()
        # print(len(hamm), num_database - 1)
        max_ = hamm[num_database - 1]
        max_ = int(max_)
        t = 0
        for i in range(1, max_):
            if i in hamm:
                idd = hamm.index(i)
                if idd != 0:
                    sum1 = np.sum(gnd[:idd])
                    precision[t] += sum1 / idd
                    recall[t] += sum1 / all_sum
                else:
                    precision[t] += 0
                    recall[t] += 0
                t += 1
        # precision[t] += all_sum / num_database
        # recall[t] += 1
        for i in range(t,  bit + 1):
            precision[i] += all_sum / num_database
            recall[i] += 1
    true_recall = recall / num_query
    precision = precision / num_query
    print(true_recall)
    print(precision)
    return true_recall, precision

def precision_topn(qB, rB, queryL, retrievalL, topk=1000):
    n = topk // 100
    precision = np.zeros(n)
    num_query = queryL.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        for i in range(1, n + 1):
            a = gnd[:i * 100]
            precision[i - 1] += float(a.sum()) / (i * 100.)
    a_precision = precision / num_query
    return a_precision

