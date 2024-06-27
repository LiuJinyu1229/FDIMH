import argparse
import time
import utils
import os
from engine import *

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def _main(config, logger, running_cnt):
    model = Engine(config=config, logger=logger, running_cnt=running_cnt)

    logger.info('===========================================================================')
    logger.info('Training stage!')
    start_time = time.time() * 1000
    model.warmup()
    model.train()
    train_time = time.time() * 1000 - start_time
    logger.info('Training time: %.6f' % (train_time / 1000))
    logger.info('===========================================================================')

    logger.info('===========================================================================')
    logger.info('Testing stage!')
    start_time = time.time() * 1000
    model.test()
    test_time = time.time() * 1000 - start_time
    logger.info('Testing time: %.6f' % (test_time / 1000))
    logger.info('===========================================================================')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco', help='Dataset: flickr/nuswide/coco')
    parser.add_argument('--alpha_train', type=float, default=0, help='Missing ratio of train set.')
    parser.add_argument('--alpha_query', type=float, default=0, help='Missing ratio of query set.')
    parser.add_argument('--alpha_retrieval', type=float, default=0, help='Missing ratio of retrieval set.')
    parser.add_argument('--beta_train', type=float, default=0.5, help='Missing ratio of image or test')
    parser.add_argument('--beta_query', type=float, default=0.5)
    parser.add_argument('--beta_retrieval', type=float, default=0.5)

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--warmup_epochs', type=int, default=1000)

    parser.add_argument('--lr', type=float, default=0.0006)
    parser.add_argument('--lr_warm', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--image_hidden_dim', type=list, default=[4096, 128], help='Construct imageMLP')
    parser.add_argument('--text_hidden_dim', type=list, default=[4096, 128], help='Construct textMLP')
    parser.add_argument('--fusion_dim', type=int, default=512)
    parser.add_argument('--nbit', type=int, default=32)
    
    parser.add_argument('--param_intra', type=float, default=0.1, help='IntraModality loss.')
    parser.add_argument('--param_inter', type=float, default=0.1, help='IntreModality loss.')
    parser.add_argument('--param_sim', type=float, default=1, help='Similarity loss.')
    parser.add_argument('--param_sign', type=float, default=0.1, help='Sign loss.')

    parser.add_argument('--run_times', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2025)

    parser.add_argument('--completion_hidden_dim', type=int, default=2048)
    parser.add_argument('--map_dim', type=int, default=2048)
    parser.add_argument('--memory_size', type=int, default=2048)
    parser.add_argument('--memory_dim', type=int, default=1000)

    logger = utils.logger()

    logger.info('===========================================================================')
    logger.info('Current File: {}'.format(__file__))
    config = parser.parse_args()
    utils.log_params(logger, vars(config))
    logger.info('===========================================================================')
    # utils.seed_setting(seed=config.seed) 
    for i in range(config.run_times):
        _main(config=config, logger=logger, running_cnt=i+1)