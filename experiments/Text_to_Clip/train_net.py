# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# By Huijuan Xu
# --------------------------------------------------------

#!/usr/bin/env python


import _init_paths
import caffe
import argparse
import pprint
import numpy as np
import sys
import cPickle
import copy
import h5py

from tdcnn.config import cfg, cfg_from_file, cfg_from_list
from tdcnn.train_retrieval_pairwiserank_adam_caption import train_net


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Text-to-Clip Retrieval network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=550000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_train_roidb(path):
    data = cPickle.load(open(path + 'train_data_modified_5fps_flipped_caption_768.pkl'))
    return data

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()


    ### load HDF5 feature and reshape
    HDF_filelist = "............../experiments/extract_HDF_for_LSTM/hdf5data/hdf5_chunk_list.txt"
    f = open(HDF_filelist,'r')
    hdf_files = f.readlines()
    f.close()

    all_fc6 = np.empty((0,4096), dtype=np.float32)
    all_target_sent_reshaped = np.empty((0,cfg.MAX_WORDS), dtype=np.int32)
    all_input_sent_reshaped = np.empty((0,cfg.MAX_WORDS), dtype=np.int32)
    all_cont_sent_reshaped = np.empty((0,cfg.MAX_WORDS), dtype=np.int32)
    for ff in hdf_files:
        file_tmp = ff[:-1]
        f = h5py.File(file_tmp, 'r')
        all_fc6 = np.vstack((all_fc6, f['all_fc6_batched']))
        tmp_target = f['all_target_sent_batched'][:]
        tmp_input = f['all_input_sent_batched'][:]
        tmp_cont = f['all_cont_sent_batched'][:]
        f.close()        
        num_batches = tmp_target.shape[0]/cfg.MAX_WORDS
        for n in xrange(num_batches):
            all_target_sent_reshaped = np.vstack((all_target_sent_reshaped, tmp_target[cfg.MAX_WORDS*n:cfg.MAX_WORDS*(n+1),:].transpose((1, 0)) ))
            all_input_sent_reshaped = np.vstack((all_input_sent_reshaped, tmp_input[cfg.MAX_WORDS*n:cfg.MAX_WORDS*(n+1),:].transpose((1, 0)) ))
            all_cont_sent_reshaped = np.vstack((all_cont_sent_reshaped, tmp_cont[cfg.MAX_WORDS*n:cfg.MAX_WORDS*(n+1),:].transpose((1, 0)) ))



    output_dir = './experiments/Text_to_Clip/snapshot/'
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, all_fc6, all_target_sent_reshaped, all_input_sent_reshaped, all_cont_sent_reshaped, output_dir, pretrained_model=args.pretrained_model, max_iters=args.max_iters)

