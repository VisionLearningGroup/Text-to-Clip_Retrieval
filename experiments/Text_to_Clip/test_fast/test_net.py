# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# By Huijuan Xu
# --------------------------------------------------------


#!/usr/bin/env python


"""Test a Text-to-Clip Retrieval network on an video database."""

import _init_paths
from tdcnn.test_retrieval_caption_share_fast import test_net
from tdcnn.config import cfg, cfg_from_file, cfg_from_list
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
import random
import copy
import cPickle

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Text-to-Clip Retrieval network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--def-lstm', dest='lstm_prototxt',
                        help='prototxt file defining the fc embedding',
                        default=None, type=str)
    parser.add_argument('--def-retrieval', dest='retrieval_prototxt',
                        help='prototxt file defining the word embedding',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--netRetrieval', dest='Retrieval_caffemodel',
                        help='retrieval model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_test_roidb(path):
    test_data = cPickle.load(open(path + 'test_data_modified_5fps_caption_768_1_full_retrieval.pkl'))
    return test_data

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

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    FcEmbed_net = caffe.Net(args.lstm_prototxt, args.Retrieval_caffemodel, caffe.TEST)
    WordEmbed_net = caffe.Net(args.retrieval_prototxt, args.Retrieval_caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    path = './experiments/preprocess/'
    imdb = get_test_roidb(path)
    
    vocabulary = ['<EOS>'] + [line.strip() for line in open(path+'vocabulary.txt').readlines()]
    print len(vocabulary)


    parsed_iter=args.Retrieval_caffemodel.split('.')[-2].split('_')[-1]
    sim_pickle_path = './experiments/Text_to_Clip/test_fast/sim_pickle/sim_iter_'+parsed_iter+'.p'

    test_net(net, imdb, FcEmbed_net, WordEmbed_net, sim_pickle_path, vocabulary, max_per_image=args.max_per_image, thresh=0.005, vis=args.vis)


