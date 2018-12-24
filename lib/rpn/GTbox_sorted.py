# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# By Huijuan Xu
# --------------------------------------------------------


import caffe
import yaml
import numpy as np
import numpy.random as npr
from tdcnn.config import cfg
from tdcnn.twin_transform import twin_transform
from utils.cython_twin import twin_overlaps

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        # gt_rois
        top[0].reshape(1, 3)
        top[1].reshape(1, bottom[1].data.shape[1])

    def forward(self, bottom, top):
        gt_rois = bottom[0].data
        gt_features = bottom[1].data

        # zeros = np.zeros((gt_wins.shape[0], 1), dtype=gt_wins.dtype)
        # gt_rois = np.hstack((zeros, gt_wins[:, :-1]))
        ### sort according to rois[:,2], namely, sort according to acctivity end time                                                                 
        sort_end_ind = gt_rois[:,2].argsort()
        gt_rois = gt_rois[sort_end_ind,:]
        gt_features = gt_features[sort_end_ind,:]

        top[0].reshape(*gt_rois.shape)
        top[0].data[...] = gt_rois

        top[1].reshape(*gt_features.shape)
        top[1].data[...] = gt_features

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


