# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# By Huijuan Xu
# --------------------------------------------------------


import caffe
import numpy as np
import yaml
from tdcnn.config import cfg
from generate_anchors import generate_anchors
from tdcnn.twin_transform import twin_transform_inv, clip_wins
from tdcnn.nms_wrapper import nms

DEBUG = False

class CaptionLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular wins (called "anchors").
    """

    def setup(self, bottom, top):
        # rois blob: holds R regions of interest, each is a 3-tuple
        # (n, x1, x2) specifying an video batch index n and a
        # rectangle (x1, x2)
        top[0].reshape(1, 3)  

        # fc6
        top[1].reshape(1, bottom[3].data.shape[1])

        # scores blob: holds scores for R regions of interest
        if len(top) > 2:
            top[2].reshape(1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor wins centered on cell i
        #   apply predicted twin deltas at cell i to each of the A anchors
        # clip predicted wins to video
        # remove predicted wins with length < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        cfg_key = 'TRAIN' if self.phase == 0 else 'TEST'
        topN      = cfg[cfg_key].CAPTION_TOP_N
        min_size  = cfg[cfg_key].CAPTION_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, 1]
        twin_deltas = bottom[1].data[:,2:]
        twins = bottom[2].data[:,1:]
        fc_features = bottom[3].data

        # Convert anchors into proposals via twin transformations
        proposals = twin_transform_inv(twins, twin_deltas)

        # 2. clip predicted wins to video
        length = cfg.TRAIN.LENGTH[0] if cfg_key == 'TRAIN' else cfg.TEST.LENGTH[0]
        proposals = clip_wins(proposals, length)

        # 3. remove predicted wins with either height or width < threshold
        # (NOTE: convert min_size to input video scale stored in im_info[2])
        keep = _filter_wins(proposals, min_size)
        proposals = proposals[keep, :]
        scores = scores[keep]
        fc_features = fc_features[keep]
        self.keep_ = keep

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top topN (e.g. 32)
        order = scores.ravel().argsort()[::-1]
        if topN > 0:
            order = order[:topN]
        else:
            order = order[:1]  ### this is an approximate solution, get top1, since 0 proposal can't be forwarded.....
        proposals = proposals[order, :]
        scores = scores[order]
        fc_features = fc_features[order]
        self.order_ = order

        # Output rois blob
        # Our RPN implementation only supports a single input video, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        top[1].reshape(*(fc_features.shape))
        top[1].data[...] = fc_features

        # [Optional] output scores blob
        if len(top) > 2:
            top[2].reshape(*(scores.shape))
            top[2].data[...] = scores

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        tmp = np.zeros_like(bottom[3].data)
        bottom[3].diff[...] = np.zeros_like(bottom[3].data)
        for i, idx in enumerate(self.order_):
          tmp[idx, ...] = top[1].diff[i]
        for i, idx in enumerate(self.keep_):
          bottom[3].diff[idx, ...] = tmp[i]

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_wins(wins, min_size):
    """Remove all wins with any side smaller than min_size."""
    ls = wins[:, 1] - wins[:, 0] + 1
    keep = np.where(ls >= min_size)[0]
    return keep
