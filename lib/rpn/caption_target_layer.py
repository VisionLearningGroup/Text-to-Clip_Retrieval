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

class CaptionTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)

        # input captions
        top[0].reshape(cfg.MAX_WORDS, 1)
        # count
        top[1].reshape(cfg.MAX_WORDS, 1)
        # output sentence
        top[2].reshape(cfg.MAX_WORDS, 1)
        # fc
        top[3].reshape(1, bottom[3].data.shape[1])

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, x2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT wins (x1, x2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_wins = bottom[1].data
        # captions
        captions = bottom[2].data
        # fc
        fc_features = bottom[3].data
        gt_features = bottom[4].data

        # Include ground-truth wins in the set of candidate rois
        zeros = np.zeros((gt_wins.shape[0], 1), dtype=gt_wins.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_wins[:, :-1])))
        )
        fc_features = np.vstack((fc_features, gt_features))

        # Sample rois with classification labels and bounding box regression
        # targets
        cont_sent, input_sent, output_sent, fc_features, rois, keep_inds = _sample_positive_rois(
            all_rois, gt_wins, captions, fc_features)

        self.keep_inds_ = keep_inds

        # classification labels
        top[0].reshape(*input_sent.shape)
        top[0].data[...] = input_sent

        # twin_targets
        top[1].reshape(*cont_sent.shape)
        top[1].data[...] = cont_sent

        # twin_inside_weights
        top[2].reshape(*output_sent.shape)
        top[2].data[...] = output_sent

        # fc_features
        top[3].reshape(*fc_features.shape)
        top[3].data[...] = fc_features

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        bottom[3].diff[...] = np.zeros_like(bottom[3].data)
        bottom[4].diff[...] = np.zeros_like(bottom[4].data)
        size = bottom[3].data.shape[0]
        for i, idx in enumerate(self.keep_inds_):
           if idx < size:
             bottom[3].diff[idx,...] = top[3].diff[i] 
           else:
             bottom[4].diff[idx - size, ...] = top[3].diff[i]

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_twin_regression_labels(twin_target_data, num_classes):
    """Bounding-box regression targets (twin_target_data) are stored in a
    compact form N x (class, tx, tl)

    This function expands those targets into the 4-of-2*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        twin_target (ndarray): N x 4K blob of regression targets
        twin_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = twin_target_data[:, 0]
    twin_targets = np.zeros((clss.size, 2 * num_classes), dtype=np.float32)
    twin_inside_weights = np.zeros(twin_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(2 * cls)
        end = start + 2
        twin_targets[ind, start:end] = twin_target_data[ind, 1:]
        twin_inside_weights[ind, start:end] = cfg.TRAIN.TWIN_INSIDE_WEIGHTS
    return twin_targets, twin_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 2
    assert gt_rois.shape[1] == 2

    targets = twin_transform(ex_rois, gt_rois)
    if cfg.TRAIN.TWIN_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.TWIN_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.TWIN_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_positive_rois(all_rois, gt_wins, captions, fc_features):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_wins)
    overlaps = twin_overlaps(
        np.ascontiguousarray(all_rois[:, 1:3], dtype=np.float),
        np.ascontiguousarray(gt_wins[:, :2], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    # labels = gt_wins[gt_assignment, 2]
    input_sent = captions[gt_assignment, 0, :].reshape((gt_assignment.shape[0],-1)).transpose((1,0))
    cont_sent = captions[gt_assignment, 1, :].reshape((gt_assignment.shape[0],-1)).transpose((1,0))
    target_sent = captions[gt_assignment, 2, :].reshape((gt_assignment.shape[0],-1)).transpose((1,0))

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.CAPTION_FG_THRESH)[0]   # __C.TRAIN.FG_THRESH = 0.5

   # The indices that we're selecting (fg)
    keep_inds = fg_inds
    rois = all_rois[keep_inds]
    fc_features = fc_features[keep_inds, :]
    input_sent = input_sent[:, keep_inds]
    cont_sent = cont_sent[:, keep_inds]
    target_sent = target_sent[:, keep_inds]


    return cont_sent, input_sent, target_sent, fc_features, rois, keep_inds
