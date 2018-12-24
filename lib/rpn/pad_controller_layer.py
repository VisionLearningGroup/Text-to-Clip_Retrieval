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

DEBUG = False

class CaptionTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)

        # input captions
        #top_shape = (29,bottom[0].data.shape[1],bottom[0].data.shape[2])
        top_shape = (29,bottom[0].data.shape[1],bottom[0].data.shape[2])
        top[0].reshape(*top_shape)

    def forward(self, bottom, top):
        # captions
        controller = bottom[0].data
        controller_cont = bottom[1].data
        pad_num = controller_cont.shape[0]-controller.shape[0]
        pad_vec = np.zeros((pad_num, controller.shape[1], controller.shape[2]))
        controller_pad = np.vstack((controller, pad_vec))

        # input captions
        top[0].reshape(*controller_pad.shape)
        top[0].data[...] = controller_pad

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        bottom[0].diff[...] = top[0].diff[:bottom[0].data.shape[0],]

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
