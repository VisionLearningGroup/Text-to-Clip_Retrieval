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
        shape = (1,bottom[0].data.shape[1],bottom[0].data.shape[2])
        top[0].reshape(*shape)

    def forward(self, bottom, top):
        # captions
        controller_hidden = bottom[0].data
        controller_cont = bottom[1].data
        self._num = int(controller_cont.sum() + 1)
        # input captions
        output_hidden = controller_hidden[:self._num, :]
        top[0].reshape(*output_hidden.shape)
        top[0].data[...] = output_hidden

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        bottom[0].diff[:self._num,:] = top[0].diff

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
