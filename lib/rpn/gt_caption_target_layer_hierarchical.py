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
        self._num_cont = layer_params['num_cont']

        # input captions
        top[0].reshape(cfg.MAX_WORDS, 1)
        # count
        top[1].reshape(cfg.MAX_WORDS, 1)
        # output sentence
        top[2].reshape(cfg.MAX_WORDS, 1)
        # controller cont
        top[3].reshape(self._num_cont, 1)

    def forward(self, bottom, top):
        # captions
        captions = bottom[0].data

        controller_cont = np.zeros((self._num_cont, 1))
        controller_cont[1:captions.shape[0]] = 1
        input_sent = captions[:, 0, :].transpose((1,0))
        cont_sent = captions[:, 1, :].transpose((1,0))
        target_sent = captions[:, 2, :].transpose((1,0))

        # input captions
        top[0].reshape(*input_sent.shape)
        top[0].data[...] = input_sent

        # count
        top[1].reshape(*cont_sent.shape)
        top[1].data[...] = cont_sent

        # output sentence
        top[2].reshape(*target_sent.shape)
        top[2].data[...] = target_sent

        # controller cont
        top[3].reshape(*controller_cont.shape)
        top[3].data[...] = controller_cont

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
