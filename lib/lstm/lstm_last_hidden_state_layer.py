# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# By Huijuan Xu
# --------------------------------------------------------


import caffe
import numpy as np
import yaml

class LSTMLastLayer(caffe.Layer):
    """extract last hidden state."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        top_shape = bottom[0].data.shape
        top[0].reshape(1,top_shape[1],top_shape[2])

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""

        lstm = bottom[0].data
        cnt = bottom[1].data

        self.indices_ = cnt.sum(axis = 0)
        blob = np.zeros((1, lstm.shape[1],lstm.shape[2]))

        for k, idx in enumerate(self.indices_):
            #print k, int(idx)
            blob[0,k,:] = lstm[int(idx), k, :]

        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        if propagate_down[0]:
          diff = np.zeros_like(bottom[0].diff)
          for k, idx in enumerate(self.indices_):
            diff[int(idx), k, :] = top[0].diff[0, k, :]
          bottom[0].diff[...] = diff.astype(np.float32, copy=False)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

