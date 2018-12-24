# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# By Huijuan Xu
# --------------------------------------------------------

"""The data layer used during training to train a Text-to-Clip Retrieval network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from tdcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue

class RoIDataLayer(caffe.Layer):
    """data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(self._fc6.shape[0]))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self._batch_size >= self._fc6.shape[0]:
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self._batch_size]
        self._cur += self._batch_size
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_fc6 = self._fc6[db_inds,:]
        minibatch_target_sent = self._target_sent[db_inds,:]
        minibatch_input_sent = self._input_sent[db_inds,:]
        minibatch_cont_sent = self._cont_sent[db_inds,:]

        #form paired index
        tmp=np.ones([self._batch_size, self._batch_size])
        index = np.where(tmp>0)
        fc6_paired = minibatch_fc6[index[0],:]
        input_sent_paired = minibatch_input_sent[index[1],:].T
        cont_sent_paired = minibatch_cont_sent[index[1],:].T
        
        target_sent_paired = np.ones_like(input_sent_paired) * -1
        for i in xrange(self._batch_size):
           target_sent_paired[:, i*self._batch_size+i] = minibatch_target_sent[i,:]

        return fc6_paired, target_sent_paired, input_sent_paired, cont_sent_paired

    def set_roidb(self, all_fc6, all_target_sent_reshaped, all_input_sent_reshaped, all_cont_sent_reshaped):
        """Set the roidb to be used by this layer during training."""
        self._fc6 = all_fc6
        self._target_sent = all_target_sent_reshaped                
        self._input_sent = all_input_sent_reshaped                
        self._cont_sent = all_cont_sent_reshaped                
        self._shuffle_roidb_inds()


    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        self._batch_size = layer_params['batch_size']
        self._name_to_top_map = {}

        # data blob: holds a batch of N videos, each with 3 channels
        idx = 0
        top[idx].reshape(self._batch_size, 4096)
        self._name_to_top_map['fc6'] = idx
        idx += 1
        top[idx].reshape(cfg.MAX_WORDS, self._batch_size)
        self._name_to_top_map['target_sent'] = idx
        idx += 1
        top[idx].reshape(cfg.MAX_WORDS, self._batch_size)
        self._name_to_top_map['input_sent'] = idx
        idx += 1
        top[idx].reshape(cfg.MAX_WORDS, self._batch_size)
        self._name_to_top_map['cont_sent'] = idx
                        
        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)



    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""

        minibatch_fc6, minibatch_target_sent, minibatch_input_sent, minibatch_cont_sent = self._get_next_minibatch()
        
        # Reshape net's input blobs
        # Copy data into net's input blobs
        top[0].reshape(*(minibatch_fc6.shape))
        top[0].data[...] = minibatch_fc6.astype(np.float32, copy=False)
        top[1].reshape(*(minibatch_target_sent.shape))
        top[1].data[...] = minibatch_target_sent.astype(np.int32, copy=False)
        top[2].reshape(*(minibatch_input_sent.shape))
        top[2].data[...] = minibatch_input_sent.astype(np.int32, copy=False)
        top[3].reshape(*(minibatch_cont_sent.shape))
        top[3].data[...] = minibatch_cont_sent.astype(np.int32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

