# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# By Huijuan Xu
# --------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Extract the paired fc6 feature and sentence feature, to calculate the similarity score 
# ----------------------------------------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from tdcnn.config import cfg
from tdcnn.twin_transform import twin_transform
from utils.cython_twin import twin_overlaps

DEBUG = False

class ExtractPairSimLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._batch_size = layer_params['batch_size']

        dim = self._batch_size
        # paired positive sim
        top[0].reshape(2*(dim-1)*dim)
        # paired negative sim
        top[1].reshape(2*(dim-1)*dim)

    def forward(self, bottom, top):

        fc6_sent_sim = bottom[0].data

        # reshape the similarity score back to matrix
        fc6_sent_sim = fc6_sent_sim.reshape(self._batch_size, self._batch_size) 

        # vstack/hstack
        positive_sim_ind = np.empty((0, 2), dtype=np.int32)
        negative_sim_ind = np.empty((0, 2), dtype=np.int32)
        for i in xrange(self._batch_size):
            curr = np.ones(self._batch_size-1)*i
            res = np.hstack((np.arange(i), np.arange(i+1, self._batch_size)))

            # positive_sim_index
            temp_pos = np.vstack((curr, curr)).transpose((1, 0))
            positive_ind =  np.vstack((temp_pos, temp_pos))
            positive_sim_ind = np.vstack((positive_sim_ind, positive_ind))
            # negative_sim_index
            temp_neg_1 = np.vstack((res, curr)).transpose((1, 0))
            temp_neg_2 = np.vstack((curr, res)).transpose((1, 0))
            negative_ind =  np.vstack((temp_neg_1, temp_neg_2))
            negative_sim_ind = np.vstack((negative_sim_ind, negative_ind))

        # get positive_sim and negative_sim using index
        self._positive_sim_ind = positive_sim_ind.astype(int)
        self._negative_sim_ind = negative_sim_ind.astype(int)
        positive_sim = fc6_sent_sim[positive_sim_ind[:,0].astype(int), positive_sim_ind[:,1].astype(int)]
        negative_sim = fc6_sent_sim[negative_sim_ind[:,0].astype(int), negative_sim_ind[:,1].astype(int)]

        top[0].reshape(*positive_sim.shape)
        top[0].data[...] = positive_sim 
        #print positive_sim.shape

        top[1].reshape(*negative_sim.shape)
        top[1].data[...] = negative_sim
        #print negative_sim.shape

    def backward(self, top, propagate_down, bottom):
        diff = np.zeros((self._batch_size, self._batch_size))
        for k, item in enumerate(self._positive_sim_ind):
          diff[item[0],item[1]] += top[0].diff[k] 
        for k, item in enumerate(self._negative_sim_ind):
          diff[item[0],item[1]] += top[1].diff[k] 
        bottom[0].diff[...] = diff.reshape(self._batch_size * self._batch_size)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


