# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# By Huijuan Xu
# --------------------------------------------------------

"""Test a Text-to-Clip Retrieval network."""

from tdcnn.config import cfg
from tdcnn.twin_transform import clip_wins, twin_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from tdcnn.nms_wrapper import nms
import cPickle
from utils.blob import video_list_to_blob, prep_im_for_blob
import os
import random
#from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist
import pickle


### cosine similarity option 1
def cos(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))


### euclidean distance option 1
def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))



def softmax(softmax_inputs, temp):
    shifted_inputs = softmax_inputs - softmax_inputs.max()
    exp_outputs = np.exp(temp * shifted_inputs)
    exp_outputs_sum = exp_outputs.sum()
    if np.isnan(exp_outputs_sum):
        return exp_outputs * float('nan')
    assert exp_outputs_sum > 0
    if np.isinf(exp_outputs_sum):
        return np.zeros_like(exp_outputs)
    eps_sum = 1e-20
    return exp_outputs / max(exp_outputs_sum, eps_sum)


def random_choice_from_probs(softmax_inputs, temp=1):
    # temperature of infinity == take the max
    if temp == float('inf'):
        return np.argmax(softmax_inputs)
    probs = softmax(softmax_inputs, temp)
    r = random.random()
    cum_sum = 0.
    for i, p in enumerate(probs):
        cum_sum += p
        if cum_sum >= r: return i
    return 1  # return UNK?


def generate_sentence(net, fc6, temp=float('inf'), output='predict', max_words=50):
    cont_input = np.array([0])
    word_input = np.array([0])
    sentence = []
    while len(sentence) < max_words and (not sentence or sentence[-1] != 0):
        net.forward(cont_sentence=cont_input, input_sentence=word_input, caption_fc6=fc6.reshape(1,fc6.shape[0])) 
        output_preds = net.blobs[output].data[0, 0, :]
        sentence.append(random_choice_from_probs(output_preds, temp=temp))
        cont_input[0] = 1
        word_input[0] = sentence[-1]
    return sentence


def lstm_last_hidden_state(net, fc6, query, temp=float('inf'), output='predict'):
    q = query.T
    cont_input = np.ones(q.shape)
    caption_fc6 = np.tile(fc6[np.newaxis, :], (query.shape[0], 1))
    cont_input[0,:] = 0
    net.blobs['cont_sentence'].reshape(*(cont_input.shape))
    net.blobs['input_sentence'].reshape(*(q.shape))
    net.blobs['caption_fc6'].reshape(*(caption_fc6.shape))
    net.forward(cont_sentence=cont_input, input_sentence=q, caption_fc6=caption_fc6)
    indeces = (query!=-1).sum(axis=1) - 1
    output_preds = net.blobs[output].data
    selected = np.empty(output_preds.shape[1:])
    for i,k in enumerate(indeces):
      selected[i] = output_preds[k,i,:]
    return selected


def _get_video_blob(roidb,vocab):
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """

    processed_videos = []

    item = roidb

    video_length = cfg.TEST.LENGTH[0]
    video = np.zeros((video_length, cfg.TEST.CROP_SIZE,
                      cfg.TEST.CROP_SIZE, 3))

    j = 0
    random_idx = [int(cfg.TEST.FRAME_SIZE[1]-cfg.TEST.CROP_SIZE) / 2,
                  int(cfg.TEST.FRAME_SIZE[0]-cfg.TEST.CROP_SIZE) / 2]

    if cfg.INPUT == 'video':
      for video_info in item['frames']:
        prefix = item['fg_name'] if video_info[0] else item['bg_name']
        for idx in xrange(video_info[1], video_info[2], video_info[3]):
          frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))  
          frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]),
                                   cfg.TRAIN.CROP_SIZE, random_idx)   

          if item['flipped']:
              frame = frame[:, ::-1, :]  

          video[j] = frame
          j = j + 1

    else:
      for video_info in item['frames']:
        prefix = item['fg_name'] if video_info[0] else item['bg_name']
        for idx in xrange(video_info[1], video_info[2]):
          frame = cv2.imread('%s/image_%s.jpg'%(prefix,str(idx+1).zfill(5)))  
          frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TEST.FRAME_SIZE[::-1]),
                                   cfg.TEST.CROP_SIZE, random_idx)

          if item['flipped']:
              frame = frame[:, ::-1, :]

          video[j] = frame
          j = j + 1

    while ( j < video_length):
      video[j] = frame
      j = j + 1
    processed_videos.append(video)

    # Create a blob to hold the input images
    blob = video_list_to_blob(processed_videos)

    return blob

def _get_blobs(video, rois = None):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'] = video
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs


def video_detect(net, video, wins=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        wins (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        wins (ndarray): R x (4*K) array of predicted bounding wins
    """
    blobs = _get_blobs(video)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:   #no use
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        wins = wins[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if not cfg.TEST.HAS_RPN:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if not cfg.TEST.HAS_RPN:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    pred_wins = blobs_out['caption_rois']
    fc6 = blobs_out['caption_fc6']
    proposal_scores = blobs_out['proposal_scores']
    return pred_wins, fc6, proposal_scores



# function to extract the Sentence embedding from the retrieval model
def extract_retrieval_score(net, lstm_hidden):
    blobs = {'last_hidden_state': lstm_hidden}

    # reshape network inputs
    net.blobs['last_hidden_state'].reshape(*(blobs['last_hidden_state'].shape))

    # do forward
    forward_kwargs = {'last_hidden_state': blobs['last_hidden_state'].astype(np.float32, copy=False)}
    blobs_out = net.forward(**forward_kwargs)

    # fc6_embed = blobs_out['feat_i_norm']
    score = blobs_out['score']
    return score



def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        twin = dets[i, :2]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((twin[0], twin[1]),
                              twin[2] - twin[0],
                              twin[3] - twin[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_wins, thresh):
    """Apply non-maximum suppression to all predicted wins output by the
    test_net method.
    """
    num_classes = len(all_wins)
    num_images = len(all_wins[0])
    nms_wins = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_wins[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of wins
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_wins[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_wins

def test_net(net, roidb, lstm_net, retrieval_net, sim_pickle_path, vocab, max_per_image=100, thresh=0.05, vis=False):  
    num_videos = len(roidb)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    
    res = {}

    for i in xrange(num_videos):
        # filter out any ground truth wins
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['wins'][roidb[i]['gt_classes'] == 0]

        video = _get_video_blob(roidb[i], vocab)

        _t['im_detect'].tic()
        wins, fc6, proposal_scores= video_detect(net, video, box_proposals)  
        _t['im_detect'].toc()

        _t['misc'].tic()

        query = roidb[i]['input_sentence']  
        batch = 32
        num_batch = (query.shape[0] - 1) / batch + 1

        sim = np.zeros((query.shape[0], wins.shape[0]))
        for d in xrange(wins.shape[0]):
          for b in xrange(num_batch):
            fc6_temp = fc6[d,:]
            q = query[b*batch:(b+1)*batch]
            lstm_last_state = lstm_last_hidden_state(lstm_net, fc6_temp, q, output='lstm2')
            score = extract_retrieval_score(retrieval_net, lstm_last_state)
            sim[b*batch:(b+1)*batch,d] = score.squeeze()

        print sim.max()    


        vid = roidb[i]['vid']
        stride = roidb[i]['stride']
        start_frame = roidb[i]['start_frame']
        end_frame = roidb[i]['end_frame']
        FPS = roidb[i]['FPS']

        #tmp = {'query' : None, 'timestamp': None, 'sim_scores': None}
        tmp={}
        tmp['query'] = roidb[i]['target_sentence']
        left_frame = np.maximum(wins[:,1]*stride + start_frame, start_frame)/FPS
        right_frame = np.minimum(wins[:,2]*stride + start_frame, end_frame)/FPS
        tmp['timestamp'] = np.vstack((left_frame,right_frame,proposal_scores)).transpose((1,0))
        tmp['sim_scores'] = sim

        if vid not in res:
            res[vid] = []
        res[vid].append(tmp)

        
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_videos, _t['im_detect'].average_time,
                      _t['misc'].average_time)


    pickle.dump( res, open( sim_pickle_path, "wb" ) )
