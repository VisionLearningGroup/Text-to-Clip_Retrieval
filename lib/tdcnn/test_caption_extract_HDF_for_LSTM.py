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
import h5py


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



def _get_video_blob(roidb):
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """

    processed_videos = []

    item = roidb

    for key in item:
      print key, ": ", item[key]
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


def _get_blobs(video, gt_windows, gt_captions, rois = None):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'gt_boxes' : None, 'gt_captions' : None, 'rois' : None}
    blobs['data'] = video
    blobs['gt_boxes'] = gt_windows
    blobs['gt_captions'] = gt_captions
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs


def video_detect(net, video, gt_windows, gt_captions, wins=None):
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
    blobs = _get_blobs(video, gt_windows, gt_captions)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['gt_boxes'].reshape(*(blobs['gt_boxes'].shape))
    net.blobs['gt_captions'].reshape(*(blobs['gt_captions'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False), 'gt_boxes': blobs['gt_boxes'].astype(np.float32, copy=False), 'gt_captions': blobs['gt_captions'].astype(np.float32, copy=False)}

    blobs_out = net.forward(**forward_kwargs)

    input_sent = blobs_out['input_sentence']
    cont_sent = blobs_out['cont_sentence']
    target_sent = blobs_out['target_sentence']
    fc6 = blobs_out['caption_fc6_target']

    return input_sent, cont_sent, target_sent, fc6

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

def test_net(net, roidb, hdf5_path, max_per_image=100, thresh=0.05, vis=False):  
    num_videos = len(roidb)
    # all detections are collected into:
    #    all_wins[cls][image] = N x 2 array of detections in
    #    (x1, x2, score)
    # all_wins = [[[] for _ in xrange(num_videos)]
    #              for _ in xrange(cfg.NUM_CLASSES)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    all_input_sent = np.empty((cfg.MAX_WORDS, 0), dtype=np.float32)
    all_cont_sent = np.empty((cfg.MAX_WORDS, 0), dtype=np.float32)
    all_target_sent = np.empty((cfg.MAX_WORDS, 0), dtype=np.float32)
    all_fc6 = np.empty((0, 4096), dtype=np.float32)

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


        video = _get_video_blob(roidb[i])
        _t['im_detect'].tic()

        # gt windows: (x1, x2, cls)
        gt_inds = np.where(roidb[i]['gt_classes'] != 0)[0]
        gt_windows = np.empty((len(gt_inds), 3), dtype=np.float32)
        gt_windows[:, 0:2] = roidb[i]['wins'][gt_inds, :]
        gt_windows[:, -1] = roidb[i]['gt_classes'][gt_inds]
        gt_captions = np.empty((len(gt_inds), 3, cfg.MAX_WORDS), dtype=np.float32)        
        gt_captions[:, 0] = roidb[i]['input_sentence'][gt_inds,:]
        gt_captions[:, 1] = roidb[i]['cont_sentence'][gt_inds,:]
        gt_captions[:, 2] = roidb[i]['target_sentence'][gt_inds,:]


        input_sent, cont_sent, target_sent, fc6 = video_detect(net, video, gt_windows, gt_captions, box_proposals)  

        print '\n'
        print '\nnumber of GT captions in this windows: ', len(gt_inds)
        print '\nnumber of captions from the top32:',fc6.shape[0]-len(gt_inds)
        print '\n'
        
        all_input_sent = np.hstack((all_input_sent, input_sent))
        all_cont_sent = np.hstack((all_cont_sent, cont_sent))
        all_target_sent = np.hstack((all_target_sent, target_sent))
        all_fc6 = np.vstack((all_fc6, fc6))  


        
        _t['im_detect'].toc()
        _t['misc'].tic()
        _t['misc'].toc()


        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_videos, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    print "pad the data to full batch:\n"
    if all_fc6.shape[0]%cfg.LSTM_BATCH_SIZE!=0:
        num_pad = cfg.LSTM_BATCH_SIZE - all_fc6.shape[0]%cfg.LSTM_BATCH_SIZE
        ind_pad = np.random.randint(all_fc6.shape[0], size=num_pad)
        all_input_sent = np.hstack((all_input_sent, all_input_sent[:,ind_pad]))
        all_cont_sent = np.hstack((all_cont_sent, all_cont_sent[:,ind_pad]))
        all_target_sent = np.hstack((all_target_sent, all_target_sent[:,ind_pad]))
        all_fc6 = np.vstack((all_fc6, all_fc6[ind_pad,:]))  


    print "shuffle all the data:\n"
    perm = np.random.permutation(np.arange(all_fc6.shape[0]))
    all_fc6 = all_fc6[perm,:]
    all_input_sent = all_input_sent[:, perm]
    all_cont_sent = all_cont_sent[:, perm]
    all_target_sent = all_target_sent[:, perm]

    print "save data into hdf5 file:\n"
    batch_stream_length = 40000
    num_batches = (all_fc6.shape[0]-1) / batch_stream_length + 1
    files = []
    output_path = hdf5_path+'hdf5data/'
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    for k in xrange(num_batches):
        print k
        
        all_fc6_batched = all_fc6[k*batch_stream_length:(k+1)*batch_stream_length, :]
        all_input_sent_tmp = all_input_sent[:, k*batch_stream_length:(k+1)*batch_stream_length]
        all_cont_sent_tmp = all_cont_sent[:, k*batch_stream_length:(k+1)*batch_stream_length] 
        all_target_sent_tmp = all_target_sent[:, k*batch_stream_length:(k+1)*batch_stream_length]
        
        all_input_sent_batched = np.empty((0, cfg.LSTM_BATCH_SIZE), dtype=np.float32)
        all_cont_sent_batched = np.empty((0, cfg.LSTM_BATCH_SIZE), dtype=np.float32)
        all_target_sent_batched = np.empty((0, cfg.LSTM_BATCH_SIZE), dtype=np.float32)
        for m in xrange( all_fc6_batched.shape[0] /cfg.LSTM_BATCH_SIZE):
            all_input_sent_batched = np.vstack((all_input_sent_batched, all_input_sent_tmp[:, m*cfg.LSTM_BATCH_SIZE:(m+1)*cfg.LSTM_BATCH_SIZE]))
            all_cont_sent_batched = np.vstack((all_cont_sent_batched, all_cont_sent_tmp[:, m*cfg.LSTM_BATCH_SIZE:(m+1)*cfg.LSTM_BATCH_SIZE]))
            all_target_sent_batched = np.vstack((all_target_sent_batched, all_target_sent_tmp[:, m*cfg.LSTM_BATCH_SIZE:(m+1)*cfg.LSTM_BATCH_SIZE]))

        
        print all_fc6_batched.shape
        print all_input_sent_batched.shape
        print all_cont_sent_batched.shape
        print all_target_sent_batched.shape
        
        filename = '%s/batch_%d.h5' % (output_path, k)
        files.append(filename)
        h5file = h5py.File(filename, 'w')
        dataset = h5file.create_dataset('all_fc6_batched', shape=all_fc6_batched.shape, dtype=all_fc6_batched.dtype)
        dataset[:] = all_fc6_batched
        dataset = h5file.create_dataset('all_input_sent_batched', shape=all_input_sent_batched.shape, dtype=all_input_sent_batched.dtype)
        dataset[:] = all_input_sent_batched
        dataset = h5file.create_dataset('all_cont_sent_batched', shape=all_cont_sent_batched.shape, dtype=all_cont_sent_batched.dtype)
        dataset[:] = all_cont_sent_batched
        dataset = h5file.create_dataset('all_target_sent_batched', shape=all_target_sent_batched.shape, dtype=all_target_sent_batched.dtype)
        dataset[:] = all_target_sent_batched
        h5file.close()

    filelist = '%shdf5_chunk_list.txt' % output_path
    print 'dumping hdf data: ',filelist
    print '\n'
              
    with open(filelist, 'wb') as listfile:
        for f in files:
            listfile.write('%s\n' % f)
                        
    


