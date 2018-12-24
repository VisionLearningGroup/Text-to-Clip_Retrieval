# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# By Huijuan Xu
# --------------------------------------------------------


import os
import copy
import json
import cPickle
import subprocess
import numpy as np
from util_2 import *


FPS = 25.0
LENGTH = 768
min_length = 0 # frame num (filter out)
overlap_thresh = 0.7 
STEP = LENGTH / 4
WINS = [LENGTH * 5] 
max_words = 10

vocab_out_path = './vocabulary.txt'
TRAIN_META_FILE = 'caption_gt_train.json'
train_data = json.load(open(TRAIN_META_FILE))

#pre-processing

train_caption =[]
for vid in train_data.keys():
  vinfo = train_data[vid]
  for k in vinfo['sentences']:
    train_caption.append(split_sentence(k))

print '\nthe total number of captions are:',len(train_caption)

#init-vocabulary
vocab,vocab_inverted = init_vocabulary(train_caption, min_count=1)  
dump_vocabulary(vocab_inverted, vocab_out_path)


path = '........./preprocess'
print('Generate Training Segments')
train_segment = generate_segment(path, 'frames',train_data,vocab,max_words)


TEST_META_FILE = 'caption_gt_test.json'
test_data = json.load(open(TEST_META_FILE))
print('Generate Testing Segments')
test_segment = generate_segment(path,'frames',test_data,vocab,max_words)




def generate_roi(rois, rois_lstm, video, start, end, stride, split):
  tmp = {}
  tmp['wins'] = ( rois[:,:2] - start ) / stride
  tmp['durations'] = tmp['wins'][:,1] - tmp['wins'][:,0]+1
  tmp['gt_classes'] = np.ones(rois.shape[0])
  tmp['input_sentence'] = rois_lstm[:,0]
  tmp['cont_sentence'] = rois_lstm[:,1]
  tmp['target_sentence'] = rois_lstm[:,2]
  tmp['max_classes'] = np.ones(rois.shape[0])
  tmp['max_overlaps'] = np.ones(len(rois))
  tmp['flipped'] = False
  tmp['frames'] = np.array([[0, start, end, stride]])
  tmp['vid'] = video
  tmp['stride'] = stride
  tmp['start_frame'] = start
  tmp['end_frame'] = end
  tmp['FPS'] = FPS
  tmp['bg_name'] = path + '/'+split+'/' + video
  tmp['fg_name'] = path + '/'+split+'/' + video
  if not os.path.isfile(tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg'):
    print  tmp['bg_name'] + '/image_' + str(end-1).zfill(5) + '.jpg'
  return tmp




def generate_roidb(split, segment):
  max_num_seg = 0
  VIDEO_PATH = '%s/%s/' % (path,split)
  video_list = set(os.listdir(VIDEO_PATH))
  remove = 0
  overall = 0
  duration = []
  roidb = []
  for vid in segment.keys():
    if vid in video_list:
      length = len(os.listdir(VIDEO_PATH + vid))
      seg_tmp = segment[vid]
      db=[]
      db_lstm = []
      for s in seg_tmp:
        db.append([s[0],s[1]])
        db_lstm.append([s[2],s[3],s[4]])
      db = np.array(db)
      db_lstm = np.array(db_lstm)
      db = db * FPS
      rois = db
      rois_lstm = db_lstm
      overall += len(db)
      if len(db) == 0:
        continue
      debug = []

      for win in WINS:
        stride = win / LENGTH
        step = stride * STEP
        # Forward Direction
        for start in xrange(0, max(1, length - win + 1), step):
          end = min(start + win, length)
          assert end <= length
          # Add data
          tmp = generate_roi(rois, rois_lstm, vid, start, end, stride, split)
          if tmp['end_frame']==0:
            print tmp['vid']
          else:
            roidb.append(tmp)
          if USE_FLIPPED:
            flipped_tmp = copy.deepcopy(tmp)
            flipped_tmp['flipped'] = True
            if flipped_tmp['end_frame']==0:
              print flipped_tmp['vid']
            else:
              roidb.append(tmp)
          for d in rois:
            debug.append(d)
              
        # Backward Direction
        for end in xrange(length, win-1, - step):
          start = end - win
          assert start >= 0
          # Add data
          tmp = generate_roi(rois, rois_lstm, vid, start, end, stride, split)
          if tmp['end_frame']==0:
            print tmp['vid']
          else:
            roidb.append(tmp)
          #roidb.append(tmp)
          if USE_FLIPPED:
            flipped_tmp = copy.deepcopy(tmp)
            flipped_tmp['flipped'] = True
            if flipped_tmp['end_frame']==0:
              print flipped_tmp['vid']
            else:
              roidb.append(tmp)
            #roidb.append(flipped_tmp)
          for d in rois:
            debug.append(d)

      debug_res=[list(x) for x in set(tuple(x) for x in debug)]
      if len(debug_res) < len(db):
        remove += len(db) - len(debug_res)

  print '\nthe maximum number of segments in each window is:', max_num_seg
  print remove, ' / ', overall
  return roidb


USE_FLIPPED = False
test_roidb = generate_roidb('frames', test_segment)
print len(test_roidb)

print "Save dictionary"



## for retrieval test
cPickle.dump(test_roidb, open('./test_data_modified_5fps_caption_768_1_full_retrieval.pkl','w'), cPickle.HIGHEST_PROTOCOL)

