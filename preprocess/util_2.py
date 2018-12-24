# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------


import subprocess
import json
import shutil
import os, errno
#import cv2
import re




SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def split_sentence(sentence):
  # break sentence into a list of words and punctuation
  sentence = [s.lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]
  # remove the '.' from the end of the sentence
  if sentence[-1] != '.':
    # print "Warning: sentence doesn't end with '.'; ends with: %s" % sentence[-1]                                                  
    return sentence
  return sentence[:-1]

UNK_IDENTIFIER = '<unk>'
def init_vocabulary(image_annotations, min_count=1):
  words_to_count = {}
  for annotation in image_annotations:
    for word in annotation:
      word = word.strip()
      if word not in words_to_count:
        words_to_count[word] = 0
      words_to_count[word] += 1
  # Sort words by count, then alphabetically                                                                                                                                                        
  words_by_count = sorted(words_to_count.keys(), key=lambda w: (-words_to_count[w], w))
  print 'Initialized vocabulary with %d words; top 10 words:' % len(words_by_count)
  for word in words_by_count[:10]:
    print '\t%s (%d)' % (word, words_to_count[word])
  # Add words to vocabulary                                                                                                                                                                         
  vocabulary = {UNK_IDENTIFIER: 0}
  vocabulary_inverted = [UNK_IDENTIFIER]
  for index, word in enumerate(words_by_count):
    word = word.strip()
    if words_to_count[word] < min_count:
      break
    vocabulary_inverted.append(word)
    vocabulary[word] = index + 1
  print 'Final vocabulary (restricted to words with counts of %d+) has %d words' % \
      (min_count, len(vocabulary))
  return vocabulary, vocabulary_inverted

def dump_vocabulary(vocabulary_inverted, vocab_filename):
  print 'Dumping vocabulary to file: %s' % vocab_filename
  with open(vocab_filename, 'w') as vocab_file:
    for word in vocabulary_inverted:
      vocab_file.write('%s\n' % word.encode('utf-8'))
  print 'Done.'


def line_to_stream(sentence, vocabulary):
  stream = []
  for word in sentence:
    word = word.strip()
    if word in vocabulary:
      stream.append(vocabulary[word])
    else:  # unknown word; append UNK                                                                                                   
      stream.append(vocabulary[UNK_IDENTIFIER])
  # increment the stream -- 0 will be the EOS character                                                                                 
  stream = [s + 1 for s in stream]                                                                                                    
  #stream = [s for s in stream]   # no need for EOS token in memory network                                                              
  return stream




def generate_segment(path, split, data, vocab, max_words):
  miss_cnt=0
  segment = {}
  VIDEO_PATH = '%s/%s/' % (path, split)
  video_list = os.listdir(VIDEO_PATH)
  for vid in data.keys():
    v_folder = [v for v in video_list if vid in v]
    if v_folder==[]:
      miss_cnt = miss_cnt+1
      print miss_cnt
    vinfo = data[vid]
    if (v_folder!=[]) and (v_folder[0] in video_list):    #len(vid_name) == 1:
      segment[vid] = []
      for k in xrange(len(vinfo['timestamps'])):
        start_time = vinfo['timestamps'][k][0]
        end_time = vinfo['timestamps'][k][1]
        caption = split_sentence(vinfo['sentences'][k])
        stream = line_to_stream(caption, vocab)
        pad = max_words - (len(stream) + 1)
        out = {}
        if pad <= 0:
          out['input_sentence'] = [0] + stream[:max_words-1]
          out['cont_sentence'] = [0] + [1] * (max_words - 1)
          out['target_sentence'] = stream[:max_words-1] + [0]
        else:
          out['input_sentence'] = [0] + stream + [-1] * pad
          out['cont_sentence'] = [0] + [1] * len(stream) + [0] * pad
          out['target_sentence'] = stream + [0] + [-1] * pad
        segment[vid].append([start_time, end_time, out['input_sentence'], out['cont_sentence'],out['target_sentence']])
  return segment





def generate_segment_withRandomQueries(path, split, data, vocab, max_words):
  miss_cnt=0
  segment = {}
  VIDEO_PATH = '%s/%s/' % (path, split)
  video_list = os.listdir(VIDEO_PATH)
  for vid in data.keys():
    v_folder = [v for v in video_list if vid in v]
    if v_folder==[]:
      miss_cnt = miss_cnt+1
      print miss_cnt
    vinfo = data[vid]
    if (v_folder!=[]) and (v_folder[0] in video_list):    #len(vid_name) == 1:
      segment[vid] = []
      for k in xrange(len(vinfo['sentences_random_1'])):
        caption = split_sentence(vinfo['sentences_random_1'][k])
        stream = line_to_stream(caption, vocab)
        pad = max_words - (len(stream) + 1)
        out = {}
        if pad <= 0:
          out['input_sentence'] = [0] + stream[:max_words-1]
          out['cont_sentence'] = [0] + [1] * (max_words - 1)
          out['target_sentence'] = stream[:max_words-1] + [0]
        else:
          out['input_sentence'] = [0] + stream + [-1] * pad
          out['cont_sentence'] = [0] + [1] * len(stream) + [0] * pad
          out['target_sentence'] = stream + [0] + [-1] * pad

        start_time = 0
        end_time = 0
        segment[vid].append([start_time, end_time, out['input_sentence'], out['cont_sentence'],out['target_sentence']])
  return segment





def get_fps(filename):
  command = ["ffprobe", "-loglevel", "quiet", "-show_format", "-show_streams", filename]
  pipe = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  out, err = pipe.communicate()

  return [d for d in out.split() if 'nb_frames' in d][0].split('=')[1] # 0 is usually video

def get_length(filename):
  command = ["ffprobe", "-loglevel", "quiet", "-print_format", "json", "-show_format", "-show_streams", filename]
  pipe = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  out, err = pipe.communicate()
  res = json.loads(out)

  if 'format' in res:
    if 'duration' in res['format']:
      return float(res['format']['duration'])

  if 'streams' in res:
    # commonly stream 0 is the video
    for s in res['streams']:
      if 'duration' in s:
        return float(s['duration'])

  raise Exception('I found no duration')

def mkdir(path):
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise

def rm(path):
  try:
    shutil.rmtree(path)
  except OSError as e:
    if e.errno != errno.ENOENT:
      raise

def ffmpeg(filename, outfile, fps):
  command = ["ffmpeg", "-i", filename, "-q:v", "1", "-r", str(fps), outfile]
  pipe = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  pipe.communicate()

def ffmpeg_convert_fps(filename, outfile, fps, ext):
  outfile += '.mp4'
  if ext == "webm":
    command = ["ffmpeg", "-fflags", "+genpts", "-i", filename, "-q:v", "0", "-r", str(fps), "-y",outfile]
  elif ext == "mkv":
    command = ["ffmpeg", "-i", filename, "-q:v", "0", "-r", str(fps), "-y",outfile]
  else:
    command = ["ffmpeg", "-i", filename, "-q:v", "0", "-r", str(fps), "-y",outfile]
  pipe = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  pipe.communicate()

def resize(filename, size = (171, 128)):
  img = cv2.imread(filename, 100)
  img2 = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
  cv2.imwrite(filename, img2, [100])
