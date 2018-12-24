# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# By Huijuan Xu
# --------------------------------------------------------


#export CUDA_HOME=/usr/local/cuda-7.5
#export LD_LIBRARY_PATH=${CUDA_HOME}/lib64

GPU_ID=0
EX_DIR=extract_HDF_for_LSTM

export PYTHONUNBUFFERED=true


for (( i=30; i<=30; i+=10 )); do

LOG="experiments/${EX_DIR}/test_log_${i}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

time python ./experiments/${EX_DIR}/test_net.py --gpu ${GPU_ID} \
  --def ./experiments/${EX_DIR}/test_rpn.prototxt \
  --net ../train_rpn/snapshot/activitynet_iter_${i}000.caffemodel \
  --cfg ./experiments/${EX_DIR}/td_cnn_end2end.yml \
  ${EXTRA_ARGS} \
  2>&1 | tee $LOG

done


