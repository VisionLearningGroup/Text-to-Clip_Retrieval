# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# By Huijuan Xu
# --------------------------------------------------------


#export CUDA_HOME=/usr/local/cuda-7.5
#export LD_LIBRARY_PATH=${CUDA_HOME}/lib64

GPU_ID=0

EX_PATH=experiments
EX_DIR=Text_to_Clip


export PYTHONUNBUFFERED=true


for (( i=5; i<=5; i+=1 )); do



LOG="${EX_PATH}/${EX_DIR}/test_fast/test_log_${i}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"


time python ./${EX_PATH}/${EX_DIR}/test_fast/test_net.py --gpu ${GPU_ID} \
  --def ./${EX_PATH}/${EX_DIR}/test_fast/test_rpn.prototxt \
  --def-lstm ./${EX_PATH}/${EX_DIR}/test_fast/test_lstm.prototxt \
  --def-retrieval ./${EX_PATH}/${EX_DIR}/test_fast/test_retrieval.prototxt \
  --net ...../train_rpn/snapshot/activitynet_iter_30000.caffemodel \
  --netRetrieval ./${EX_PATH}/${EX_DIR}/snapshot/lstm_lm_iter_${i}000.caffemodel \
  --cfg ./${EX_PATH}/${EX_DIR}/test_fast/td_cnn_end2end.yml \
  ${EXTRA_ARGS} \
  2>&1 | tee $LOG

done




