# Multilevel Language and Vision Integration for Text-to-Clip Retrieval

Code released by Huijuan Xu (Boston University).

### Introduction

We address the problem of text-based activity retrieval in video. Given a
sentence describing an activity, our task is to retrieve matching clips
from an untrimmed video. Our model learns a fine-grained similarity metric
for retrieval and uses visual features to modulate the processing of query
sentences at the word level in a recurrent neural network. A multi-task
loss is also employed by adding query re-generation as an auxiliary task.


### License

Our code is released under the MIT License (refer to the LICENSE file for
details).

### Citing

If you find our paper useful in your research, please consider citing:


    @inproceedings{xu2019multilevel,
    title={Multilevel Language and Vision Integration for Text-to-Clip Retrieval.},
    author={Xu, Huijuan and He, Kun and Plummer, Bryan A. and Sigal, Leonid and Sclaroff,
    Stan and Saenko, Kate},
    booktitle={AAAI},
    year={2019}
    }


### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train Proposal Network](#train_proposal_network)
4. [Extract Proposal Features](#extract_proposal_features)
5. [Training](#training)
6. [Testing](#testing)

### Installation:

1. Clone the Text-to-Clip_Retrieval repository.
   ```Shell
   git clone --recursive git@github.com:VisionLearningGroup/Text-to-Clip_Retrieval.git
   ```

2. Build `Caffe3d` with `pycaffe` (see: [Caffe installation
instructions](http://caffe.berkeleyvision.org/installation.html)).

   **Note:** Caffe must be built with Python support!
  
  ```Shell
  cd ./caffe3d
  
  # If have all of the requirements installed and your Makefile.config in
    place, then simply do:
  make -j8 && make pycaffe
  ```

3. Build lib folder.

   ```Shell
   cd ./lib    
   make
   ```

### Preparation:

1. We convert the [orginal data annotation files](https://github.com/jiyanggao/TALL) into json format.

   ```Shell
   # train data json file
   caption_gt_train.json 
   # test data json file
   caption_gt_test.json
   ```

2. Download the videos in [Charades
dataset](https://allenai.org/plato/charades/) and extract frames at 25fps.



### Train Proposal Network:

1. Generate the pickle data for training proposal network model.

   ```Shell
   cd ./preprocess
   # generate training data
   python generate_roidb_modified_freq1.py
   ```

2. Download C3D classification [pretrain model](https://drive.google.com/file/d/1os4a1K4pgjhRh8oiL7gO_DhM0NnCURHN/view) to ./pretrain/ .

3. In root folder, run proposal network training:
   ```Shell
   bash ./experiments/train_rpn/script_train.sh
   ```

4. We provide one set of trained proposal network [model weights](https://drive.google.com/file/d/1w8TL-lm7wjOVTYgzBdHGvXbJc5AHZ16g/view).


### Extract Proposal Features:

1. In root folder, extract proposal features for training data and save as
   hdf5 data.
   ```Shell
   bash ./experiments/extract_HDF_for_LSTM/script_test.sh
   ```


### Training:

1. In root folder, run:
   ```Shell
   bash ./experiments/Text_to_Clip/script_train.sh
   ```

### Testing:

1. Generate the pickle data for testing the Text_to_Clip model.

   ```Shell
   cd ./preprocess
   # generate test data
   python generate_roidb_modified_freq1_full_retrieval_test.py
   ```

2. Download one sample model to ./experiments/Text_to_Clip/snapshot/ .

   One Text_to_Clip model on Charades-STA dataset is provided in:
   [caffemodel
   .](https://drive.google.com/file/d/10C2gPLQXyNZ39CVLVKiWpYy5qu1xGvtX/view)

   The provided model has Recall@1 (tIoU=0.7) score ~15.6% on the
   test set.
   
3. In root folder, generate the similarity scores on the test set and save
   as pickle file.
   ```Shell
   bash ./experiments/Text_to_Clip/test_fast/script_test.sh 
   ```
   
4. Get the evaluation results.
   ```Shell
   cd ./experiments/Text_to_Clip/test_fast/evaluation/
   bash bash.sh
   ```

