#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/video_segment_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
VideoSegmentDataLayer<Dtype>::~VideoSegmentDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void VideoSegmentDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>&
      bottom, const vector<Blob<Dtype>*>& top) {
  const int new_length = this->layer_param_.video_data_param().new_length();
  const int new_height = this->layer_param_.video_data_param().new_height();
  const int new_width  = this->layer_param_.video_data_param().new_width();
  const bool is_color  = this->layer_param_.video_data_param().is_color();
  string root_folder = this->layer_param_.video_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.video_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t posa, posb, posc, posd, pose;
  while (std::getline(infile, line)) {
    multilet video_segments;
    posa = line.find(' ');
    video_segments.name = line.substr(0, posa);
    posb = line.find(' ', posa + 1);
    video_segments.start = atoi(line.substr(posa+1, posb).c_str());
    posc = line.find(' ', posb + 1);
    video_segments.end = atoi(line.substr(posb+1, posc).c_str());
    posd = line.find(' ', posc + 1);
    video_segments.stride = atoi(line.substr(posc+1, posd).c_str());
    pose = line.find(' ', posd + 1);
    video_segments.num_segs = atoi(line.substr(posd+1, pose).c_str());
    
    line = line.erase(0, pose+1);
    for ( size_t i = 0; i < video_segments.num_segs; ++i ) {
      posa = line.find(' ');
      posb = line.find(' ', posa + 1);
      posc = line.find(' ', posb + 1);
      float left  = atof(line.substr(0, posa).c_str());
      float right = atof(line.substr(posa+1, posb).c_str());
      int score = atoi(line.substr(posb+1, posc).c_str());
      video_segments.segs.push_back(left);
      video_segments.segs.push_back(right);
      video_segments.labels.push_back(score);
      line = line.erase(0, posc+1);
    } 
    lines_.push_back(video_segments);
  }

  if (this->layer_param_.video_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleVideos();
  }
  LOG(INFO) << "A total of " << lines_.size() << " video chunks.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.video_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.video_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read a video clip, and use it to initialize the top blob.
  std::vector<cv::Mat> cv_imgs;
  bool read_video_result = ReadVideoToCVMat(root_folder +
                                            lines_[lines_id_].name,
                                            lines_[lines_id_].start,
                                            lines_[lines_id_].end,
                                            lines_[lines_id_].stride,
                                            new_length, new_height, new_width,
                                            is_color,
                                            &cv_imgs);
  CHECK(read_video_result) << "Could not load " << lines_[lines_id_].name <<
                              " at frame " << lines_[lines_id_].start << ".";
  CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
                                          lines_[lines_id_].name <<
                                          " at frame " <<
                                          lines_[lines_id_].start <<
                                          " correctly.";
  // Use data_transformer to infer the expected blob shape from a cv_image.
  const bool is_video = true;
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_imgs,
                                                                  is_video);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.video_data_param().batch_size();
  CHECK_EQ(batch_size, 1) << "Batch size is required to 1";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->shape(0) << ","
      << top[0]->shape(1) << "," << top[0]->shape(2) << ","
      << top[0]->shape(3) << "," << top[0]->shape(4);
  // label
  vector<int> label_shape(2, 10);
  label_shape[1] = 3;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void VideoSegmentDataLayer<Dtype>::ShuffleVideos() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void VideoSegmentDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  VideoDataParameter video_data_param = this->layer_param_.video_data_param();
  const int batch_size = video_data_param.batch_size();
  const int new_length = video_data_param.new_length();
  const int new_height = video_data_param.new_height();
  const int new_width = video_data_param.new_width();
  const bool is_color = video_data_param.is_color();
  string root_folder = video_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  std::vector<cv::Mat> cv_imgs;
  bool read_video_result = ReadVideoToCVMat(root_folder +
                                             lines_[lines_id_].name,
                                             lines_[lines_id_].start,
                                             lines_[lines_id_].end,
                                             lines_[lines_id_].stride,
                                             new_length, new_height, new_width,
                                             is_color,
                                             &cv_imgs);
  CHECK(read_video_result) << "Could not load " << lines_[lines_id_].name <<
                              " at frame " << lines_[lines_id_].start << ".";
  CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
                                          lines_[lines_id_].name <<
                                          " at frame " <<
                                          lines_[lines_id_].start <<
                                          " correctly.";
  // Use data_transformer to infer the expected blob shape from a cv_imgs.
  bool is_video = true;
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_imgs,
                                                                  is_video);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  vector<int> blob_offset(5);
  blob_offset[1] = 0;
  blob_offset[2] = 0;
  blob_offset[3] = 0;
  blob_offset[4] = 0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    std::vector<cv::Mat> cv_imgs;
    bool read_video_result = ReadVideoToCVMat(root_folder +
                                               lines_[lines_id_].name,
                                               lines_[lines_id_].start,
                                               lines_[lines_id_].end,
                                               lines_[lines_id_].stride,
                                               new_length, new_height,
                                               new_width, is_color, &cv_imgs);
    CHECK(read_video_result) << "Could not load " << lines_[lines_id_].name <<
                                " at frame " << lines_[lines_id_].start << ".";
    CHECK_EQ(cv_imgs.size(), new_length) << "Could not load " <<
                                             lines_[lines_id_].name <<
                                            " at frame " <<
                                            lines_[lines_id_].start <<
                                            " correctly.";
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    blob_offset[0] = item_id;
    int offset = batch->data_.offset(blob_offset);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    const bool is_video = true;
    this->data_transformer_->Transform(cv_imgs, &(this->transformed_data_),
                                       is_video);
    trans_time += timer.MicroSeconds();

    num_seg_ = lines_[lines_id_].labels.size();
    vector<int> label_shape(2, num_seg_); 
    label_shape[1] = 3;
    
    batch->label_.Reshape(label_shape);
    for (int seg_id = 0; seg_id < num_seg_; ++seg_id){
      prefetch_label[0] = lines_[lines_id_].segs[2 * seg_id];
      prefetch_label[1] = lines_[lines_id_].segs[2 * seg_id + 1];
      prefetch_label[2] = lines_[lines_id_].labels[seg_id];
      prefetch_label += 3;
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.video_data_param().shuffle()) {
        ShuffleVideos();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(VideoSegmentDataLayer);
REGISTER_LAYER_CLASS(VideoSegmentData);

}  // namespace caffe
#endif  // USE_OPENCV
