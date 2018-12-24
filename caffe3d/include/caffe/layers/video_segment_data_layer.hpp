#ifndef CAFFE_VIDEO_SEGMENT_DATA_LAYER_HPP_
#define CAFFE_VIDEO_SEGMENT_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

struct multilet {
  std::string name;           // video filename
  int start, end, stride, num_segs;     // start frame, stride
  std::vector<float> segs;  // segments
  std::vector<int> labels;     // labels
};

namespace caffe {

/**
 * @brief Provides data to the Net from video files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class VideoSegmentDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VideoSegmentDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~VideoSegmentDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VideoSegmentData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleVideos();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<multilet> lines_;
  int lines_id_;
  int num_seg_;
};


}  // namespace caffe

#endif  // CAFFE_VIDEO_SEGMENT_DATA_LAYER_HPP_
