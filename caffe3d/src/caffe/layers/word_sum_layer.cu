#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/word_sum_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WordSumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int j=0; j<bottom[0]->shape(0); ++j){
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
        weight, bottom_data + j*M_*N_, (Dtype)0., top_data + j*K_*N_);
  }
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, K_*N_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(),
        (Dtype)1., top_data);
  }
}

template <typename Dtype>
void WordSumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  if (this->param_propagate_down_[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    for (int j=0; j<bottom[0]->shape(0); ++j){
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, M_, N_, (Dtype)1.,
          top_diff + j*K_*N_, bottom_data + j*M_*N_, (Dtype)1.,
          this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, B_, K_*N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    for (int j=0; j<bottom[0]->shape(0); ++j){
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, K_, (Dtype)1.,
          this->blobs_[0]->gpu_data(), top_diff + j*K_*N_, (Dtype)0.,
          bottom[0]->mutable_gpu_diff() + j*N_*M_);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WordSumLayer);

}  // namespace caffe
