#include "caffe/filler.hpp"
#include "caffe/layers/word_sum_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WordSumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  bias_term_ = this->layer_param_.word_sum_param().bias_term();
  K_ = this->layer_param_.word_sum_param().num_output(); // New Embedding Size
  B_ = bottom[0]->shape(0); // Batch Size
  M_ = bottom[0]->shape(1); // Word Size
  N_ = bottom[0]->shape(2); // Embedding Size
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    } 
    // Initialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = K_;
    weight_shape[1] = M_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.word_sum_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    if (bias_term_) {
      vector<int> bias_shape(1, K_*N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.word_sum_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  
  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void WordSumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = K_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, B_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(B_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void WordSumLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int j=0; j<bottom[0]->shape(0); ++j){
      // W (K_,M_) * X (M_,N_)
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
        weight, bottom_data + j*M_*N_, (Dtype)0., top_data + j*K_*N_);
  }
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, B_, K_*N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(), this->blobs_[1]->cpu_data(),
        (Dtype)1., top_data);
  }
}

template <typename Dtype>
void WordSumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  if (this->param_propagate_down_[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    for (int j=0; j<bottom[0]->shape(0); ++j){
      // D (K_*N) * X (M,N)  // change 1 to K_
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, M_, N_, (Dtype)1.,
          top_diff + j*K_*N_, bottom_data + j*M_*N_, (Dtype)1.,
          this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    // Gradient with respect to bias
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_*N_, B_, (Dtype)1.,
        bias_multiplier_.cpu_data(), top_diff, (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    for (int j=0; j<bottom[0]->shape(0); ++j){  // B_ batches
      // W (1*M) * D (1,N)  // this should be (K_*M_) * (K_,N_)
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, M_, N_, K_, (Dtype)1.,
          this->blobs_[0]->cpu_data(), top_diff + j*K_*N_, (Dtype)0.,
          bottom[0]->mutable_cpu_diff() + j*N_*M_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WordSumLayer);
#endif

INSTANTIATE_CLASS(WordSumLayer);
REGISTER_LAYER_CLASS(WordSum);

}  // namespace caffe
