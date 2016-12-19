#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/fea_sim_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeaSimLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  //image1 minus image2
  caffe_gpu_sub(
    count,
    bottom[0]->gpu_data(),
    bottom[1]->gpu_data(),
    diff_.mutable_gpu_data());
  //square the diff
  caffe_gpu_powx(
    count,
    diff_.gpu_data(),
    Dtype(2),
    pow_diff_.mutable_gpu_data());
  //diff divide by param -p^2
  caffe_gpu_powx(
    channels_,
    this->blobs_[0]->gpu_data(),
    Dtype(2),
    pow_p_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), pow_p_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
  caffe_gpu_div(
    count,
    pow_diff_.gpu_data(),
    temp_.gpu_data(),
    win_diff_.mutable_gpu_data());
  caffe_gpu_scale(
    count,
    Dtype(-1),
    win_diff_.gpu_data(),
    win_diff_.mutable_gpu_data());
  //make a exp
  caffe_gpu_exp(
    count,
    win_diff_.gpu_data(),
    top[0]->mutable_gpu_data());
  //LOG(INFO) << "top[0]->count: " << top[0]->count();
}

template <typename Dtype>
void FeaSimLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  int num = bottom[0]->shape(0);
  CHECK_GT(num, 0) << "height gets the error value";
  CHECK_GT(count, 0) << "count gets the error value";
  //LOG(INFO) << "the program had come here";
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), this->blobs_[0]->gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp_p_.mutable_gpu_data());
  caffe_gpu_mul(count, top[0]->gpu_diff(), top[0]->gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_div(count, temp_.gpu_data(), temp_p_.gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_scale(count, Dtype(2), temp_.gpu_data(), temp_.mutable_gpu_data());
  //bottom[0]_diff
  caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), temp_sub_.mutable_gpu_data());
  caffe_gpu_mul(count, temp_.gpu_data(), temp_sub_.gpu_data(), bottom[0]->mutable_gpu_diff());
  //bottom[1]_diff
  caffe_gpu_sub(count, bottom[1]->gpu_data(), bottom[0]->gpu_data(), temp_sub_.mutable_gpu_data());
  caffe_gpu_mul(count, temp_.gpu_data(), temp_sub_.gpu_data(), bottom[1]->mutable_gpu_diff());
  //blob_diff
    //bottom[0]-bottom[1] pow2
  caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_powx(count, temp_.gpu_data(), Dtype(2), temp_.mutable_gpu_data());
    //blob[0]pow2
  caffe_gpu_powx(count, temp_p_.gpu_data(), Dtype(3), temp_p_.mutable_gpu_data());
    //blob[0]diff
  caffe_gpu_mul(count, top[0]->gpu_diff(), top[0]->gpu_data(), temp_blob_.mutable_gpu_data());
  caffe_gpu_mul(count, temp_blob_.gpu_data(), temp_.gpu_data(), temp_blob_.mutable_gpu_data());
  caffe_gpu_div(count, temp_blob_.gpu_data(), temp_p_.gpu_data(), temp_blob_.mutable_gpu_data());
  caffe_gpu_scale(count, Dtype(2), temp_blob_.gpu_data(), temp_blob_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_, 1, spatial_dim ,1.,
    temp_blob_.gpu_data(), spatial_sum_multiplier_.gpu_data(), 0., num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, channels_, num, 1.,
    batch_sum_multiplier_.gpu_data(), num_by_chans_.gpu_data(), 0., this->blobs_[0]->mutable_gpu_diff());
  //LOG(INFO) << "the program had come here";
}
INSTANTIATE_LAYER_GPU_FUNCS(FeaSimLayer);

}  // namespace caffe