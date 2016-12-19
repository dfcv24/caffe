#include <vector>

#include "caffe/layers/lt_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LtNormLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->shape(0);
  //L2 normalization
  caffe_gpu_powx(count, bottom[0]->gpu_data(), Dtype(2), pow_temp_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, 1, count / num, 1.,
      pow_temp_.gpu_data(), piece_sum_multiplier_.gpu_data(), 0.,
      num_by_piece_.mutable_gpu_data());
  caffe_gpu_powx(num, num_by_piece_.gpu_data(), Dtype(0.5), temp_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, count / num, 1, 1.,
      temp_.gpu_data(), piece_sum_multiplier_.gpu_data(), 0.,
      mul_temp_.mutable_gpu_data());
  caffe_gpu_div(count, bottom[0]->gpu_data(), mul_temp_.gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void LtNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();    
  int num = bottom[0]->shape(0);
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* dot = temp_.mutable_gpu_data();
  for (int i = 0; i< num; i++) {
    caffe_gpu_dot(count / num, top_data, top_data, dot);
    top_data += count / num;
    ++dot;
  }
  CHECK_GT(num, 0) << "height gets the error value";
  CHECK_GT(count, 0) << "count gets the error value";
  //LOG(INFO) << "the program had come here";
  //dot_ = caffe_gpu_dot(count / num, top[0]->gpu_data(), top[0]->gpu_diff());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, count / num, 1, 1.,
    temp_.gpu_data(), piece_sum_multiplier_.gpu_data(), 0.,
    mul_temp_.mutable_gpu_data());
  caffe_gpu_sub(count, top[0]->gpu_diff(), mul_temp_.gpu_data(), mul_temp_.mutable_gpu_data());

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, 1, count / num, 1.,
      bottom[0]->gpu_data(), piece_sum_multiplier_.gpu_data(), 0.,
      num_by_piece_.mutable_gpu_data());
  caffe_gpu_powx(num, num_by_piece_.gpu_data(), Dtype(0.5), temp_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, count / num, 1, 1.,
      temp_.gpu_data(), piece_sum_multiplier_.gpu_data(), 0.,
      temp_.mutable_gpu_data());
  caffe_gpu_div(count, mul_temp_.gpu_data(), temp_.gpu_data(), bottom[0]->mutable_gpu_diff());
  //LOG(INFO) << "the program had come here";
}
INSTANTIATE_LAYER_GPU_FUNCS(LtNormLayer);

}  // namespace caffe
