#include <vector>

#include "caffe/layers/gate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GateLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->shape(0);
  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num * height,
      width, 1, 1., bottom[1]->gpu_data(),
      width_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
  //Dtype* dot = mul_temp_.mutable_gpu_data();
  caffe_gpu_mul(count, bottom[0]->gpu_data(), temp_.gpu_data(), mul_temp_.mutable_gpu_data());
  caffe_gpu_add(
    count,
    bottom[0]->gpu_data(),
    mul_temp_.gpu_data(),
    top[0]->mutable_gpu_data());
}

template <typename Dtype>
void GateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();    
  int num = bottom[0]->shape(0);
  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);
  CHECK_GT(width, 0) << "width getst the error value";
  CHECK_GT(height, 0) << "height gets the error value";
  CHECK_GT(num, 0) << "height gets the error value";
  CHECK_GT(count, 0) << "count gets the error value";
  // LOG(INFO) << "the program had come here";
  //gate diff
  caffe_gpu_mul(count, top[0]->gpu_diff(), bottom[0]->gpu_data(), temp_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, count / width , 1, width, 1.,
      temp_.gpu_data(), width_sum_multiplier_.gpu_data(), 0., bottom[1]->mutable_gpu_diff());
  //prelu diff
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num * height,
      width, 1, 1., bottom[1]->gpu_data(),
      width_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
  caffe_gpu_mul(count, top[0]->gpu_diff(), temp_.gpu_data(), mul_temp_.mutable_gpu_data());
  caffe_gpu_add(count, top[0]->gpu_diff(), mul_temp_.gpu_data(), bottom[0]->mutable_gpu_diff());
  // LOG(INFO) << "the program had come here";

}

INSTANTIATE_LAYER_GPU_FUNCS(GateLayer);
}  // namespace caffe
