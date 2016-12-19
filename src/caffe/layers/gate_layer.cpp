#include <vector>

#include "caffe/layers/gate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GateLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[1]->width(), 1);
  channels_ = bottom[0]->shape(1);
}

template <typename Dtype>
void GateLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> sz;
  temp_.ReshapeLike(*bottom[0]);
  mul_temp_.ReshapeLike(*bottom[0]);
  int numbywidth = bottom[0]->shape(3);
  if (width_sum_multiplier_.num_axes() == 0 ||
      width_sum_multiplier_.shape(0) != numbywidth) {
    sz.push_back(numbywidth);
    width_sum_multiplier_.Reshape(sz);
    caffe_set(width_sum_multiplier_.count(), Dtype(1),
        width_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void GateLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->shape(0);
  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num * height,
      width, 1, 1., bottom[1]->cpu_data(),
      width_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  //Dtype* dot = mul_temp_.mutable_cpu_data();
  caffe_mul(count, bottom[0]->cpu_data(), temp_.cpu_data(), mul_temp_.mutable_cpu_data());
  caffe_add(
    count,
    bottom[0]->cpu_data(),
    mul_temp_.cpu_data(),
    top[0]->mutable_cpu_data());
}

template <typename Dtype>
void GateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
  caffe_mul(count, top[0]->cpu_diff(), bottom[0]->cpu_data(), temp_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, count / width , 1, width, 1.,
      temp_.cpu_data(), width_sum_multiplier_.cpu_data(), 0., bottom[1]->mutable_cpu_diff());
  //prelu diff
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num * height,
      width, 1, 1., bottom[1]->cpu_data(),
      width_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_mul(count, top[0]->cpu_diff(), temp_.cpu_data(), mul_temp_.mutable_cpu_data());
  caffe_add(count, top[0]->cpu_diff(), mul_temp_.cpu_data(), bottom[0]->mutable_cpu_diff());
  // LOG(INFO) << "the program had come here";

}

#ifdef CPU_ONLY
STUB_GPU(GateLayer);
#endif

INSTANTIATE_CLASS(GateLayer);
REGISTER_LAYER_CLASS(Gate);
}  // namespace caffe
