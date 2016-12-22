#include <vector>

#include "caffe/layers/square_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SquareLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[1]->width(), bottom[1]->width());
}

template <typename Dtype>
void SquareLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  temp_.ReshapeLike(*bottom[0]);
  count = bottom[0].count();
}

template <typename Dtype>
void SquareLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  caffe_gpu_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), temp_.mutable_cpu_data());
  caffe_gpu_powx(count, temp_.cpu_data(), Dtype(2), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void SquareLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  caffe_gpu_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), temp_.mutable_cpu_data());
  caffe_gpu_scale(count, Dtype(2), temp_.cpu_data(), temp_.mutable_cpu_data());
  caffe_gpu_mul(count, top[0]->cpu_diff(), temp_.cpu_data(), bottom[0]->mutable_cpu_diff());
  caffe_gpu_sub(count, bottom[1]->cpu_data(), bottom[0]->cpu_data(), temp_.mutable_cpu_data());
  caffe_gpu_scale(count, Dtype(2), temp_.cpu_data(), temp_.mutable_cpu_data());
  caffe_gpu_mul(count, top[0]->cpu_diff(), temp_.cpu_data(), bottom[1]->mutable_cpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(SquareLayer);
}  // namespace caffe
