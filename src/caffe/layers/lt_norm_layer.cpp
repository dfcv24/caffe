#include <vector>

#include "caffe/layers/lt_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LtNormLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  count_ = bottom[0]->count();
  num_ = bottom[0]->shape(0);
  //channels_ = bottom[0]->shape(1);
}

template <typename Dtype>
void LtNormLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  pow_temp_.ReshapeLike(*bottom[0]);
  mul_temp_.ReshapeLike(*bottom[0]);
  vector<int> sz;
  sz.push_back(count_ / num_);
  piece_sum_multiplier_.Reshape(sz);
  sz[0] = num_;
  temp_.Reshape(sz);
  int numbypiece = count_ / num_;
  if (num_by_piece_.num_axes() == 0 ||
      num_by_piece_.shape(0) != numbypiece) {
    sz[0] = numbypiece;
    num_by_piece_.Reshape(sz);
    caffe_set(piece_sum_multiplier_.count(), Dtype(1),
        piece_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void LtNormLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->shape(0);
  //L2 normalization
  caffe_powx(count, bottom[0]->cpu_data(), Dtype(2), pow_temp_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, 1, count / num, 1.,
      pow_temp_.cpu_data(), piece_sum_multiplier_.cpu_data(), 0.,
      num_by_piece_.mutable_cpu_data());
  caffe_sqr(num, num_by_piece_.cpu_data(), temp_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, count / num, 1, 1.,
      temp_.cpu_data(), piece_sum_multiplier_.cpu_data(), 0.,
      mul_temp_.mutable_cpu_data());
  caffe_div(count, bottom[0]->cpu_data(), mul_temp_.cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void LtNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();    
  int num = bottom[0]->shape(0);
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* dot = temp_.mutable_cpu_data();
  for (int i = 0; i< num; i++) {
    *dot = caffe_cpu_dot(count / num, top_data, top_data);
    top_data += count / num;
    ++dot;
  }
  CHECK_GT(num, 0) << "height gets the error value";
  CHECK_GT(count, 0) << "count gets the error value";
  //LOG(INFO) << "the program had come here";
  //dot_ = caffe_cpu_dot(count / num, top[0]->cpu_data(), top[0]->cpu_diff());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, count / num, 1, 1.,
    temp_.cpu_data(), piece_sum_multiplier_.cpu_data(), 0.,
    mul_temp_.mutable_cpu_data());
  caffe_sub(count, top[0]->cpu_diff(), mul_temp_.cpu_data(), mul_temp_.mutable_cpu_data());

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, 1, count / num, 1.,
      bottom[0]->cpu_data(), piece_sum_multiplier_.cpu_data(), 0.,
      num_by_piece_.mutable_cpu_data());
  caffe_sqr(num, num_by_piece_.cpu_data(), temp_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, count / num, 1, 1.,
      temp_.cpu_data(), piece_sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_div(count, mul_temp_.cpu_data(), temp_.cpu_data(), bottom[0]->mutable_cpu_diff());
  //LOG(INFO) << "the program had come here";
}

#ifdef CPU_ONLY
STUB_GPU(LtNormLayer);
#endif

INSTANTIATE_CLASS(LtNormLayer);
REGISTER_LAYER_CLASS(LtNorm);

}  // namespace caffe
