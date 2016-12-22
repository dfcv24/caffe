#include <vector>

#include "caffe/layers/siamese_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SiameseLabelLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SiameseLabelLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SiameseLabelLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* a = bottom[0]->cpu_data();
  const Dtype* b = bottom[1]->cpu_data();
  Dtype* out = top[0]->mutable_cpu_data();
  int num = bottom[0]->shape(0);
  for(int i=0; i<num; i++,a++,b++,out++)
    *out = (*a == *b) ? 0 : 1;
}

template <typename Dtype>
void SiameseLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(SiameseLabelLayer);
#endif

INSTANTIATE_CLASS(SiameseLabelLayer);
REGISTER_LAYER_CLASS(SiameseLabel);
}  // namespace caffe
