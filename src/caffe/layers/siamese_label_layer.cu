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
#if 0
template <typename Dtype>
void SiameseLabelLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->shape(0);
  std::cout << num << std::endl;
  const Dtype* a = bottom[0]->gpu_data();
  const Dtype* b = bottom[1]->gpu_data();
  Dtype* out = top[0]->mutable_gpu_data();
  //for(int i = 0; i < num; i++)
   // out[i] = (a[i] == b[i]) ? Dtype(0) : Dtype(1);
  std::cout << num << std::endl;
}

template <typename Dtype>
void SiameseLabelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}
#endif
//INSTANTIATE_LAYER_GPU_FUNCS(SiameseLabelLayer);

}  // namespace caffe
