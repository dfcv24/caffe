#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/fea_sim_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeaSimLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "bottom[0] and bottom[1] has different channels";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "bottom[0] and bottom[1] has different channels";
  CHECK_EQ(bottom[0]->width(), 1) << "bottom[0] width does not equal to 1";
  CHECK_EQ(bottom[1]->width(), 1) << "bottom[1] width does not equal to 1";
  channels_ = bottom[0]->shape(1);
  vector<int> p_shape(1, channels_);
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
  }
  this->blobs_[0].reset(new Blob<Dtype>(p_shape));
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
    this->layer_param_.fea_sim_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());
}

template <typename Dtype>
void FeaSimLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
    << "Inputs must have the same dimension.";
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(channels_);
  sz[0] = bottom[0]->shape(0);
  batch_sum_multiplier_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);
  diff_.ReshapeLike(*bottom[0]);
  pow_diff_.ReshapeLike(*bottom[0]);
  win_diff_.ReshapeLike(*bottom[0]);
  temp_sub_.ReshapeLike(*bottom[0]);
  temp_p_.ReshapeLike(*bottom[0]);
  temp_blob_.ReshapeLike(*bottom[0]);

  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz[0] = spatial_dim;
    spatial_sum_multiplier_.Reshape(sz);
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
  }

  int numbychans = channels_*bottom[0]->shape(0);
  if (num_by_chans_.num_axes() == 0 ||
      num_by_chans_.shape(0) != numbychans) {
    sz[0] = numbychans;
    num_by_chans_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void FeaSimLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  //image1 minus image2
  caffe_sub(
    count,
    bottom[0]->cpu_data(),
    bottom[1]->cpu_data(),
    diff_.mutable_cpu_data());
  //square the diff
  caffe_powx(
    count,
    diff_.cpu_data(),
    Dtype(2),
    pow_diff_.mutable_cpu_data());
  //diff divide by param -p
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), this->blobs_[0]->cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_div(
    count,
    pow_diff_.cpu_data(),
    temp_.cpu_data(),
    win_diff_.mutable_cpu_data());
  //make a exp
  caffe_exp(
    count,
    win_diff_.cpu_data(),
    top[0]->mutable_cpu_data());
}

template <typename Dtype>
void FeaSimLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), this->blobs_[0]->cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., temp_p_.mutable_cpu_data());
  caffe_mul(count, top[0]->cpu_diff(), top[0]->cpu_data(), temp_.mutable_cpu_data());
  caffe_div(count, temp_.cpu_data(), temp_p_.cpu_data(), temp_.mutable_cpu_data());
  caffe_cpu_scale(count, Dtype(2), temp_.cpu_data(), temp_.mutable_cpu_data());
  //bottom[0]_diff
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), temp_sub_.mutable_cpu_data());
  caffe_mul(count, temp_.cpu_data(), temp_sub_.cpu_data(), bottom[0]->mutable_cpu_diff());
  //bottom[1]_diff
  caffe_sub(count, bottom[1]->cpu_data(), bottom[0]->cpu_data(), temp_sub_.mutable_cpu_data());
  caffe_mul(count, temp_.cpu_data(), temp_sub_.cpu_data(), bottom[1]->mutable_cpu_diff());
  //blob_diff
    //bottom[0]-bottom[1] pow2
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), temp_.mutable_cpu_data());
  caffe_powx(count, temp_.cpu_data(), Dtype(2), temp_.mutable_cpu_data());
    //blob[0]pow2
  caffe_powx(count, temp_p_.cpu_data(), Dtype(2), temp_p_.mutable_cpu_data());
    //blob[0]diff
  caffe_mul(count, top[0]->cpu_diff(), top[0]->cpu_data(), temp_blob_.mutable_cpu_data());
  caffe_mul(count, temp_blob_.cpu_data(), temp_.cpu_data(), temp_blob_.mutable_cpu_data());
  caffe_div(count, temp_blob_.cpu_data(), temp_p_.cpu_data(), temp_blob_.mutable_cpu_data());
  caffe_cpu_scale(count, Dtype(-1), temp_blob_.cpu_data(), temp_blob_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_, 1, spatial_dim ,1.,
    temp_blob_.cpu_data(), spatial_sum_multiplier_.cpu_data(), 0., num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, channels_, num, 1.,
    batch_sum_multiplier_.cpu_data(), num_by_chans_.cpu_data(), 0., this->blobs_[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(FeaSimLayer);
#endif

INSTANTIATE_CLASS(FeaSimLayer);
REGISTER_LAYER_CLASS(FeaSim);

}  // namespace caffe