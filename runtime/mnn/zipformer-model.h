// runtime/mnn/zipformer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2025  frankyj@foxmail.com  (authors: Jian You)

#ifndef SHERPA_DEPLOY_MNN_ZIPFORMER_MODEL_H_
#define SHERPA_DEPLOY_MNN_ZIPFORMER_MODEL_H_
#include <string>
#include <utility>
#include <vector>

#include "model.h"

namespace MNN {
  class Tensor;
  struct ScheduleConfig;
  class Session;
  class Interpreter;
}

namespace SherpaDeploy {
class ZipformerModel : public Model {
 public:
  explicit ZipformerModel(const ModelConfig &config);
#if __ANDROID_API__ >= 9
  ZipformerModel(AAssetManager *mgr, const ModelConfig &config);
#endif

  // MNN forbids Tensor's copy/move constructor and copy/move assignment
  // so should only use Tensor pointer to store or transfer Tensors.

  std::vector<TensorPtr> GetEncoderInitStates() const override;

  std::pair<TensorPtr, std::vector<TensorPtr>> RunEncoder(
      TensorPtr features, const std::vector<TensorPtr>& states) override;

  TensorPtr RunDecoder(TensorPtr decoder_input) override;

  TensorPtr RunJoiner(TensorPtr encoder_out, TensorPtr decoder_out) override;

  int32_t Segment() const override {
    // T_ = decode_chunk_length_ + pad_length_;
    return T_;
  }

  // Advance the feature extract by this number of frames after
  // running the encoder network
  int32_t Offset() const override { return decode_chunk_length_; }

  int32_t ContextSize() const override { return context_size_; }

 private:
  void InitEncoder(const char* model_path, const MNN::ScheduleConfig& schedule_config);
  void InitDecoder(const char* model_path, const MNN::ScheduleConfig& schedule_config);
  void InitJoiner(const char* model_path, const MNN::ScheduleConfig& schedule_config);

#if __ANDROID_API__ >= 9
  void InitEncoder(AAssetManager *mgr, const std::string &encoder_param,
                   const std::string &encoder_bin);
  void InitDecoder(AAssetManager *mgr, const std::string &decoder_param,
                   const std::string &decoder_bin);
  void InitJoiner(AAssetManager *mgr, const std::string &joiner_param,
                  const std::string &joiner_bin);
#endif

  std::vector<TensorPtr> GetEncoderInitStates1() const;
  std::vector<TensorPtr> GetEncoderInitStates2() const;

 private:
  std::unique_ptr<MNN::Interpreter> encoder_net_;
  std::unique_ptr<MNN::Interpreter> decoder_net_;
  std::unique_ptr<MNN::Interpreter> joiner_net_;

  MNN::Session* encoder_sess_ = nullptr;
  MNN::Session* decoder_sess_ = nullptr;
  MNN::Session* joiner_sess_ = nullptr;

  std::string model_type_ = "zipformer"; 

  int32_t decode_chunk_length_ = 32; 
  int32_t T_ = 45; // T_ = decode_chunk_length_ + pad_length_

  int32_t feature_dim_ = 80;

  // common for zipformer & zipformer2
  std::vector<int32_t> num_encoder_layers_;             
  std::vector<int32_t> encoder_dims_;                   
  std::vector<int32_t> cnn_module_kernels_;              
  std::vector<int32_t> left_context_len_;
  // zipformer
  std::vector<int32_t> attention_dims_;
  // zipformer2
  std::vector<int32_t> num_heads_; 
  std::vector<int32_t> query_head_dims_;
  std::vector<int32_t> value_head_dims_;

  int32_t context_size_ = 0;
  int32_t vocab_size_ = 0;

  std::vector<std::string> encoder_input_names_;
  std::vector<std::string> encoder_output_names_;
  std::vector<std::string> decoder_input_names_;
  std::vector<std::string> decoder_output_names_;
  std::vector<std::string> joiner_input_names_;
  std::vector<std::string> joiner_output_names_;
};

}  // namespace SherpaDeploy

#endif  // SHERPA_DEPLOY_MNN_ZIPFORMER_MODEL_H_
