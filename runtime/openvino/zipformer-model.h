// runtime/mnn/zipformer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2025  frankyj@foxmail.com  (authors: Jian You)

#ifndef SHERPA_DEPLOY_OPENVINO_ZIPFORMER_MODEL_H_
#define SHERPA_DEPLOY_OPENVINO_ZIPFORMER_MODEL_H_
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

  std::vector<ov::Tensor> GetEncoderInitStates() const override;

  std::pair<ov::Tensor, std::vector<ov::Tensor>> RunEncoder(
      ov::Tensor features, const std::vector<ov::Tensor>& states) override;

  ov::Tensor RunDecoder(ov::Tensor decoder_input) override;

  ov::Tensor RunJoiner(ov::Tensor encoder_out, ov::Tensor decoder_out) override;

  int32_t Segment() const override {
    // pad_length 7, because the subsampling expression is
    // ((x_len - 7) // 2 + 1)//2, we need to pad 7 frames
    //
    // decode chunk length before subsample is 32 frames
    //
    // So each segment is pad_length + decode_chunk_length = 7 + 32 = 39
    // return decode_chunk_length_ + pad_length_;
    return T_;
  }

  // Advance the feature extract by this number of frames after
  // running the encoder network
  int32_t Offset() const override { return decode_chunk_length_; }

  int32_t ContextSize() const override { return context_size_; }

 private:
  void InitEncoder(const std::string& ir_path);
  void InitDecoder(const std::string& ir_path);
  void InitJoiner(const std::string& ir_path);

  std::vector<ov::Tensor> GetEncoderInitStates1() const;
  std::vector<ov::Tensor> GetEncoderInitStates2() const;

#if __ANDROID_API__ >= 9
  void InitEncoder(AAssetManager *mgr, const std::string &encoder_param,
                   const std::string &encoder_bin);
  void InitDecoder(AAssetManager *mgr, const std::string &decoder_param,
                   const std::string &decoder_bin);
  void InitJoiner(AAssetManager *mgr, const std::string &joiner_param,
                  const std::string &joiner_bin);
#endif

 private:
  // ov::AnyMap cpu_config_;
  std::shared_ptr<ov::Core> core_;
  std::string device_; // CPU | GPU | AUTO
  // ov::Core core_;

  std::shared_ptr<ov::CompiledModel> encoder_compile_model_;
  std::shared_ptr<ov::CompiledModel> decoder_compile_model_;
  std::shared_ptr<ov::CompiledModel> joiner_compile_model_;

  std::shared_ptr<ov::InferRequest> encoder_infer_;
  std::shared_ptr<ov::InferRequest> decoder_infer_;
  std::shared_ptr<ov::InferRequest> joiner_infer_;

  std::string model_type_ = "zipformer";

  int32_t decode_chunk_length_ = 32; 
  int32_t T_ = 39; // T_ = decode_chunk_length_ + pad_length_

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
  // std::map<std::string, ov::Output<const ov::Node>> encoder_inputs_map_;
  // std::map<std::string, ov::Output<const ov::Node>> decoder_inputs_map_;
  // std::map<std::string, ov::Output<const ov::Node>> joiner_inputs_map_;

  // std::vector<ov::Output<const ov::Node>> encoder_inputs_;
  // std::vector<ov::Output<const ov::Node>> decoder_inputs_;
  // std::vector<ov::Output<const ov::Node>> joiner_inputs_;
};

}  // namespace SherpaDeploy

#endif  // SHERPA_DEPLOY_OPENVINO_ZIPFORMER_MODEL_H_
