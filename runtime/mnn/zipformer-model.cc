// runtime/mnn/zipformer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2025  frankyj@foxmail.com  (authors: Jian You)

#include "zipformer-model.h"
#include "mnn-utils.h"

#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <numeric>

#include "MNN/expr/Module.hpp"       // NOLINT
#include "MNN/Interpreter.hpp"  // NOLINT

// #define PRINT_MODEL_METADATA

static std::vector<int> convertStringToVector(const std::string& input) {
  std::vector<int> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, ',')) {
    result.push_back(std::stoi(token));
  }

  return result;
}

namespace SherpaDeploy {

ZipformerModel::ZipformerModel(const ModelConfig &config) {

  InitEncoder(config.encoder_mnn.c_str(), config.schedule_config);
  InitDecoder(config.decoder_mnn.c_str(), config.schedule_config);
  InitJoiner(config.joiner_mnn.c_str(), config.schedule_config);

}

#if __ANDROID_API__ >= 9
ZipformerModel::ZipformerModel(AAssetManager *mgr, const ModelConfig &config) {

  InitEncoder(mgr, config.encoder_mnn.c_str(), config.schedule_config);
  InitDecoder(mgr, config.decoder_mnn.c_str(), config.schedule_config);
  InitJoiner(mgr, config.joiner_mnn.c_str(), config.schedule_config);

}
#endif

std::pair<TensorPtr, std::vector<TensorPtr>> ZipformerModel::RunEncoder(
    TensorPtr features, const std::vector<TensorPtr>& states) {
  std::vector<TensorPtr> _states;

  if (states.empty()) {
    _states = GetEncoderInitStates();
  } else {
    _states = states;
  }

  auto featuresTensor = encoder_net_->getSessionInput(encoder_sess_, encoder_input_names_[0].c_str());
  featuresTensor->copyFromHostTensor(features.get()); 
  for (size_t i = 1; i < encoder_input_names_.size(); ++i) {
    auto inputTensor = encoder_net_->getSessionInput(encoder_sess_, encoder_input_names_[i].c_str());

    // for using dynamic axes when export ONNX
    // auto shape = inputTensor->shape();
    // std::replace(shape.begin(), shape.end(), -1, 1);
    // encoder_net_->resizeTensor(inputTensor, shape);
    // encoder_net_->resizeSession(encoder_sess_);

    inputTensor->copyFromHostTensor(_states[i-1].get());
  } 

  // run network
  encoder_net_->runSession(encoder_sess_);

  auto encoderOutTensor = encoder_net_->getSessionOutput(encoder_sess_, encoder_output_names_[0].c_str());
  TensorPtr encoderOutTensor_host = TensorPtr(
                                        MNN::Tensor::create(
                                        encoderOutTensor->shape(), 
                                        encoderOutTensor->getType(), 
                                        nullptr, 
                                        encoderOutTensor->getDimensionType())
                                        );
  encoderOutTensor->copyToHostTensor(encoderOutTensor_host.get());

  std::vector<TensorPtr> nextStatesTensor_host(_states.size());
  for (size_t i = 1; i < encoder_output_names_.size(); ++i) {
    auto nextStateTensor = encoder_net_->getSessionOutput(encoder_sess_, encoder_output_names_[i].c_str());
    nextStatesTensor_host[i-1] = TensorPtr(MNN::Tensor::create(
                                  nextStateTensor->shape(), 
                                  nextStateTensor->getType(), 
                                  nullptr, 
                                  nextStateTensor->getDimensionType()));
    nextStateTensor->copyToHostTensor(nextStatesTensor_host[i-1].get());
  }

  return {encoderOutTensor_host, nextStatesTensor_host};
}

TensorPtr ZipformerModel::RunDecoder(TensorPtr decoder_input) {

  auto decoderInputTensor = decoder_net_->getSessionInput(decoder_sess_, decoder_input_names_[0].c_str());
  decoderInputTensor->copyFromHostTensor(decoder_input.get()); 

  decoder_net_->runSession(decoder_sess_);

  auto decoderOutTensor = decoder_net_->getSessionOutput(decoder_sess_, decoder_output_names_[0].c_str());
  TensorPtr decoderOutTensor_host = TensorPtr(MNN::Tensor::create(
                                        decoderOutTensor->shape(), 
                                        decoderOutTensor->getType(), 
                                        nullptr, 
                                        decoderOutTensor->getDimensionType()));  
  decoderOutTensor->copyToHostTensor(decoderOutTensor_host.get());


  return decoderOutTensor_host;
}

TensorPtr ZipformerModel::RunJoiner(TensorPtr encoder_out, TensorPtr decoder_out) {

  auto encoderOutTensor = joiner_net_->getSessionInput(joiner_sess_, joiner_input_names_[0].c_str());
  auto decoderOutTensor = joiner_net_->getSessionInput(joiner_sess_, joiner_input_names_[1].c_str());
  encoderOutTensor->copyFromHostTensor(encoder_out.get());
  decoderOutTensor->copyFromHostTensor(decoder_out.get());

  joiner_net_->runSession(joiner_sess_);

  auto joinerOutTensor = joiner_net_->getSessionOutput(joiner_sess_, joiner_output_names_[0].c_str());
  TensorPtr joinerOutTensor_host = TensorPtr(MNN::Tensor::create(
                                        joinerOutTensor->shape(), 
                                        joinerOutTensor->getType(), 
                                        nullptr, 
                                        joinerOutTensor->getDimensionType())); 
  joinerOutTensor->copyToHostTensor(joinerOutTensor_host.get());

  return joinerOutTensor_host;
}

void ZipformerModel::InitEncoder(const char* model_path, const MNN::ScheduleConfig& schedule_config) {

  InitNet(encoder_net_, encoder_sess_, model_path, schedule_config);

  std::vector<std::string> empty;
  std::shared_ptr<MNN::Express::Module> module(MNN::Express::Module::load(empty, empty, model_path));
  if (nullptr == module.get()) {
    fprintf(stderr, "Load MNN from %s Failed\n", model_path);
    return;
  }

  GetInputNames(module, encoder_input_names_);
  GetOutputNames(module, encoder_output_names_);

  auto info = module->getInfo();
  // MNN_ASSERT(info->inputNames.size() == info->inputs.size());

  if (!info->metaData.empty()) {
#ifdef PRINT_MODEL_METADATA
    fprintf(stderr, "\n------------ Encoder MetaData: Begin ------------\n");
#endif
    for (auto& iter : info->metaData) {
#ifdef PRINT_MODEL_METADATA
      fprintf(stderr, "[Meta] %s : %s\n", iter.first.c_str(), iter.second.c_str());
#endif
      if (iter.first == "model_type") {
        model_type_ = iter.second;
        fprintf(stderr, "/*********** %s ***********/\n", model_type_.c_str());
      }
      else if (iter.first == "attention_dims") {
        attention_dims_ = std::move(convertStringToVector(iter.second));
      }
      else if (iter.first == "num_heads") {
        num_heads_ = std::move(convertStringToVector(iter.second));
      }
      else if (iter.first == "query_head_dims") {
        query_head_dims_ = std::move(convertStringToVector(iter.second));
      }
      else if (iter.first == "value_head_dims") {
        value_head_dims_ = std::move(convertStringToVector(iter.second));
      }
      else if (iter.first == "cnn_module_kernels") {
        cnn_module_kernels_ = std::move(convertStringToVector(iter.second));
      }
      else if (iter.first == "decode_chunk_len") {
        decode_chunk_length_ = std::stoi(iter.second);
      }
      else if (iter.first == "T") {
        T_ = std::stoi(iter.second);
      }
      else if (iter.first == "encoder_dims") {
        encoder_dims_ = std::move(convertStringToVector(iter.second));
      }
      else if (iter.first == "left_context_len") {
        left_context_len_ = std::move(convertStringToVector(iter.second));
      }
      else if (iter.first == "num_encoder_layers") {
        num_encoder_layers_ = std::move(convertStringToVector(iter.second));
      }
    }
#ifdef PRINT_MODEL_METADATA
    fprintf(stderr, "------------ Encoder MetaData: End ------------\n");
#endif
  }
}

void ZipformerModel::InitDecoder(const char* model_path, const MNN::ScheduleConfig& schedule_config) {
  InitNet(decoder_net_, decoder_sess_, model_path, schedule_config);

  std::vector<std::string> empty;
  std::shared_ptr<MNN::Express::Module> module(MNN::Express::Module::load(empty, empty, model_path));
  if (nullptr == module.get()) {
    fprintf(stderr, "Load MNN from %s Failed\n", model_path);
    return;
  }

  GetInputNames(module, decoder_input_names_);
  GetOutputNames(module, decoder_output_names_);

  auto info = module->getInfo();
  // MNN_ASSERT(info->inputNames.size() == info->inputs.size());
  if (!info->metaData.empty()) {
#ifdef PRINT_MODEL_METADATA
    fprintf(stderr, "\n------------ Decoder MetaData: Begin ------------\n");
#endif
    for (auto& iter : info->metaData) {
#ifdef PRINT_MODEL_METADATA
      fprintf(stderr, "[Meta] %s : %s\n", iter.first.c_str(), iter.second.c_str());
#endif
      if (iter.first == "context_size") {
        context_size_ = std::stoi(iter.second);
      }
      else if (iter.first == "vocab_size") {
        vocab_size_ = std::stoi(iter.second);
      }
    }
#ifdef PRINT_MODEL_METADATA
    fprintf(stderr, "------------ Decoder MetaData: End ------------\n");
#endif
  }
}

void ZipformerModel::InitJoiner(const char* model_path, const MNN::ScheduleConfig& schedule_config) {
  InitNet(joiner_net_, joiner_sess_, model_path, schedule_config);

  std::vector<std::string> empty;
  std::shared_ptr<MNN::Express::Module> module(MNN::Express::Module::load(empty, empty, model_path));
  if (nullptr == module.get()) {
    fprintf(stderr, "Load MNN from %s Failed\n", model_path);
    return;
  }

  GetInputNames(module, joiner_input_names_);
  GetOutputNames(module, joiner_output_names_);

  auto info = module->getInfo();
  // MNN_ASSERT(info->inputNames.size() == info->inputs.size());
#ifdef PRINT_MODEL_METADATA
  if (!info->metaData.empty()) {
    fprintf(stderr, "\n------------ Decoder MetaData: Begin ------------\n");
    for (auto& iter : info->metaData) {
      fprintf(stderr, "[Meta] %s : %s\n", iter.first.c_str(), iter.second.c_str());
    }
    fprintf(stderr, "------------ Decoder MetaData: End ------------\n");
  }
#endif
}

#if __ANDROID_API__ >= 9
// TODO: do we need to define below methods for android like NCNN?

// void ZipformerModel::InitEncoder(AAssetManager *mgr,
//                                  const std::string &encoder_param,
//                                  const std::string &encoder_bin) {
//   RegisterCustomLayers(encoder_);
//   InitNet(mgr, encoder_, encoder_param, encoder_bin);
//   InitEncoderPostProcessing();
// }

// void ZipformerModel::InitDecoder(AAssetManager *mgr,
//                                  const std::string &decoder_param,
//                                  const std::string &decoder_bin) {
//   InitNet(mgr, decoder_, decoder_param, decoder_bin);
// }

// void ZipformerModel::InitJoiner(AAssetManager *mgr,
//                                 const std::string &joiner_param,
//                                 const std::string &joiner_bin) {
//   InitNet(mgr, joiner_, joiner_param, joiner_bin);
// }
#endif

std::vector<TensorPtr> ZipformerModel::GetEncoderInitStates() const {
  if (model_type_ == "zipformer") {
    return GetEncoderInitStates1();
  }
  else if (model_type_ == "zipformer2") {
    return GetEncoderInitStates2();
  }
}

// see
// https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py#L673
std::vector<TensorPtr> ZipformerModel::GetEncoderInitStates1() const {
  // each layer has 7 states:
  // cached_len, (num_layers,)
  // cached_avg, (num_layers, encoder_dim)
  // cached_key, (num_layers, left_context_length, attention_dim)
  // cached_val, (num_layers, left_context_length, attention_dim / 2)
  // cached_val2, (num_layers, left_context_length, attention_dim / 2)
  // cached_conv1, (num_layers, encoder_dim, cnn_module_kernel_ - 1)
  // cached_conv2, (num_layers, encoder_dim, cnn_module_kernel_ - 1)

  std::vector<TensorPtr> cached_len_vec;
  std::vector<TensorPtr> cached_avg_vec;
  std::vector<TensorPtr> cached_key_vec;
  std::vector<TensorPtr> cached_val_vec;
  std::vector<TensorPtr> cached_val2_vec;
  std::vector<TensorPtr> cached_conv1_vec;
  std::vector<TensorPtr> cached_conv2_vec;

  cached_len_vec.reserve(num_encoder_layers_.size());
  cached_avg_vec.reserve(num_encoder_layers_.size());
  cached_key_vec.reserve(num_encoder_layers_.size());
  cached_val_vec.reserve(num_encoder_layers_.size());
  cached_val2_vec.reserve(num_encoder_layers_.size());
  cached_conv1_vec.reserve(num_encoder_layers_.size());
  cached_conv2_vec.reserve(num_encoder_layers_.size());

  for (size_t i = 0; i != num_encoder_layers_.size(); ++i) {
    int32_t num_layers = num_encoder_layers_[i];
    int32_t attention_dim = attention_dims_[i];
    int32_t left_context_len = left_context_len_[i];
    int32_t encoder_dim = encoder_dims_[i];
    int32_t cnn_module_kernel = cnn_module_kernels_[i];

    // dims and dims_type info is from the output info of "./GetMNNInfo encoder.mnn"
    auto cached_len = TensorPtr(MNN::Tensor::create<int32_t>({num_layers, 1}, NULL, MNN::Tensor::CAFFE));
    auto cached_avg = TensorPtr(MNN::Tensor::create<float>({num_layers, 1, encoder_dim}, NULL, MNN::Tensor::CAFFE));
    auto cached_key = TensorPtr(MNN::Tensor::create<float>({num_layers, left_context_len, 1, attention_dim}, NULL, MNN::Tensor::CAFFE));
    auto cached_val = TensorPtr(MNN::Tensor::create<float>({num_layers, left_context_len, 1, attention_dim / 2}, NULL, MNN::Tensor::CAFFE));
    auto cached_val2 = TensorPtr(MNN::Tensor::create<float>({num_layers, left_context_len, 1, attention_dim / 2}, NULL, MNN::Tensor::CAFFE));
    auto cached_conv1 = TensorPtr(MNN::Tensor::create<float>({num_layers, 1, encoder_dim, cnn_module_kernel - 1}, NULL, MNN::Tensor::CAFFE));
    auto cached_conv2 = TensorPtr(MNN::Tensor::create<float>({num_layers, 1, encoder_dim, cnn_module_kernel - 1}, NULL, MNN::Tensor::CAFFE));

    Fill(cached_len, (int32_t)0);
    Fill(cached_avg, 0.0f);
    Fill(cached_key, 0.0f);
    Fill(cached_val, 0.0f);
    Fill(cached_val2, 0.0f);
    Fill(cached_conv1, 0.0f);
    Fill(cached_conv2, 0.0f);

    cached_len_vec.push_back(cached_len);
    cached_avg_vec.push_back(cached_avg);
    cached_key_vec.push_back(cached_key);
    cached_val_vec.push_back(cached_val);
    cached_val2_vec.push_back(cached_val2);
    cached_conv1_vec.push_back(cached_conv1);
    cached_conv2_vec.push_back(cached_conv2);
  }

  std::vector<TensorPtr> states;

  states.reserve(num_encoder_layers_.size() * 7);
  states.insert(states.end(), cached_len_vec.begin(), cached_len_vec.end());
  states.insert(states.end(), cached_avg_vec.begin(), cached_avg_vec.end());
  states.insert(states.end(), cached_key_vec.begin(), cached_key_vec.end());
  states.insert(states.end(), cached_val_vec.begin(), cached_val_vec.end());
  states.insert(states.end(), cached_val2_vec.begin(), cached_val2_vec.end());
  states.insert(states.end(), cached_conv1_vec.begin(), cached_conv1_vec.end());
  states.insert(states.end(), cached_conv2_vec.begin(), cached_conv2_vec.end());

  return states;
}

std::vector<TensorPtr> ZipformerModel::GetEncoderInitStates2() const {

  int32_t n = static_cast<int32_t>(encoder_dims_.size());
  int32_t m = std::accumulate(num_encoder_layers_.begin(), num_encoder_layers_.end(), 0);

  std::vector<TensorPtr> states;
  states.reserve(m * 6 + 2);

  for (size_t i = 0; i != num_encoder_layers_.size(); ++i) {
    int32_t num_layers = num_encoder_layers_[i];
    int32_t key_dim = query_head_dims_[i] * num_heads_[i];
    int32_t encoder_dim = encoder_dims_[i];
    int32_t nonlin_attn_head_dim = 3 * encoder_dim / 4;
    int32_t value_dim = value_head_dims_[i] * num_heads_[i];
    int32_t conv_left_pad = cnn_module_kernels_[i] / 2;
    int32_t left_context_len = left_context_len_[i];

    for (int32_t j = 0; j != num_layers; ++j) {
      // dims and dims_type info is from the output info of "./GetMNNInfo encoder.mnn"
      auto cached_key = TensorPtr(MNN::Tensor::create<float>({left_context_len, 1, key_dim}, NULL, MNN::Tensor::CAFFE));
      auto cached_nonlin_attn = TensorPtr(MNN::Tensor::create<float>({1, 1, left_context_len, nonlin_attn_head_dim}, NULL, MNN::Tensor::CAFFE));
      auto cached_val1 = TensorPtr(MNN::Tensor::create<float>({left_context_len, 1, value_dim}, NULL, MNN::Tensor::CAFFE));
      auto cached_val2 = TensorPtr(MNN::Tensor::create<float>({left_context_len, 1, value_dim}, NULL, MNN::Tensor::CAFFE));
      auto cached_conv1 = TensorPtr(MNN::Tensor::create<float>({1, encoder_dim, conv_left_pad}, NULL, MNN::Tensor::CAFFE));
      auto cached_conv2 = TensorPtr(MNN::Tensor::create<float>({1, encoder_dim, conv_left_pad}, NULL, MNN::Tensor::CAFFE));      

      Fill(cached_key, 0.0f);
      Fill(cached_nonlin_attn, 0.0f);
      Fill(cached_val1, 0.0f);
      Fill(cached_val2, 0.0f);
      Fill(cached_conv1, 0.0f);
      Fill(cached_conv2, 0.0f);

      states.push_back(std::move(cached_key));
      states.push_back(std::move(cached_nonlin_attn));
      states.push_back(std::move(cached_val1));
      states.push_back(std::move(cached_val2));
      states.push_back(std::move(cached_conv1));
      states.push_back(std::move(cached_conv2));
    }
  }

  int32_t embed_dim = (((feature_dim_ - 1) / 2) - 1) / 2;
  auto embed_states = TensorPtr(MNN::Tensor::create<float>({1, 128, 3, embed_dim}, NULL, MNN::Tensor::CAFFE));
  Fill(embed_states, 0.0f);
  states.push_back(std::move(embed_states));

  auto processed_lens = TensorPtr(MNN::Tensor::create<int32_t>({1}, NULL, MNN::Tensor::CAFFE));
  Fill(processed_lens, 0);
  states.push_back(std::move(processed_lens));

  return states;
}

}  // namespace SherpaDeploy
