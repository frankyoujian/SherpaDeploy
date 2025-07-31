// runtime/mnn/zipformer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2025  frankyj@foxmail.com  (authors: Jian You)

#include "zipformer-model.h"
#include "openvino/openvino.hpp"

#include <regex>  // NOLINT
#include <string>
#include <utility>
#include <vector>
#include <sstream>

#include <iostream>
#include <filesystem>

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

template <typename T = float>
void Fill(ov::Tensor tensor, T value) {
  auto data = tensor.data<T>();
  auto size = tensor.get_size();
  std::fill(data, data + size, value);
}

namespace SherpaDeploy {

ZipformerModel::ZipformerModel(const ModelConfig &config) {

  auto ov_version = ov::get_openvino_version();
  std::cout << "OPENVINO|VERSION|" << ov_version << std::endl;

  core_ = std::make_shared<ov::Core>();

  std::vector<std::string> availableDevices = core_->get_available_devices();
  fprintf(stderr, "Available devices: ");
  for (auto& device : availableDevices) {
    fprintf(stderr, "%s  ", device.c_str());
  }

  std::vector<std::string> caps = core_->get_property(config.device, ov::device::capabilities);
  // Find 'EXPORT_IMPORT' capability in supported capabilities
  if (bool cachingSupported = std::find(caps.begin(), caps.end(), ov::device::capability::EXPORT_IMPORT) != caps.end(); cachingSupported) {
    core_->set_property(ov::cache_dir(std::filesystem::current_path().string()));
    fprintf(stderr, "\n\nSaved compiled model cache in: %s\n", std::filesystem::current_path().string().c_str());
  }

  device_ = config.device;

  // There is a single infer request and we want it infer as qucikly as possible, therefore use LATENCY
  core_->set_property(device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)); // LATENCY | THROUGHPUT
  if (device_ == "CPU") {
    // Refer to below link for properties in latency hint:
    // https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.html#latency-hint
    core_->set_property(device_, ov::num_streams(1)); // only one infer request at any time, so set to 1
    core_->set_property(device_, ov::inference_num_threads(config.num_threads));
    core_->set_property(device_, ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ANY_CORE));
    core_->set_property(device_, ov::hint::enable_hyper_threading(false));
    core_->set_property(device_, ov::hint::enable_cpu_pinning(false));
  }

  InitEncoder(config.encoder_xml);
  InitDecoder(config.decoder_xml);
  InitJoiner(config.joiner_xml);

}

std::pair<ov::Tensor, std::vector<ov::Tensor>> ZipformerModel::RunEncoder(
    ov::Tensor features, const std::vector<ov::Tensor>& states) {
  std::vector<ov::Tensor> _states;

  if (states.empty()) {
    _states = GetEncoderInitStates();
  } else {
    _states = states;
  }

  encoder_infer_->set_tensor(encoder_input_names_[0], features);
  // encoder_infer_->set_input_tensor(0, features);

  for (size_t i = 1; i < encoder_input_names_.size(); ++i) {
    encoder_infer_->set_tensor(encoder_input_names_[i], _states[i-1]);
    // encoder_infer_->set_input_tensor(i, p[i-1]);
  } 

  encoder_infer_->infer();

  // ov::Tensor encoder_out = encoder_infer_->get_output_tensor(0);
  ov::Tensor encoder_out = encoder_infer_->get_tensor(encoder_output_names_[0]);

  std::vector<ov::Tensor> next_states(_states.size());
  for (size_t i=1; i<encoder_output_names_.size(); ++i) {
    next_states[i-1] = encoder_infer_->get_tensor(encoder_output_names_[i]);
  }

  return {encoder_out, next_states};
}

ov::Tensor ZipformerModel::RunDecoder(ov::Tensor decoder_input) {

  decoder_infer_->set_tensor(decoder_input_names_[0], decoder_input);

  decoder_infer_->infer();

  ov::Tensor decoder_out = decoder_infer_->get_output_tensor();

  return decoder_out;
}

ov::Tensor ZipformerModel::RunJoiner(ov::Tensor encoder_out, ov::Tensor decoder_out) {

  joiner_infer_->set_tensor(joiner_input_names_[0], encoder_out);
  joiner_infer_->set_tensor(joiner_input_names_[1], decoder_out);

  joiner_infer_->infer();

  ov::Tensor joiner_out = joiner_infer_->get_output_tensor();

  return joiner_out;
}

void ZipformerModel::InitEncoder(const std::string& ir_path) {
  std::shared_ptr<ov::Model> encoder_model = core_->read_model(ir_path);

  // contains dynamic shape, needs to reshape to fixed batch size
  if (encoder_model->is_dynamic()) {
    auto inputs = encoder_model->inputs();

    std::map<size_t, ov::PartialShape> idx_to_shape;
    for (size_t i=0; i<inputs.size(); ++i) {

      auto pshape = inputs[i].get_partial_shape();
      for (size_t j=0; j<pshape.size(); ++j) {
        if (pshape[j].is_dynamic()) {
          pshape[j] = 1;
        }
      }
      idx_to_shape[i] = pshape;
    }

    encoder_model->reshape(idx_to_shape);
  }

  encoder_compile_model_ = std::make_shared<ov::CompiledModel>(
    std::move(core_->compile_model(encoder_model, device_)));

  encoder_infer_ = std::make_shared<ov::InferRequest>(
    std::move(encoder_compile_model_->create_infer_request()));

  auto inputs = encoder_model->inputs();
#ifdef PRINT_MODEL_METADATA
  fprintf(stderr, "\n===== encoder inputs =====\n");
#endif
  for (size_t i=0; i<inputs.size(); ++i) {
    // auto name = inputs[i].get_any_name();
    auto name = *(inputs[i].get_names().begin()); // input.get_any_name() can cause heap error in debug mode
    encoder_input_names_.push_back(name);

#ifdef PRINT_MODEL_METADATA
    fprintf(stderr, "name[%d]:%s, shape:%s, type:%s\n", 
                        i, 
                        name, 
                        inputs[i].get_partial_shape().to_string().c_str(), 
                        inputs[i].get_element_type().get_type_name().c_str());
#endif
  }

  auto outputs = encoder_model->outputs();

  for (size_t i=0; i<outputs.size(); ++i) {
    // auto name = outputs[i].get_any_name();
    auto name = *(outputs[i].get_names().begin()); // input.get_any_name() can cause heap error in debug mode
    encoder_output_names_.push_back(name);
  }

  if (encoder_model->has_rt_info("framework")) {
    
    auto metadata = encoder_model->get_rt_info<ov::AnyMap>("framework");

    model_type_ = metadata["model_type"].as<std::string>();
    fprintf(stderr, "/*********** %s ***********/\n", model_type_.c_str());

    if (model_type_ == "zipformer") {
      attention_dims_       = std::move(convertStringToVector(metadata["attention_dims"].as<std::string>()));
    }
    else if (model_type_ == "zipformer2") {
      num_heads_            = std::move(convertStringToVector(metadata["num_heads"].as<std::string>()));
      query_head_dims_      = std::move(convertStringToVector(metadata["query_head_dims"].as<std::string>()));
      value_head_dims_      = std::move(convertStringToVector(metadata["value_head_dims"].as<std::string>()));      
    }

    cnn_module_kernels_   = std::move(convertStringToVector(metadata["cnn_module_kernels"].as<std::string>()));
    encoder_dims_         = std::move(convertStringToVector(metadata["encoder_dims"].as<std::string>()));
    left_context_len_     = std::move(convertStringToVector(metadata["left_context_len"].as<std::string>()));
    num_encoder_layers_   = std::move(convertStringToVector(metadata["num_encoder_layers"].as<std::string>()));
    decode_chunk_length_  = metadata["decode_chunk_len"].as<int>();
    T_                    = metadata["T"].as<int>();
  
    fprintf(stderr, "\n------ encoder metadata ------\n");
    if (model_type_ == "zipformer") {
      fprintf(stderr, "attention_dims :      %s\n",   metadata["attention_dims"].as<std::string>().c_str());
    }
    else if (model_type_ == "zipformer2") {
      fprintf(stderr, "num_heads :      %s\n",        metadata["num_heads"].as<std::string>().c_str());
      fprintf(stderr, "query_head_dims :      %s\n",        metadata["query_head_dims"].as<std::string>().c_str());
      fprintf(stderr, "value_head_dims :      %s\n",        metadata["value_head_dims"].as<std::string>().c_str());    
    }

    fprintf(stderr, "num_encoder_layers :  %s\n",   metadata["num_encoder_layers"].as<std::string>().c_str());
    fprintf(stderr, "encoder_dims :        %s\n",   metadata["encoder_dims"].as<std::string>().c_str());
    fprintf(stderr, "cnn_module_kernels :  %s\n",   metadata["cnn_module_kernels"].as<std::string>().c_str());
    fprintf(stderr, "left_context_len :    %s\n",   metadata["left_context_len"].as<std::string>().c_str());
    fprintf(stderr, "decode_chunk_length : %d\n",   metadata["decode_chunk_len"].as<int>());
    fprintf(stderr, "T :                   %d\n",   metadata["T"].as<int>());
  }

}

void ZipformerModel::InitDecoder(const std::string& ir_path) {
  std::shared_ptr<ov::Model> decoder_model = core_->read_model(ir_path);

  // contains dynamic shape, needs to reshape to fixed batch size
  if (decoder_model->is_dynamic()) {
    auto inputs = decoder_model->inputs();

    std::map<size_t, ov::PartialShape> idx_to_shape;
    for (size_t i=0; i<inputs.size(); ++i) {

      auto pshape = inputs[i].get_partial_shape();
      for (size_t j=0; j<pshape.size(); ++j) {
        if (pshape[j].is_dynamic()) {
          pshape[j] = 1;
        }
      }
      idx_to_shape[i] = pshape;
    }

    decoder_model->reshape(idx_to_shape);
  }

  decoder_compile_model_ = std::make_shared<ov::CompiledModel>(
    std::move(core_->compile_model(decoder_model, device_)));

  decoder_infer_ = std::make_shared<ov::InferRequest>(
    std::move(decoder_compile_model_->create_infer_request()));

  auto inputs = decoder_compile_model_->inputs();
#ifdef PRINT_MODEL_METADATA
  fprintf(stderr, "\n===== decoder inputs =====\n");
#endif
  for (size_t i=0; i<inputs.size(); ++i) {
    // auto name = inputs[i].get_any_name();
    auto name = *(inputs[i].get_names().begin());
    decoder_input_names_.push_back(name);

#ifdef PRINT_MODEL_METADATA
    fprintf(stderr, "name[%d]:%s, shape:%s, type:%s\n", 
                        i, 
                        name, 
                        inputs[i].get_partial_shape().to_string().c_str(), 
                        inputs[i].get_element_type().get_type_name().c_str());
#endif
  }

  auto outputs = decoder_compile_model_->outputs();

  for (size_t i=0; i<outputs.size(); ++i) {
    // auto name = outputs[i].get_any_name();
    auto name = *(outputs[i].get_names().begin()); // input.get_any_name() can cause heap error in debug mode
    decoder_output_names_.push_back(name);
  }

  if (decoder_model->has_rt_info("framework")) {
    auto metadata = decoder_model->get_rt_info<ov::AnyMap>("framework");

    context_size_ = metadata["context_size"].as<int>();
    vocab_size_ = metadata["vocab_size"].as<int>();

    fprintf(stderr, "\n----- decoder metadata -----\n");
    fprintf(stderr, "context_size : %d\n", metadata["context_size"].as<int>());
    fprintf(stderr, "vocab_size : %d\n",   metadata["vocab_size"].as<int>());  
  }
}

void ZipformerModel::InitJoiner(const std::string& ir_path) {
  std::shared_ptr<ov::Model> joiner_model = core_->read_model(ir_path);

  // contains dynamic shape, needs to reshape to fixed batch size
  if (joiner_model->is_dynamic()) {
    auto inputs = joiner_model->inputs();

    std::map<size_t, ov::PartialShape> idx_to_shape;
    for (size_t i=0; i<inputs.size(); ++i) {

      auto pshape = inputs[i].get_partial_shape();
      for (size_t j=0; j<pshape.size(); ++j) {
        if (pshape[j].is_dynamic()) {
          pshape[j] = 1;
        }
      }
      idx_to_shape[i] = pshape;
    }

    joiner_model->reshape(idx_to_shape);
  }

  joiner_compile_model_ = std::make_shared<ov::CompiledModel>(
    std::move(core_->compile_model(joiner_model, device_)));

  joiner_infer_ = std::make_shared<ov::InferRequest>(
    std::move(joiner_compile_model_->create_infer_request()));

  auto inputs = joiner_compile_model_->inputs();
#ifdef PRINT_MODEL_METADATA
  fprintf(stderr, "\n===== joiner inputs =====\n");
#endif
  for (size_t i=0; i<inputs.size(); ++i) {
    // auto name = inputs[i].get_any_name();
    auto name = *(inputs[i].get_names().begin());
    joiner_input_names_.push_back(name);

#ifdef PRINT_MODEL_METADATA
    fprintf(stderr, "name[%d]:%s, shape:%s, type:%s\n", 
                        i, 
                        name, 
                        inputs[i].get_partial_shape().to_string().c_str(), 
                        inputs[i].get_element_type().get_type_name().c_str());
#endif
  }

  auto outputs = joiner_compile_model_->outputs();

  for (size_t i=0; i<outputs.size(); ++i) {
    // auto name = outputs[i].get_any_name();
    auto name = *(outputs[i].get_names().begin()); // input.get_any_name() can cause heap error in debug mode
    joiner_output_names_.push_back(name);
  }
}

std::vector<ov::Tensor> ZipformerModel::GetEncoderInitStates() const {
  if (model_type_ == "zipformer") {
    return GetEncoderInitStates1();
  }
  else if (model_type_ == "zipformer2") {
    return GetEncoderInitStates2();
  }
}

// see
// https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py#L673
std::vector<ov::Tensor> ZipformerModel::GetEncoderInitStates1() const {
  // each layer has 7 states:
  // cached_len, (num_layers,)
  // cached_avg, (num_layers, encoder_dim)
  // cached_key, (num_layers, left_context_length, attention_dim)
  // cached_val, (num_layers, left_context_length, attention_dim / 2)
  // cached_val2, (num_layers, left_context_length, attention_dim / 2)
  // cached_conv1, (num_layers, encoder_dim, cnn_module_kernel_ - 1)
  // cached_conv2, (num_layers, encoder_dim, cnn_module_kernel_ - 1)

  std::vector<ov::Tensor> cached_len_vec;
  std::vector<ov::Tensor> cached_avg_vec;
  std::vector<ov::Tensor> cached_key_vec;
  std::vector<ov::Tensor> cached_val_vec;
  std::vector<ov::Tensor> cached_val2_vec;
  std::vector<ov::Tensor> cached_conv1_vec;
  std::vector<ov::Tensor> cached_conv2_vec;

  cached_len_vec.reserve(num_encoder_layers_.size());
  cached_avg_vec.reserve(num_encoder_layers_.size());
  cached_key_vec.reserve(num_encoder_layers_.size());
  cached_val_vec.reserve(num_encoder_layers_.size());
  cached_val2_vec.reserve(num_encoder_layers_.size());
  cached_conv1_vec.reserve(num_encoder_layers_.size());
  cached_conv2_vec.reserve(num_encoder_layers_.size());

  for (size_t i = 0; i != num_encoder_layers_.size(); ++i) {
    size_t num_layers = num_encoder_layers_[i];
    size_t attention_dim = attention_dims_[i];
    size_t left_context_len = left_context_len_[i];
    size_t encoder_dim = encoder_dims_[i];
    size_t cnn_module_kernel = cnn_module_kernels_[i];

    ov::Tensor cached_len = ov::Tensor(ov::element::i64, {num_layers, 1});
    ov::Tensor cached_avg = ov::Tensor(ov::element::f32, {num_layers, 1, encoder_dim});
    ov::Tensor cached_key = ov::Tensor(ov::element::f32, {num_layers, left_context_len, 1, attention_dim});
    ov::Tensor cached_val = ov::Tensor(ov::element::f32, {num_layers, left_context_len, 1, attention_dim / 2});
    ov::Tensor cached_val2 = ov::Tensor(ov::element::f32, {num_layers, left_context_len, 1, attention_dim / 2});
    ov::Tensor cached_conv1 = ov::Tensor(ov::element::f32, {num_layers, 1, encoder_dim, cnn_module_kernel - 1});
    ov::Tensor cached_conv2 = ov::Tensor(ov::element::f32, {num_layers, 1, encoder_dim, cnn_module_kernel - 1});

    Fill(cached_len, (int64_t)0);
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

  std::vector<ov::Tensor> states;

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

std::vector<ov::Tensor> ZipformerModel::GetEncoderInitStates2() const {

  int32_t n = static_cast<int32_t>(encoder_dims_.size());
  int32_t m = std::accumulate(num_encoder_layers_.begin(), num_encoder_layers_.end(), 0);

  std::vector<ov::Tensor> states;
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
      ov::Shape cached_key_shape={size_t(left_context_len),1,size_t(key_dim)};
      ov::Shape cached_nonlin_attn_shape={1,1,size_t(left_context_len),size_t(nonlin_attn_head_dim)};
      ov::Shape cached_val1_shape={size_t(left_context_len),1,size_t(value_dim)};
      ov::Shape cached_val2_shape={size_t(left_context_len),1,size_t(value_dim)};
      ov::Shape cached_conv1_shape={1,size_t(encoder_dim),size_t(conv_left_pad)};
      ov::Shape cached_conv2_shape={1,size_t(encoder_dim),size_t(conv_left_pad)};

      ov::Tensor cached_key = ov::Tensor(ov::element::f32, cached_key_shape);
      ov::Tensor cached_nonlin_attn = ov::Tensor(ov::element::f32, cached_nonlin_attn_shape);
      ov::Tensor cached_val1 = ov::Tensor(ov::element::f32, cached_val1_shape);
      ov::Tensor cached_val2 = ov::Tensor(ov::element::f32, cached_val2_shape);
      ov::Tensor cached_conv1 = ov::Tensor(ov::element::f32, cached_conv1_shape);
      ov::Tensor cached_conv2 = ov::Tensor(ov::element::f32, cached_conv2_shape);

      float *cached_key_pointer = cached_key.data<float>();
      float *cached_nonlin_attn_pointer = cached_nonlin_attn.data<float>();
      float *cached_val1_pointer = cached_val1.data<float>();
      float *cached_val2_pointer = cached_val2.data<float>();
      float *cached_conv1_pointer = cached_conv1.data<float>();
      float *cached_conv2_pointer = cached_conv2.data<float>();

      memset(cached_key_pointer, 0,
              sizeof(float)*(key_dim * left_context_len));
      memset(cached_nonlin_attn_pointer, 0,
              sizeof(float)*(nonlin_attn_head_dim * left_context_len));
      memset(cached_val1_pointer, 0,
              sizeof(float)*(value_dim * left_context_len));
      memset(cached_val2_pointer, 0,
              sizeof(float)*(value_dim * left_context_len));
      memset(cached_conv1_pointer, 0,
              sizeof(float)*(conv_left_pad * encoder_dim));
      memset(cached_conv2_pointer, 0,
              sizeof(float)*(conv_left_pad * encoder_dim));

      states.push_back(std::move(cached_key));
      states.push_back(std::move(cached_nonlin_attn));
      states.push_back(std::move(cached_val1));
      states.push_back(std::move(cached_val2));
      states.push_back(std::move(cached_conv1));
      states.push_back(std::move(cached_conv2));
    }
  }

  int32_t embed_dim = (((feature_dim_ - 1) / 2) - 1) / 2;
  ov::Tensor embed_states = ov::Tensor(ov::element::f32, {1, 128, 3, size_t(embed_dim)});
  float *embed_states_pointer = embed_states.data<float>();
  memset(embed_states_pointer, 0, sizeof(float)*(embed_dim * 128 * 3));
  states.push_back(std::move(embed_states));

  ov::Tensor processed_lens = ov::Tensor(ov::element::i64, {1});
  int64_t *processed_lens_pointer = processed_lens.data<int64_t>();
  memset(processed_lens_pointer, 0, sizeof(int64_t)*1);
  states.push_back(std::move(processed_lens));

  return states;
}

}  // namespace SherpaDeploy
