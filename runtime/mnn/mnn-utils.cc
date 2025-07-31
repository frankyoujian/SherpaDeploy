// runtime/mnn/mnn-utils.cc
//
// Copyright (c)  2025  frankyj@foxmail.com  (authors: Jian You)

#include "mnn-utils.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "MNN/expr/Module.hpp"       // NOLINT

// #define PRINT_MODEL_METADATA

// from GetMNNInfo.cpp
static std::string _getDataType(const halide_type_t& type) {
    switch (type.code) {
        case halide_type_float:
            if (type.bits == 32) {
                return "float";
            }
            if (type.bits == 16) {
                return "half";
            }
            break;
        case halide_type_uint:
            if (type.bits == 32) {
                return "uint32";
            }
            if (type.bits == 16) {
                return "uint16";
            }
            if (type.bits == 8) {
                return "uint8";
            }
            break;
        case halide_type_int:
            if (type.bits == 32) {
                return "int32";
            }
            if (type.bits == 16) {
                return "int16";
            }
            if (type.bits == 8) {
                return "int8";
            }
            break;
        default:
            break;
    }
    return "Unknown";
}
static std::string _getFormatString(MNN::Express::Dimensionformat format) {
    switch (format) {
        case MNN::Express::NCHW:
            return "NCHW";
        case MNN::Express::NHWC:
            return "NHWC";
        case MNN::Express::NC4HW4:
            return "NC4HW4";
        default:
            break;
    }
    return "Unknown";
}

namespace SherpaDeploy {

void GetInputNames(const std::shared_ptr<MNN::Express::Module>& module, std::vector<std::string>& input_names) {
  auto info = module->getInfo();
  // MNN_ASSERT(info->inputNames.size() == info->inputs.size());
  // MNN_PRINT("Model default dimensionFormat is %s\n", _getFormatString(info->defaultFormat).c_str());
  size_t names_count = info->inputNames.size();
  input_names.resize(names_count);

#ifdef PRINT_MODEL_METADATA
  fprintf(stderr, "\n================= Model Inputs: =================\n");
#endif
  for (size_t i=0; i<names_count; ++i) {
      auto& varInfo = info->inputs[i];

      input_names[i] = std::move(info->inputNames[i]);

#ifdef PRINT_MODEL_METADATA
      fprintf(stderr, "[%s]: dimensionFormat: %s, ", input_names[i].c_str(), _getFormatString(varInfo.order).c_str());
      fprintf(stderr, "size: [ ");
      if (varInfo.dim.size() > 0) {
          for (int j=0; j<(int)varInfo.dim.size() - 1; ++j) {
              fprintf(stderr, "%d,", varInfo.dim[j]);
          }
          fprintf(stderr, "%d ", varInfo.dim[(int)varInfo.dim.size() - 1]);
      }
      fprintf(stderr, "], ");
      fprintf(stderr, "type is %s\n", _getDataType(varInfo.type).c_str());
#endif
  }
}

void GetOutputNames(const std::shared_ptr<MNN::Express::Module>& module, std::vector<std::string>& output_names) {
  auto info = module->getInfo();
  // MNN_ASSERT(info->outputNames.size() == info->outputs.size());
  size_t names_count = info->outputNames.size();
  output_names.resize(names_count);

#ifdef PRINT_MODEL_METADATA
  fprintf(stderr, "\n================= Model Outputs: =================\n");
#endif
  for (size_t i=0; i<names_count; ++i) {

    output_names[i] = std::move(info->outputNames[i]);

#ifdef PRINT_MODEL_METADATA
    fprintf(stderr, "[%s]\n", output_names[i].c_str());
#endif
  }
}

TensorPtr GetEncoderOutFrame(TensorPtr encoder_out, int32_t t) {
  std::vector<int> encoder_out_shape = encoder_out->shape();

  auto shape = encoder_out->shape();

  auto batch_size = encoder_out_shape[0];
  auto num_frames = encoder_out_shape[1];
  // TODO: add assert()
  // assert(t < num_frames);

  auto encoder_out_dim = encoder_out_shape[2];

  auto offset = num_frames * encoder_out_dim;

  TensorPtr ans = TensorPtr(MNN::Tensor::create<float>({batch_size, encoder_out_dim}, NULL, MNN::Tensor::CAFFE));

  float* p_dst = ans->host<float>();
  const float* src = encoder_out->host<float>();

  for (int32_t i = 0; i != batch_size; ++i) {
    std::copy(src + t * encoder_out_dim, src + (t + 1) * encoder_out_dim, p_dst);
    src += offset;
    p_dst += encoder_out_dim;
  }
  return ans;
}

}  // namespace SherpaDeploy
