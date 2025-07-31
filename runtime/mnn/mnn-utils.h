// runtime/mnn/mnn-utils.h
//
// Copyright (c)  2025  frankyj@foxmail.com  (authors: Jian You)

#ifndef SHERPA_DEPLOY_MNN_UTILS_H_
#define SHERPA_DEPLOY_MNN_UTILS_H_

#include <cassert>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "MNN/Tensor.hpp"   // NOLINT

namespace MNN {
  namespace Express {
    class Module;
  }
}

namespace SherpaDeploy {

using TensorPtr = std::shared_ptr<MNN::Tensor>;

// /**
//  * Get the input names of a model.
//  *
//  * @param module: MNN module
//  * @param input_names. On return, it contains the input names of the model.
//  */
void GetInputNames(const std::shared_ptr<MNN::Express::Module>& module, std::vector<std::string>& input_names);

// /**
//  * Get the output names of a model.
//  *
//  * @param module: MNN module
//  * @param output_names. On return, it contains the output names of the model.
//  */
void GetOutputNames(const std::shared_ptr<MNN::Express::Module>& module, std::vector<std::string>& output_names);

/**
 * Get the output frame of Encoder
 *
 * @param encoder_out encoder out tensor
 * @param t frame_index
 *
 */
TensorPtr GetEncoderOutFrame(TensorPtr encoder_out, int32_t t);


template <typename T = float>
void Fill(TensorPtr tensor, T value) {
  auto data = tensor->host<T>();
  auto size = tensor->elementSize();
  std::fill(data, data + size, value);
}


}  // namespace SherpaDeploy

#endif  // SHERPA_DEPLOY_MNN_UTILS_H_
