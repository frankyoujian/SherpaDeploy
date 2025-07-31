/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 * Copyright (c)  2022                     (Pingfeng Luo)
 * Copyright (c)  2025  frankyj@foxmail.com  (authors: Jian You)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "greedy-search-decoder.h"

#include <vector>
#include <algorithm>
#include "MNN/Tensor.hpp"   // NOLINT
#include "mnn-utils.h"

namespace SherpaDeploy {

TensorPtr GreedySearchDecoder::BuildDecoderInput(
    const DecoderResult &result) const {

  int32_t context_size = model_->ContextSize();
  TensorPtr decoder_input = TensorPtr(MNN::Tensor::create<int32_t>({1, context_size}, NULL, MNN::Tensor::CAFFE));

  int32_t* p_dst = decoder_input->host<int32_t>();
  for (int32_t i = 0; i != context_size; ++i) {
    *p_dst = *(result.tokens.end() - context_size + i);
    p_dst++;
  }
  return decoder_input;
}

DecoderResult GreedySearchDecoder::GetEmptyResult() const {
  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0
  DecoderResult r;
  r.tokens.resize(context_size, blank_id);

  return r;
}

void GreedySearchDecoder::StripLeadingBlanks(DecoderResult *r) const {
  int32_t context_size = model_->ContextSize();

  auto start = r->tokens.begin() + context_size;
  auto end = r->tokens.end();

  r->tokens = std::vector<int32_t>(start, end);
}

void GreedySearchDecoder::Decode(TensorPtr encoder_out, DecoderResult *result) {

  auto encoder_out_shape = encoder_out->shape();
  int32_t batch_size = encoder_out_shape[0];
  int32_t num_frames = encoder_out_shape[1];

  // TODO(fangjun): Cache the result of decoder_out
  TensorPtr decoder_out = result->decoder_out;
  if (nullptr == decoder_out) {
    TensorPtr decoder_input = BuildDecoderInput(*result);
    decoder_out = model_->RunDecoder(decoder_input);
  }

  int32_t frame_offset = result->frame_offset;
  for (int32_t t = 0; t != num_frames; ++t) {
    TensorPtr encoder_out_t = GetEncoderOutFrame(encoder_out, t);
    TensorPtr joiner_out = model_->RunJoiner(encoder_out_t, decoder_out);
    auto joiner_out_shape = joiner_out->shape();
    int32_t vocab_size = joiner_out_shape[1];

    const float* joiner_out_ptr = joiner_out->host<float>();

    auto new_token = static_cast<int32_t>(std::distance(
        joiner_out_ptr,
        std::max_element(joiner_out_ptr, joiner_out_ptr + vocab_size)));

    // the blank ID is fixed to 0
    if (new_token != 0 && new_token != 2) {
      result->tokens.push_back(new_token);
      TensorPtr decoder_input = BuildDecoderInput(*result);
      decoder_out = model_->RunDecoder(decoder_input);

      result->num_trailing_blanks = 0;
      result->timestamps.push_back(t + frame_offset);
    } else {
      ++result->num_trailing_blanks;
    }

  }

  result->frame_offset += num_frames;
  result->decoder_out = decoder_out;
}

}  // namespace SherpaDeploy
