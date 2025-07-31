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
#include <iostream>

namespace SherpaDeploy {

ov::Tensor GreedySearchDecoder::GetEncoderOutFrame(ov::Tensor encoder_out, int32_t t) const {
  auto encoder_out_shape = encoder_out.get_shape();

  auto batch_size = encoder_out_shape[0];
  auto num_frames = encoder_out_shape[1];
  // TODO: add assert()
  // assert(t < num_frames);

  auto encoder_out_dim = encoder_out_shape[2];

  auto offset = num_frames * encoder_out_dim;

  ov::Tensor ans = ov::Tensor(ov::element::f32, {batch_size, encoder_out_dim});

  float* p_dst = ans.data<float>();
  const float* src = encoder_out.data<float>();

  for (int32_t i = 0; i != batch_size; ++i) {
    std::copy(src + t * encoder_out_dim, src + (t + 1) * encoder_out_dim, p_dst);
    src += offset;
    p_dst += encoder_out_dim;
  }
  return ans;
}

ov::Tensor GreedySearchDecoder::BuildDecoderInput(
    const DecoderResult &result) const {

  size_t context_size = model_->ContextSize();
  ov::Tensor decoder_input = ov::Tensor(ov::element::i64, {1, context_size});

  int64_t* p_dst = decoder_input.data<int64_t>();
  for (int32_t i = 0; i != context_size; ++i) {
    *p_dst = static_cast<int64_t>(*(result.tokens.end() - context_size + i));
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

void GreedySearchDecoder::Decode(ov::Tensor encoder_out, DecoderResult *result) {

  int32_t batch_size = encoder_out.get_shape()[0];
  int32_t num_frames = encoder_out.get_shape()[1];

  // TODO(fangjun): Cache the result of decoder_out
  ov::Tensor decoder_out = result->decoder_out;
  if (!decoder_out) {
    ov::Tensor decoder_input = BuildDecoderInput(*result);
    decoder_out = model_->RunDecoder(decoder_input);
  }

  int32_t frame_offset = result->frame_offset;
  for (int32_t t = 0; t != num_frames; ++t) {
    ov::Tensor encoder_out_t = GetEncoderOutFrame(encoder_out, t);
    ov::Tensor joiner_out = model_->RunJoiner(encoder_out_t, decoder_out);

    int32_t vocab_size = joiner_out.get_shape()[1];

    const float* joiner_out_ptr = joiner_out.data<float>();

    auto new_token = static_cast<int32_t>(std::distance(
        joiner_out_ptr,
        std::max_element(joiner_out_ptr, joiner_out_ptr + vocab_size)));

    // the blank ID is fixed to 0
    if (new_token != 0 && new_token != 2) {
      result->tokens.push_back(new_token);
      ov::Tensor decoder_input = BuildDecoderInput(*result);
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
