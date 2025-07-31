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

#include "modified-beam-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "runtime/core/math.h"

namespace SherpaDeploy {

DecoderResult ModifiedBeamSearchDecoder::GetEmptyResult() const {
  DecoderResult r;

  int32_t context_size = model_->ContextSize();
  int32_t blank_id = 0;  // always 0

  std::vector<int32_t> blanks(context_size, blank_id);
  SherpaDeploy::Hypotheses blank_hyp({{blanks, 0}});

  r.hyps = std::move(blank_hyp);
  r.tokens = std::move(blanks);
  return r;
}

void ModifiedBeamSearchDecoder::StripLeadingBlanks(DecoderResult *r) const {
  int32_t context_size = model_->ContextSize();
  auto hyp = r->hyps.GetMostProbable(true);

  auto start = hyp.ys.begin() + context_size;
  auto end = hyp.ys.end();

  r->tokens = std::vector<int32_t>(start, end);
  r->timestamps = std::move(hyp.timestamps);
  r->num_trailing_blanks = hyp.num_trailing_blanks;
}

// Compute log_softmax in-place.
//
// The log_softmax of each row is computed.
//
// @param in_out A 2-D tensor
static void LogSoftmax(ov::Tensor in_out) {
  int32_t h = in_out.get_shape()[0];
  int32_t w = in_out.get_shape()[1];
  float* p_src = in_out.data<float>();
  for (int32_t y = 0; y != h; ++y) {
    float* p = p_src + y * w;
    SherpaDeploy::LogSoftmax(p, w);
  }
}

// The decoder model contains an embedding layer, which only supports
// 1-D output.
// This is a wrapper to support 2-D decoder output.
//
// @param model_ The NN model.
// @param decoder_input A 2-D tensor of shape (num_active_paths, context_size)
// @return Return a 2-D tensor of shape (num_active_paths, decoder_dim)
//
// TODO(fangjun): Change Embed in ncnn to output 2-d tensors
static ov::Tensor RunDecoder2D(Model *model_, ov::Tensor decoder_input) {
  ov::Tensor decoder_out;
  size_t h = decoder_input.get_shape()[0];
  size_t w = decoder_input.get_shape()[1];
  int64_t* p_src = decoder_input.data<int64_t>();

  for (int32_t y = 0; y != h; ++y) {
    ov::Tensor decoder_input_t = ov::Tensor(ov::element::i64, {1, w});
    int64_t* p_dst = decoder_input_t.data<int64_t>();
    std::copy(p_src, p_src + w, p_dst);
    p_src += w;

    ov::Tensor tmp = model_->RunDecoder(decoder_input_t);
    size_t decoder_out_dim = tmp.get_shape()[1];

    if (y == 0) {
      decoder_out = ov::Tensor(ov::element::f32, {h, decoder_out_dim});
    }

    const float *ptr = tmp.data<float>();
    float* out_ptr = decoder_out.data<float>() + y * decoder_out_dim;
    std::copy(ptr, ptr + decoder_out_dim, out_ptr);

  }

  return decoder_out;
}

ov::Tensor ModifiedBeamSearchDecoder::BuildDecoderInput(
    const std::vector<SherpaDeploy::Hypothesis> &hyps) const {
  size_t num_hyps = static_cast<int32_t>(hyps.size());
  size_t context_size = model_->ContextSize();

  ov::Tensor decoder_input = ov::Tensor(ov::element::i64, {num_hyps, context_size});
  int64_t* p = decoder_input.data<int64_t>();

  for (const auto &hyp : hyps) {
    const auto &ys = hyp.ys;
    // transform and copy
    std::transform(ys.end() - context_size, ys.end(), p,                  
                     [](int32_t val) {  
                         return static_cast<int64_t>(val);
                     });
    p += context_size;
  }

  return decoder_input;
}

void ModifiedBeamSearchDecoder::Decode(ov::Tensor encoder_out,
                                       DecoderResult *result) {
  Decode(encoder_out, nullptr, result);
}

void ModifiedBeamSearchDecoder::Decode(ov::Tensor encoder_out, Stream *s,
                                       DecoderResult *result) {
  size_t batch_size = encoder_out.get_shape()[0];
  size_t num_frames = encoder_out.get_shape()[1];
  size_t encoder_out_dim = encoder_out.get_shape()[2];

  int32_t context_size = model_->ContextSize();
  SherpaDeploy::Hypotheses cur = std::move(result->hyps);

  const float* p_src = encoder_out.data<float>();

  /* encoder_out.w == encoder_out_dim, encoder_out.h == num_frames. */
  for (size_t t = 0; t != num_frames; ++t) {
    std::vector<SherpaDeploy::Hypothesis> prev = cur.GetTopK(num_active_paths_, true);
    cur.Clear();

    ov::Tensor decoder_input = BuildDecoderInput(prev);
    ov::Tensor decoder_out;
    if (t == 0 && prev.size() == 1 && prev[0].ys.size() == context_size &&
        result->decoder_out) {
      // When an endpoint is detected, we keep the decoder_out
      decoder_out = result->decoder_out;
    } else {
      decoder_out = RunDecoder2D(model_, decoder_input);
    }

    // decoder_out.w == decoder_dim
    // decoder_out.h == num_active_paths
    ov::Tensor encoder_out_t = ov::Tensor(ov::element::f32, {1, encoder_out_dim});
    float* p_dst = encoder_out_t.data<float>();
    std::copy(p_src, p_src + encoder_out_dim, p_dst);
    p_src += encoder_out_dim;

    ov::Tensor joiner_out = model_->RunJoiner(encoder_out_t, decoder_out);
    // joiner_out.w == vocab_size
    // joiner_out.h == num_active_paths
    int32_t num_active_paths = joiner_out.get_shape()[0];
    int32_t vocab_size = joiner_out.get_shape()[1];

    LogSoftmax(joiner_out);

    float* p_joiner_out = joiner_out.data<float>();

    for (int32_t i = 0; i != num_active_paths; ++i) {
      float prev_log_prob = prev[i].log_prob;
      for (int32_t k = 0; k != vocab_size; ++k, ++p_joiner_out) {
        *p_joiner_out += prev_log_prob;
      }
    }

    p_joiner_out = joiner_out.data<float>();
    auto topk = SherpaDeploy::TopkIndex(p_joiner_out,
                          num_active_paths * vocab_size, num_active_paths_);

    int32_t frame_offset = result->frame_offset;
    for (auto i : topk) {
      int32_t hyp_index = i / vocab_size;
      int32_t new_token = i % vocab_size;

      const float* p = p_joiner_out + hyp_index * vocab_size;

      SherpaDeploy::Hypothesis new_hyp = prev[hyp_index];
      // const float prev_lm_log_prob = new_hyp.lm_log_prob;
      float context_score = 0;
      auto context_state = new_hyp.context_state;
      // blank id is fixed to 0
      if (new_token != 0 && new_token != 2) {
        new_hyp.ys.push_back(new_token);
        new_hyp.num_trailing_blanks = 0;
        new_hyp.timestamps.push_back(t + frame_offset);
        if (s && s->GetContextGraph()) {
          auto context_res = s->GetContextGraph()->ForwardOneStep(
              context_state, new_token, false /*strict_mode*/);
          context_score = std::get<0>(context_res);
          new_hyp.context_state = std::get<1>(context_res);
        }
      } else {
        ++new_hyp.num_trailing_blanks;
      }
      // We have already added prev[hyp_index].log_prob to p[new_token]
      new_hyp.log_prob = p[new_token] + context_score;

      cur.Add(std::move(new_hyp));
    }
  }

  result->hyps = std::move(cur);
  result->frame_offset += num_frames;
  auto hyp = result->hyps.GetMostProbable(true);

  // set decoder_out in case of endpointing
  ov::Tensor decoder_input = BuildDecoderInput({hyp});

  result->decoder_out = model_->RunDecoder(decoder_input);

  result->tokens = std::move(hyp.ys);
  result->num_trailing_blanks = hyp.num_trailing_blanks;
}

}  // namespace SherpaDeploy
