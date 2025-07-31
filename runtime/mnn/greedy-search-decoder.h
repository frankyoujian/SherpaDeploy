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

#ifndef SHERPA_DEPLOY_MNN_GREEDY_SEARCH_DECODER_H_
#define SHERPA_DEPLOY_MNN_GREEDY_SEARCH_DECODER_H_

#include "decoder.h"
#include "model.h"

namespace MNN {
    class Tensor;
}

namespace SherpaDeploy {

class GreedySearchDecoder : public Decoder {
 public:
  explicit GreedySearchDecoder(Model *model) : model_(model) {}

  DecoderResult GetEmptyResult() const override;

  void StripLeadingBlanks(DecoderResult * /*r*/) const override;

  void Decode(TensorPtr encoder_out, DecoderResult *result) override;

 private:
  TensorPtr BuildDecoderInput(const DecoderResult &result) const;

 private:
  Model *model_;  // not owned
};

}  // namespace SherpaDeploy

#endif  // SHERPA_DEPLOY_MNN_GREEDY_SEARCH_DECODER_H_
