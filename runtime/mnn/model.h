/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#ifndef SHERPA_DEPLOY_MNN_MODEL_H_
#define SHERPA_DEPLOY_MNN_MODEL_H_

#include "MNN/Interpreter.hpp" // NOLINT

#include <memory>
#include <string>
#include <utility>
#include <vector>


namespace SherpaDeploy {

using TensorPtr = std::shared_ptr<MNN::Tensor>;

struct ModelConfig {
  std::string encoder_mnn;  // path to encoder.mnn
  std::string decoder_mnn;  // path to decoder.mnn
  std::string joiner_mnn;   // path to joiner.mnn
  std::string tokens;         // path to tokens.txt

  std::string modeling_unit = "bpe"; // default "bpe"
  std::string bpe_vocab;

  MNN::ScheduleConfig schedule_config;

  std::string ToString() const;
};

class Model {
 public:
  virtual ~Model() = default;

  /** Create a model from a config. */
  static std::unique_ptr<Model> Create(const ModelConfig &config);

// #if __ANDROID_API__ >= 9
//   static std::unique_ptr<Model> Create(AAssetManager *mgr,
//                                        const ModelConfig &config);
// #endif

  virtual std::vector<TensorPtr> GetEncoderInitStates() const = 0;

  /** Run the encoder network.
   *
   * @param features  A 2-d Tensor of shape (num_frames, feature_dim).
   *                  Note: features shape[0] = num_frames.
   *                        features shape[1] = feature_dim.
   * @param states It contains the states for the encoder network. Its exact
   *               content is determined by the underlying network.
   *
   * @return Return a pair containing:
   *   - encoder_out
   *   - next_states
  //  */
  virtual std::pair<TensorPtr, std::vector<TensorPtr>> RunEncoder(
      TensorPtr features, const std::vector<TensorPtr>& states) = 0;

  /** Run the decoder network.
   *
   * @param  decoder_input A Tensor of shape (num_paths, context_size). Note: Its underlying
   *                       content consists of integers, though its type is
   *                       float.
   *
   * @return Return a Tensor of shape (num_paths, decoder_dim)
   */
  virtual TensorPtr RunDecoder(TensorPtr decoder_input) = 0;

  /** Run the joiner network.
   *
   * @param encoder_out  A Tensor of shape (num_frames, encoder_dim)
   * @param decoder_out  A Tensor of shape (num_paths, decoder_dim)
   *
   * @return Return the joiner output which is of shape (num_paths, vocab_size)
   */
  virtual TensorPtr RunJoiner(TensorPtr encoder_out,
                              TensorPtr decoder_out) = 0;

  virtual int32_t ContextSize() const = 0;

  virtual int32_t BlankId() const { return 0; }

  // The encoder takes this number of frames as input
  virtual int32_t Segment() const = 0;

  // Advance the feature extractor by this number of frames after
  // running the encoder network
  virtual int32_t Offset() const = 0;

  static void InitNet(std::unique_ptr<MNN::Interpreter>& net, 
                      MNN::Session*& session, 
                      const char* model_path,
                      const MNN::ScheduleConfig& schedule_config);

// #if __ANDROID_API__ >= 9
//   static void InitNet(AAssetManager *mgr, std::unique_ptr<MNN::Interpreter>& net, 
                      // MNN::Session*& session, 
                      // const char* model_path,
                      // const MNN::ScheduleConfig& schedule_config);
// #endif
};

}  // namespace SherpaDeploy

#endif  // SHERPA_DEPLOY_MNN_MODEL_H_
