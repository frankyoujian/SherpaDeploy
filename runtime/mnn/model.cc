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
#include "model.h"
#include "zipformer-model.h"

#include <sstream>

namespace SherpaDeploy {

std::string ModelConfig::ToString() const {
  std::ostringstream os;
  os << "ModelConfig(";
  os << "encoder_mnn=\"" << encoder_mnn << "\", ";
  os << "decoder_mnn=\"" << decoder_mnn << "\", ";
  os << "joiner_mnn=\"" << joiner_mnn << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "modeling_unit=\"" << modeling_unit << "\", ";
  os << "bpe_vocab=\"" << bpe_vocab << "\", ";
  os << "num_threads=" << schedule_config.numThread << ")";

  return os.str();
}

void Model::InitNet(std::unique_ptr<MNN::Interpreter>& net, 
                      MNN::Session*& session, 
                      const char* model_path, 
                      const MNN::ScheduleConfig& schedule_config) {
  net = std::unique_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path));

  session = net->createSession(schedule_config);

  // for using dynamic axes when export ONNX, must comment below line, because we need to resize tensor during inference
  // see below for details:
  // https://mnn-docs.readthedocs.io/en/latest/cpp/Interpreter.html#releasemodel
  net->releaseModel();
}

// #if __ANDROID_API__ >= 9
// void Model::InitNet(AAssetManager *mgr, std::unique_ptr<MNN::Interpreter>& net, 
                      // MNN::Session*& session, 
                      // const char* model_path,
                      // const MNN::ScheduleConfig& schedule_config) {
// }
// #endif


std::unique_ptr<Model> Model::Create(const ModelConfig &config) {
  return std::make_unique<ZipformerModel>(config);
}

#if __ANDROID_API__ >= 9
std::unique_ptr<Model> Model::Create(AAssetManager *mgr,
                                     const ModelConfig &config) {
  return std::make_unique<ZipformerModel>(mgr, config);
}
#endif

}  // namespace SherpaDeploy
