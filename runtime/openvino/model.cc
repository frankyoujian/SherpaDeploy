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
  os << "encoder_xml=\"" << encoder_xml << "\", ";
  os << "decoder_xml=\"" << decoder_xml << "\", ";
  os << "joiner_xml=\"" << joiner_xml << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "device=\"" << device << "\", ";
  os << "num_threads=\"" << num_threads << "\")";

  return os.str();
}

std::unique_ptr<Model> Model::Create(const ModelConfig &config) {
  return std::make_unique<ZipformerModel>(config);
}

}  // namespace SherpaDeploy
