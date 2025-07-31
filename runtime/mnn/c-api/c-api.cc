/**
 * Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
 * Copyright (c)  2025  frankyj@foxmail.com      (authors: Jian You)
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

#include "c-api.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "runtime/core/display.h"
#include "runtime/mnn/model.h"
#include "runtime/mnn/recognizer.h"

struct SherpaDeployMnnRecognizer {
  std::unique_ptr<SherpaDeploy::Recognizer> recognizer;
};

struct SherpaDeployMnnStream {
  std::unique_ptr<SherpaDeploy::Stream> stream;
};

struct SherpaDeployMnnDisplay {
  std::unique_ptr<SherpaDeploy::Display> impl;
};

#define SHERPA_DEPLOY_OR(x, y) (x ? x : y)

SherpaDeployMnnRecognizer *CreateRecognizer(
    const SherpaDeployMnnRecognizerConfig *in_config) {
  // model_config
  SherpaDeploy::RecognizerConfig config;
  config.model_config.encoder_mnn = in_config->model_config.encoder_mnn;
  config.model_config.decoder_mnn = in_config->model_config.decoder_mnn;
  config.model_config.joiner_mnn = in_config->model_config.joiner_mnn;

  config.model_config.tokens = in_config->model_config.tokens;

  config.model_config.modeling_unit = SHERPA_DEPLOY_OR(in_config->model_config.modeling_unit, "bpe");
  config.model_config.bpe_vocab = SHERPA_DEPLOY_OR(in_config->model_config.bpe_vocab, "");

  int32_t num_threads = SHERPA_DEPLOY_OR(in_config->model_config.num_threads, 1);

  config.model_config.schedule_config.numThread = num_threads;
  config.model_config.schedule_config.type      = (MNNForwardType) in_config->model_config.forward_type;

  MNN::BackendConfig backendConfig;
  backendConfig.precision = (MNN::BackendConfig::PrecisionMode) in_config->model_config.backend_precision_mode;
  backendConfig.power = (MNN::BackendConfig::PowerMode) in_config->model_config.backend_power_mode;
  backendConfig.memory = (MNN::BackendConfig::MemoryMode) in_config->model_config.backend_memory_mode;
  config.model_config.schedule_config.backendConfig = &backendConfig;

  // decoder_config
  config.decoder_config.method = SHERPA_DEPLOY_OR(in_config->decoder_config.decoding_method, "greedy_search");
  config.decoder_config.num_active_paths = SHERPA_DEPLOY_OR(in_config->decoder_config.num_active_paths, 4);

  config.hotwords_file = SHERPA_DEPLOY_OR(in_config->hotwords_file, "");
  config.hotwords_score = SHERPA_DEPLOY_OR(in_config->hotwords_score, 1.5);

  config.enable_endpoint = in_config->enable_endpoint;

  config.endpoint_config.rule1.min_trailing_silence =
      in_config->rule1_min_trailing_silence;

  config.endpoint_config.rule2.min_trailing_silence =
      in_config->rule2_min_trailing_silence;

  config.endpoint_config.rule3.min_utterance_length =
      in_config->rule3_min_utterance_length;

  config.feat_config.sampling_rate =
      SHERPA_DEPLOY_OR(in_config->feat_config.sampling_rate, 16000);

  config.feat_config.feature_dim =
      SHERPA_DEPLOY_OR(in_config->feat_config.feature_dim, 80);

  auto recognizer = std::make_unique<SherpaDeploy::Recognizer>(config);

  if (!recognizer->GetModel()) {
    fprintf(stderr,"Failed to create the recognizer! Please check your config: %s",
              config.ToString().c_str());
    return nullptr;
  }

  fprintf(stderr, config.ToString().c_str());
  auto ans = new SherpaDeployMnnRecognizer;
  ans->recognizer = std::move(recognizer);
  return ans;
}

void DestroyRecognizer(SherpaDeployMnnRecognizer *p) { delete p; }

SherpaDeployMnnStream *CreateStream(SherpaDeployMnnRecognizer *p) {
  auto ans = new SherpaDeployMnnStream;
  ans->stream = p->recognizer->CreateStream();
  return ans;
}

void DestroyStream(SherpaDeployMnnStream *s) { delete s; }

void AcceptWaveform(SherpaDeployMnnStream *s, float sample_rate,
                    const float *samples, int32_t n) {
  s->stream->AcceptWaveform(sample_rate, samples, n);
}

int32_t IsReady(SherpaDeployMnnRecognizer *p, SherpaDeployMnnStream *s) {
  return p->recognizer->IsReady(s->stream.get());
}

void Decode(SherpaDeployMnnRecognizer *p, SherpaDeployMnnStream *s) {
  p->recognizer->DecodeStream(s->stream.get());
}

SherpaDeployMnnResult *GetResult(SherpaDeployMnnRecognizer *p, SherpaDeployMnnStream *s) {
  std::string text = p->recognizer->GetResult(s->stream.get()).text;
  auto res = p->recognizer->GetResult(s->stream.get());

  auto r = new SherpaDeployMnnResult;
  r->text = new char[text.size() + 1];
  std::copy(text.begin(), text.end(), const_cast<char *>(r->text));
  const_cast<char *>(r->text)[text.size()] = 0;
  r->count = res.tokens.size();
  if (r->count > 0) {
    // Each word ends with nullptr
    r->tokens = new char[text.size() + r->count];
    memset(reinterpret_cast<void *>(const_cast<char *>(r->tokens)), 0,
           text.size() + r->count);
    r->timestamps = new float[r->count];
    int pos = 0;
    for (int32_t i = 0; i < r->count; ++i) {
      memcpy(reinterpret_cast<void *>(const_cast<char *>(r->tokens + pos)),
             res.stokens[i].c_str(), res.stokens[i].size());
      pos += res.stokens[i].size() + 1;
      r->timestamps[i] = res.timestamps[i];
    }
  } else {
    r->timestamps = nullptr;
    r->tokens = nullptr;
  }

  return r;
}

void DestroyResult(const SherpaDeployMnnResult *r) {
  delete[] r->text;
  delete[] r->timestamps;  // it is ok to delete a nullptr
  delete[] r->tokens;
  delete r;
}

void Reset(SherpaDeployMnnRecognizer *p, SherpaDeployMnnStream *s) {
  p->recognizer->Reset(s->stream.get());
}

void InputFinished(SherpaDeployMnnStream *s) { s->stream->InputFinished(); }

void Finalize(SherpaDeployMnnStream *s) { s->stream->Finalize(); }

int32_t IsEndpoint(SherpaDeployMnnRecognizer *p, SherpaDeployMnnStream *s) {
  return p->recognizer->IsEndpoint(s->stream.get());
}

SherpaDeployMnnDisplay *CreateDisplay(int32_t max_word_per_line) {
  SherpaDeployMnnDisplay *ans = new SherpaDeployMnnDisplay;
  ans->impl = std::make_unique<SherpaDeploy::Display>(max_word_per_line);
  return ans;
}

void DestroyDisplay(SherpaDeployMnnDisplay *display) { delete display; }

void SherpaDeployMnnPrint(SherpaDeployMnnDisplay *display, int32_t idx, const char *s) {
  display->impl->Print(idx, s);
}
