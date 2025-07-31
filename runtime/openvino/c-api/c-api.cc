/**
 * Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "c-api.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "runtime/core/display.h"
#include "runtime/openvino/model.h"
#include "runtime/openvino/recognizer.h"

struct SherpaOVRecognizer {
  std::unique_ptr<SherpaDeploy::Recognizer> recognizer;
};

struct SherpaOVStream {
  std::unique_ptr<SherpaDeploy::Stream> stream;
};

struct SherpaOVDisplay {
  std::unique_ptr<SherpaDeploy::Display> impl;
};

#define SHERPA_DEPLOY_OR(x, y) (x ? x : y)

SherpaOVRecognizer *CreateRecognizer(
    const SherpaOVRecognizerConfig *in_config) {
  // model_config
  SherpaDeploy::RecognizerConfig config;
  config.model_config.encoder_xml = in_config->model_config.encoder_xml;
  config.model_config.decoder_xml = in_config->model_config.decoder_xml;
  config.model_config.joiner_xml = in_config->model_config.joiner_xml;
  config.model_config.tokens = in_config->model_config.tokens;

  config.model_config.device = in_config->model_config.device;
  int32_t num_threads = SHERPA_DEPLOY_OR(in_config->model_config.num_threads, 1);
  config.model_config.num_threads = num_threads;

  // decoder_config
  config.decoder_config.method = in_config->decoder_config.decoding_method;
  config.decoder_config.num_active_paths =
      in_config->decoder_config.num_active_paths;

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
  auto ans = new SherpaOVRecognizer;
  ans->recognizer = std::move(recognizer);
  return ans;
}

void DestroyRecognizer(SherpaOVRecognizer *p) { delete p; }

SherpaOVStream *CreateStream(SherpaOVRecognizer *p) {
  auto ans = new SherpaOVStream;
  ans->stream = p->recognizer->CreateStream();
  return ans;
}

void DestroyStream(SherpaOVStream *s) { delete s; }

void AcceptWaveform(SherpaOVStream *s, float sample_rate,
                    const float *samples, int32_t n) {
  s->stream->AcceptWaveform(sample_rate, samples, n);
}

int32_t IsReady(SherpaOVRecognizer *p, SherpaOVStream *s) {
  return p->recognizer->IsReady(s->stream.get());
}

void Decode(SherpaOVRecognizer *p, SherpaOVStream *s) {
  p->recognizer->DecodeStream(s->stream.get());
}

SherpaOVResult *GetResult(SherpaOVRecognizer *p, SherpaOVStream *s) {
  std::string text = p->recognizer->GetResult(s->stream.get()).text;
  auto res = p->recognizer->GetResult(s->stream.get());

  auto r = new SherpaOVResult;
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

void DestroyResult(const SherpaOVResult *r) {
  delete[] r->text;
  delete[] r->timestamps;  // it is ok to delete a nullptr
  delete[] r->tokens;
  delete r;
}

void Reset(SherpaOVRecognizer *p, SherpaOVStream *s) {
  p->recognizer->Reset(s->stream.get());
}

void InputFinished(SherpaOVStream *s) { s->stream->InputFinished(); }

void Finalize(SherpaOVStream *s) { s->stream->Finalize(); }

int32_t IsEndpoint(SherpaOVRecognizer *p, SherpaOVStream *s) {
  return p->recognizer->IsEndpoint(s->stream.get());
}

SherpaOVDisplay *CreateDisplay(int32_t max_word_per_line) {
  SherpaOVDisplay *ans = new SherpaOVDisplay;
  ans->impl = std::make_unique<SherpaDeploy::Display>(max_word_per_line);
  return ans;
}

void DestroyDisplay(SherpaOVDisplay *display) { delete display; }

void SherpaOVPrint(SherpaOVDisplay *display, int32_t idx, const char *s) {
  display->impl->Print(idx, s);
}
