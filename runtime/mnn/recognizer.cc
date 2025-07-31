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

#include "recognizer.h"

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime/core/context-graph.h"
#include "runtime/core/utils.h"
#include "decoder.h"
#include "greedy-search-decoder.h"
#include "modified-beam-search-decoder.h"
#include "mnn-utils.h"

#include "ssentencepiece/csrc/ssentencepiece.h"

#if __ANDROID_API__ >= 9
#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#include "android/log.h"
#endif

namespace SherpaDeploy {

static RecognitionResult Convert(const DecoderResult &src,
                                 const SherpaDeploy::SymbolTable &sym_table,
                                 int32_t frame_shift_ms,
                                 int32_t subsampling_factor) {
  RecognitionResult ans;
  ans.stokens.reserve(src.tokens.size());
  ans.timestamps.reserve(src.timestamps.size());

  std::string text;
  for (auto i : src.tokens) {
    auto sym = sym_table[i];
    text.append(sym);
    ans.stokens.push_back(sym);
  }

  ans.text = std::move(text);
  ans.tokens = src.tokens;
  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;
  for (auto t : src.timestamps) {
    float time = frame_shift_s * t;
    ans.timestamps.push_back(time);
  }
  return ans;
}

std::string RecognitionResult::ToString() const {
  std::ostringstream os;

  os << "text: " << text << "\n";
  os << "timestamps: ";
  for (const auto &t : timestamps) {
    os << t << " ";
  }
  os << "\n";

  return os.str();
}

std::string RecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "RecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "decoder_config=" << decoder_config.ToString() << ", ";
  os << "endpoint_config=" << endpoint_config.ToString() << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ", ";
  os << "hotwords_file=\"" << hotwords_file << "\", ";
  os << "hotwrods_score=" << hotwords_score << ")";

  return os.str();
}

class Recognizer::Impl {
 public:
  explicit Impl(const RecognizerConfig &config)
      : config_(config),
        model_(Model::Create(config.model_config)),
        endpoint_(config.endpoint_config),
        sym_(config.model_config.tokens) {
    if (config.decoder_config.method == "greedy_search") {
      decoder_ = std::make_unique<GreedySearchDecoder>(model_.get());
    } else if (config.decoder_config.method == "modified_beam_search") {
      decoder_ = std::make_unique<ModifiedBeamSearchDecoder>(
          model_.get(), config.decoder_config.num_active_paths);

      if (!config_.model_config.bpe_vocab.empty()) {
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(
            config_.model_config.bpe_vocab);
      }

      if (!config_.hotwords_file.empty()) {
        InitHotwords();
      }
    } else {
      fprintf(stderr, "Unsupported method: %s", config.decoder_config.method.c_str());
      exit(-1);
    }
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const RecognizerConfig &config)
      : config_(config),
        model_(Model::Create(mgr, config.model_config)),
        endpoint_(config.endpoint_config),
        sym_(mgr, config.model_config.tokens) {
    if (config.decoder_config.method == "greedy_search") {
      decoder_ = std::make_unique<GreedySearchDecoder>(model_.get());
    } else if (config.decoder_config.method == "modified_beam_search") {
      decoder_ = std::make_unique<ModifiedBeamSearchDecoder>(
          model_.get(), config.decoder_config.num_active_paths);

      if (!config_.model_config.bpe_vocab.empty()) {
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(
            config_.model_config.bpe_vocab);
      }

      if (!config_.hotwords_file.empty()) {
        InitHotwords(mgr);
      }
    } else {
      fprintf(stderr, "Unsupported method: %s", config.decoder_config.method.c_str());
      exit(-1);
    }
  }
#endif

  std::unique_ptr<Stream> CreateStream() const {
    if (hotwords_.empty()) {
      auto stream = std::make_unique<Stream>(config_.feat_config);
      stream->SetResult(decoder_->GetEmptyResult());
      stream->SetStates(model_->GetEncoderInitStates());
      return stream;
    } else {
      auto r = decoder_->GetEmptyResult();

      auto context_graph =
          std::make_shared<SherpaDeploy::ContextGraph>(hotwords_, config_.hotwords_score, boost_scores_);

      auto stream =
          std::make_unique<Stream>(config_.feat_config, context_graph);

      if (stream->GetContextGraph()) {
        // r.hyps has only one element.
        for (auto it = r.hyps.begin(); it != r.hyps.end(); ++it) {
          it->second.context_state = stream->GetContextGraph()->Root();
        }
      }

      stream->SetResult(r);
      stream->SetStates(model_->GetEncoderInitStates());

      return stream;
    }
  }

  bool IsReady(Stream *s) const {
    return s->GetNumProcessedFrames() + model_->Segment() < s->NumFramesReady();
  }

  void DecodeStream(Stream *s) const {
    int32_t segment = model_->Segment();
    int32_t offset = model_->Offset();

    auto frames_out = s->GetFrames(s->GetNumProcessedFrames(), segment);
    std::vector<float> features_vec = std::get<0>(frames_out);
    int32_t feature_dim = std::get<1>(frames_out);

    TensorPtr features = TensorPtr(MNN::Tensor::create<float>({segment, feature_dim}, NULL, MNN::Tensor::CAFFE));
    float* p_dst = features->host<float>();
    float* p_src = features_vec.data();

    for (int32_t i = 0; i != segment; ++i) {
      std::copy(p_src, p_src + feature_dim, p_dst);
      p_src += feature_dim;
      p_dst += feature_dim;
    }

    s->GetNumProcessedFrames() += offset;
    std::vector<TensorPtr> pre_states = s->GetStates();
    std::vector<TensorPtr> cur_states;

    TensorPtr encoder_out;
    std::tie(encoder_out, cur_states) = model_->RunEncoder(features, pre_states);

    if (s->GetContextGraph()) {
      decoder_->Decode(encoder_out, s, &s->GetResult());
    } else {
      decoder_->Decode(encoder_out, &s->GetResult());
    }
    s->SetStates(cur_states);

  }

  bool IsEndpoint(Stream *s) const {
    if (!config_.enable_endpoint) return false;
    int32_t num_processed_frames = s->GetNumProcessedFrames();

    // frame shift is 10 milliseconds
    float frame_shift_in_seconds = 0.01;

    // subsampling factor is 4
    int32_t trailing_silence_frames = s->GetResult().num_trailing_blanks * 4;

    return endpoint_.IsEndpoint(num_processed_frames, trailing_silence_frames,
                                frame_shift_in_seconds);
  }

  void Reset(Stream *s) const {
    auto r = decoder_->GetEmptyResult();

    if (s->GetContextGraph()) {
      for (auto it = r.hyps.begin(); it != r.hyps.end(); ++it) {
        it->second.context_state = s->GetContextGraph()->Root();
      }
    }
    // Caution: We need to keep the decoder output state
    TensorPtr decoder_out = s->GetResult().decoder_out;
    s->SetResult(r);
    s->GetResult().decoder_out = decoder_out;

    // don't reset encoder state
    // s->SetStates(model_->GetEncoderInitStates());

    // reset feature extractor
    // Note: We only reset the counter. The underlying audio samples are
    // still kept in memory
    s->Reset();
  }

  RecognitionResult GetResult(Stream *s) const {
    if (IsEndpoint(s)) {
      s->Finalize();
    }
    DecoderResult decoder_result = s->GetResult();

    decoder_->StripLeadingBlanks(&decoder_result);

    // Those 2 parameters are figured out from sherpa source code
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = 4;
    return Convert(decoder_result, sym_, frame_shift_ms, subsampling_factor);
  }

  const Model *GetModel() const { return model_.get(); }

 private:
#if __ANDROID_API__ >= 9
  void InitHotwords(AAssetManager *mgr) {
    AAsset *asset = AAssetManager_open(mgr, config_.hotwords_file.c_str(),
                                       AASSET_MODE_BUFFER);
    if (!asset) {
      __android_log_print(ANDROID_LOG_FATAL, "sherpa-ncnn",
                          "hotwords_file: Load %s failed",
                          config_.hotwords_file.c_str());
      exit(-1);
    }

    auto p = reinterpret_cast<const char *>(AAsset_getBuffer(asset));
    size_t asset_length = AAsset_getLength(asset);
    std::istrstream is(p, asset_length);

    if (!EncodeHotwords(is, config_.model_config.modeling_unit, sym_,
                        bpe_encoder_.get(), &hotwords_, &boost_scores_)) {
      fprintf(stderr,
          "Failed to encode some hotwords, skip them already, see logs above "
          "for details.\n");
    }

    AAsset_close(asset);
  }
#endif

  void InitHotwords() {
    // each line in hotwords_file contains space-separated words

    std::ifstream is(config_.hotwords_file);
    if (!is) {
      fprintf(stderr, "Open hotwords file failed: %s", config_.hotwords_file.c_str());
      exit(-1);
    }

    if (!bpe_encoder_) {
      fprintf(stderr, "bpe encoder is null, can not encode hot words!");
      exit(-1);
    }
    
    if (!EncodeHotwords(is, config_.model_config.modeling_unit, sym_,
                        bpe_encoder_.get(), &hotwords_, &boost_scores_)) {
      fprintf(stderr,
          "Failed to encode some hotwords, skip them already, see logs above "
          "for details.\n");
    }
  }

 private:
  RecognizerConfig config_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<Decoder> decoder_;
  SherpaDeploy::Endpoint endpoint_;
  SherpaDeploy::SymbolTable sym_;
  std::unique_ptr<ssentencepiece::Ssentencepiece> bpe_encoder_;
  std::vector<std::vector<int32_t>> hotwords_;
  std::vector<float> boost_scores_;
};

Recognizer::Recognizer(const RecognizerConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
Recognizer::Recognizer(AAssetManager *mgr, const RecognizerConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

Recognizer::~Recognizer() = default;

std::unique_ptr<Stream> Recognizer::CreateStream() const {
  return impl_->CreateStream();
}

bool Recognizer::IsReady(Stream *s) const { return impl_->IsReady(s); }

void Recognizer::DecodeStream(Stream *s) const { impl_->DecodeStream(s); }

bool Recognizer::IsEndpoint(Stream *s) const { return impl_->IsEndpoint(s); }

void Recognizer::Reset(Stream *s) const { impl_->Reset(s); }

RecognitionResult Recognizer::GetResult(Stream *s) const {
  return impl_->GetResult(s);
}

const Model *Recognizer::GetModel() const { return impl_->GetModel(); }

}  // namespace SherpaDeploy
