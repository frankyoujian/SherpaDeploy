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

#ifndef SHERPA_DEPLOY_MNN_C_API_H_
#define SHERPA_DEPLOY_MNN_C_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(SHERPA_DEPLOY_BUILD_SHARED_LIBS)
#define SHERPA_DEPLOY_EXPORT __declspec(dllexport)
#define SHERPA_DEPLOY_IMPORT __declspec(dllimport)
#else
#define SHERPA_DEPLOY_EXPORT
#define SHERPA_DEPLOY_IMPORT
#endif
#else  // WIN32
#define SHERPA_DEPLOY_EXPORT
#define SHERPA_DEPLOY_IMPORT SHERPA_DEPLOY_EXPORT
#endif

#if defined(SHERPA_DEPLOY_BUILD_MAIN_LIB)
#define SHERPA_DEPLOY_API SHERPA_DEPLOY_EXPORT
#else
#define SHERPA_DEPLOY_API SHERPA_DEPLOY_IMPORT
#endif

SHERPA_DEPLOY_API typedef struct SherpaDeployMnnModelConfig {
  /// Path to encoder.mnn
  const char *encoder_mnn;

  /// Path to decoder.mnn
  const char *decoder_mnn;

  /// Path to joiner.mnn
  const char *joiner_mnn;

  /// Path to tokens.txt
  const char *tokens;

  // Valid values:
  //  - cjkchar
  //  - bpe
  //  - cjkchar+bpe
  const char *modeling_unit;
  const char *bpe_vocab;

  /// Number of threads for neural network computation.
  int32_t num_threads;

  int32_t forward_type;

  int32_t backend_precision_mode;

  int32_t backend_power_mode;

  int32_t backend_memory_mode;

} SherpaDeployMnnModelConfig;

SHERPA_DEPLOY_API typedef struct SherpaDeployMnnDecoderConfig {
  /// Decoding method. Supported values are:
  /// greedy_search, modified_beam_search
  const char *decoding_method;

  /// Number of active paths for modified_beam_search.
  /// It is ignored when decoding_method is greedy_search.
  int32_t num_active_paths;
} SherpaDeployMnnDecoderConfig;

SHERPA_DEPLOY_API typedef struct SherpaDeployMnnFeatureExtractorConfig {
  // Sampling rate of the input audio samples. MUST match the one
  // expected by the model. For instance, it should be 16000 for models
  // from icefall.
  float sampling_rate;

  // feature dimension. Must match the one expected by the model.
  // For instance, it should be 80 for models from icefall.
  int32_t feature_dim;
} SherpaDeployMnnFeatureExtractorConfig;

SHERPA_DEPLOY_API typedef struct SherpaDeployMnnRecognizerConfig {
  SherpaDeployMnnFeatureExtractorConfig feat_config;
  SherpaDeployMnnModelConfig model_config;
  SherpaDeployMnnDecoderConfig decoder_config;

  /// 0 to disable endpoint detection.
  /// A non-zero value to enable endpoint detection.
  int32_t enable_endpoint;

  /// An endpoint is detected if trailing silence in seconds is larger than
  /// this value even if nothing has been decoded.
  /// Used only when enable_endpoint is not 0.
  float rule1_min_trailing_silence;

  /// An endpoint is detected if trailing silence in seconds is larger than
  /// this value after something that is not blank has been decoded.
  /// Used only when enable_endpoint is not 0.
  float rule2_min_trailing_silence;

  /// An endpoint is detected if the utterance in seconds is larger than
  /// this value.
  /// Used only when enable_endpoint is not 0.
  float rule3_min_utterance_length;

  /// hotwords file, each line is a hotword which is segmented into char by
  /// space if language is something like CJK, segment manually, if language is
  /// something like English, segment by bpe model.
  const char *hotwords_file;

  /// scale of hotwords, used only when hotwords_file is not empty
  float hotwords_score;
} SherpaDeployMnnRecognizerConfig;

SHERPA_DEPLOY_API typedef struct SherpaDeployMnnResult {
  // Recognized text
  const char *text;

  // Pointer to continuous memory which holds string based tokens
  // which are seperated by \0
  const char *tokens;

  // Pointer to continuous memory which holds timestamps which
  // are seperated by \0
  float *timestamps;

  // The number of tokens/timestamps in above pointer
  int32_t count;
} SherpaDeployMnnResult;

SHERPA_DEPLOY_API typedef struct SherpaDeployMnnRecognizer SherpaDeployMnnRecognizer;
SHERPA_DEPLOY_API typedef struct SherpaDeployMnnStream SherpaDeployMnnStream;

/// Create a recognizer.
///
/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          DestroyRecognizer() to free it to avoid memory leak.
SHERPA_DEPLOY_API SherpaDeployMnnRecognizer *CreateRecognizer(
    const SherpaDeployMnnRecognizerConfig *config);

/// Free a pointer returned by CreateRecognizer()
///
/// @param p A pointer returned by CreateRecognizer()
SHERPA_DEPLOY_API void DestroyRecognizer(SherpaDeployMnnRecognizer *p);

/// Create a stream for accepting audio samples
///
/// @param p A pointer returned by CreateRecognizer
/// @return Return a pointer to a stream. The caller MUST invoke
///         DestroyStream at the end to avoid memory leak.
SHERPA_DEPLOY_API SherpaDeployMnnStream *CreateStream(SherpaDeployMnnRecognizer *p);

SHERPA_DEPLOY_API void DestroyStream(SherpaDeployMnnStream *s);

/// Accept input audio samples and compute the features.
///
/// @param s  A pointer returned by CreateStream().
/// @param sample_rate  Sample rate of the input samples. If it is different
///                     from feat_config.sampling_rate, we will do resample.
///                     Caution: You MUST not use a different sampling_rate
///                     across different calls to AcceptWaveform()
/// @param samples A pointer to a 1-D array containing audio samples.
///                The range of samples has to be normalized to [-1, 1].
/// @param n  Number of elements in the samples array.
SHERPA_DEPLOY_API void AcceptWaveform(SherpaDeployMnnStream *s, float sample_rate,
                                    const float *samples, int32_t n);

/// Test if the stream has enough frames for decoding.
///
/// The common usage is:
///   while (IsReady(p, s)) {
///      Decode(p, s);
///   }
/// @param p A pointer returned by CreateRecognizer()
/// @param s A pointer returned by CreateStream()
/// @return Return 1 if the given stream is ready for decoding.
///         Return 0 otherwise.
SHERPA_DEPLOY_API int32_t IsReady(SherpaDeployMnnRecognizer *p, SherpaDeployMnnStream *s);

/// Pre-condition for this function:
///   You must ensure that IsReady(p, s) return 1 before calling this function.
///
/// @param p A pointer returned by CreateRecognizer()
/// @param s A pointer returned by CreateStream()
SHERPA_DEPLOY_API void Decode(SherpaDeployMnnRecognizer *p, SherpaDeployMnnStream *s);

/// Get the decoding results so far.
///
/// @param p A pointer returned by CreateRecognizer().
/// @param s A pointer returned by CreateStream()
/// @return A pointer containing the result. The user has to invoke
///         DestroyResult() to free the returned pointer to avoid memory leak.
SHERPA_DEPLOY_API SherpaDeployMnnResult *GetResult(SherpaDeployMnnRecognizer *p,
                                                SherpaDeployMnnStream *s);

/// Destroy the pointer returned by GetResult().
///
/// @param r A pointer returned by GetResult()
SHERPA_DEPLOY_API void DestroyResult(const SherpaDeployMnnResult *r);

/// Reset a stream
///
/// @param p A pointer returned by CreateRecognizer().
/// @param s A pointer returned by CreateStream().
SHERPA_DEPLOY_API void Reset(SherpaDeployMnnRecognizer *p, SherpaDeployMnnStream *s);

/// Signal that no more audio samples would be available.
/// After this call, you cannot call AcceptWaveform() any more.
///
/// @param s A pointer returned by CreateStream()
SHERPA_DEPLOY_API void InputFinished(SherpaDeployMnnStream *s);

SHERPA_DEPLOY_API void Finalize(SherpaDeployMnnStream *s);

/// Return 1 is an endpoint has been detected.
///
/// Common usage:
///   if (IsEndpoint(p, s)) {
///     Reset(p, s);
///   }
///
/// @param p A pointer returned by CreateRecognizer()
/// @return Return 1 if an endpoint is detected. Return 0 otherwise.
SHERPA_DEPLOY_API int32_t IsEndpoint(SherpaDeployMnnRecognizer *p,
                                   SherpaDeployMnnStream *s);

// for displaying results on Linux/macOS.
SHERPA_DEPLOY_API typedef struct SherpaDeployMnnDisplay SherpaDeployMnnDisplay;

/// Create a display object. Must be freed using DestroyDisplay to avoid
/// memory leak.
SHERPA_DEPLOY_API SherpaDeployMnnDisplay *CreateDisplay(int32_t max_word_per_line);

SHERPA_DEPLOY_API void DestroyDisplay(SherpaDeployMnnDisplay *display);

/// Print the result.
SHERPA_DEPLOY_API void SherpaDeployMnnPrint(SherpaDeployMnnDisplay *display, int32_t idx,
                                     const char *s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // SHERPA_DEPLOY_MNN_C_API_H_
