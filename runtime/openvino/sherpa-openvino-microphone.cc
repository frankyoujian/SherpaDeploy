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

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <cctype>  // std::tolower
#include <algorithm>

#include "portaudio.h"  // NOLINT
#include "runtime/core/display.h"
#include "runtime/core/microphone.h"

#include "runtime/openvino/c-api/c-api.h"

bool stop = false;

static int32_t RecordCallback(const void *input_buffer,
                              void * /*output_buffer*/,
                              unsigned long frames_per_buffer,  // NOLINT
                              const PaStreamCallbackTimeInfo * /*time_info*/,
                              PaStreamCallbackFlags /*status_flags*/,
                              void *user_data) {
  auto s = reinterpret_cast<SherpaOVStream*>(user_data);

  AcceptWaveform(s, 16000, reinterpret_cast<const float *>(input_buffer),
                    frames_per_buffer);

  return stop ? paComplete : paContinue;
}

static void Handler(int32_t sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
};

int32_t main(int32_t argc, char *argv[]) {
  if (argc < 5 || argc > 9) {
    const char *usage = R"usage(
Usage:
  ./bin/sherpa-openvino-microphone \
    /path/to/encoder.xml \
    /path/to/decoder.xml \
    /path/to/joiner.xml \
    /path/to/tokens.txt \
    [device] [num_threads] [decode_method, can be greedy_search/modified_beam_search] [hotwords_file] [hotwords_score]

)usage";
    fprintf(stderr, "%s\n", usage);
    fprintf(stderr, "argc, %d\n", argc);

    return 0;
  }
  signal(SIGINT, Handler);

  SherpaOVRecognizerConfig config;
  config.model_config.encoder_xml = argv[1];
  config.model_config.decoder_xml = argv[2];
  config.model_config.joiner_xml = argv[3];
  config.model_config.tokens = argv[4];

  config.model_config.device = "CPU";
  if (argc >= 6) {
    config.model_config.device = argv[5];
  }

  config.model_config.num_threads = 4;
  if (argc >= 7 && atoi(argv[6]) > 0) {
    config.model_config.num_threads = atoi(argv[6]);
  }

  const float expected_sampling_rate = 16000;
  config.decoder_config.decoding_method = "greedy_search"; // default
  if (argc >= 8) {
    if (std::string method = argv[7]; method == "modified_beam_search") {
      config.decoder_config.decoding_method = argv[7];
      config.decoder_config.num_active_paths = 4;
    }
  }

  config.hotwords_file = "";
  if (argc >= 9) {
    config.hotwords_file = argv[8];
  }

  config.hotwords_score = 0;
  if (argc == 10) {
    config.hotwords_score = atof(argv[9]);
  }

  config.enable_endpoint = true;

  config.rule1_min_trailing_silence = 2.4;
  config.rule2_min_trailing_silence = 1.2;
  config.rule3_min_utterance_length = 300;

  config.feat_config.sampling_rate = expected_sampling_rate;
  config.feat_config.feature_dim = 80;

  SherpaOVRecognizer* recognizer = CreateRecognizer(&config);
  SherpaOVStream* s = CreateStream(recognizer);

  SherpaDeploy::Microphone mic;

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "Num devices: %d\n", num_devices);

  PaStreamParameters param;

  param.device = Pa_GetDefaultInputDevice();
  if (param.device == paNoDevice) {
    fprintf(stderr, "No default input device found\n");
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Use default device: %d\n", param.device);

  const PaDeviceInfo *info = Pa_GetDeviceInfo(param.device);
  fprintf(stderr, "  Name: %s\n", info->name);
  fprintf(stderr, "  Max input channels: %d\n", info->maxInputChannels);

  param.channelCount = 1;
  param.sampleFormat = paFloat32;

  param.suggestedLatency = info->defaultLowInputLatency;
  param.hostApiSpecificStreamInfo = nullptr;
  const float sample_rate = 16000;

  PaStream *stream;
  PaError err =
      Pa_OpenStream(&stream, &param, nullptr, /* &outputParameters, */
                    sample_rate,
                    0,          // frames per buffer
                    paClipOff,  // we won't output out of range samples
                                // so don't bother clipping them
                    RecordCallback, s);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  err = Pa_StartStream(stream);
  fprintf(stderr, "Started\n");

  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  std::string last_text;
  int32_t segment_index = 0;
  SherpaDeploy::Display display;
  while (!stop) {
    while (IsReady(recognizer, s)) {
      Decode(recognizer, s);
    }

    bool is_endpoint = IsEndpoint(recognizer, s);

    if (is_endpoint) {
      Finalize(s);
    }

    auto r = GetResult(recognizer, s);
    std::string text(r->text);

    if (!text.empty() && last_text != text) {
      last_text = text;

      display.Print(segment_index, text);
    }

    if (is_endpoint) {
      if (!text.empty()) {
        ++segment_index;
      }

      Reset(recognizer, s);
    }

    DestroyResult(r);

    Pa_Sleep(20);  // sleep for 20ms
  }

  err = Pa_CloseStream(stream);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(EXIT_FAILURE);
  }

  DestroyStream(s);
  DestroyRecognizer(recognizer);

  return 0;
}
