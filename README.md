# SherpaDeploy

SherpaDeploy is a project that implements various inference frameworks for K2 ASR models - [icefall](https://github.com/k2-fsa/icefall). The architecture is cloned from [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). 
The OnnxRuntime and NCNN implementations are from K2 official repos [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [sherpa-ncnn](https://github.com/k2-fsa/sherpa-ncnn), we add OpenVINO and MNN implementations. In the future, we hope add more implementations.

# Usage
## Build
```
### choose OpenVINO
mkdir build
cd build
cmake -T v142,host=x64 -A x64 -D CMAKE_BUILD_TYPE=Release -DSHERPA_DEPLOY_ENABLE_OPENVINO=ON -DBUILD_SHARED_LIBS=ON ..
cmake --build . --config Release -- -m:6
```
For other inference framework, you can set SHERPA_DEPLOY_ENABLE_MNN, SHERPA_DEPLOY_ENABLE_NCNN, etc.

## Models
You can try pretrained OpenVINO and MNN models at: https://huggingface.co/frankyoujian/SherpaDeploy/tree/main

For NCNN and ONNXRuntime models, please download from K2 official link: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html

The provided OpenVINO and MNN models are trained on Gigaspeech and Voxpopuli(en_v1).

## Run
```
  ./build/bin/Release/sherpa-openvino-microphone \
    /path/to/encoder.xml \
    /path/to/decoder.xml \
    /path/to/joiner.xml \
    /path/to/tokens.txt
```

# Note
The newly added OpenVINO and MNN inference code only fouces on Streaming mode.
