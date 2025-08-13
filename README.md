# SherpaDeploy

SherpaDeploy is a project that implements various inference frameworks for K2 ASR models - [icefall](https://github.com/k2-fsa/icefall). The architecture is cloned from [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). 
The OnnxRuntime and NCNN implementations are from K2 official repos [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) and [sherpa-ncnn](https://github.com/k2-fsa/sherpa-ncnn), we add OpenVINO and MNN implementations. In the future, we hope add more implementations.

# Usage
## e.g. Build OpenVINO inference framework
```
mkdir build
cd build
cmake -T v142,host=x64 -A x64 -D CMAKE_BUILD_TYPE=Release -DSHERPA_ONE_ENABLE_OPENVINO=ON -DBUILD_SHARED_LIBS=ON ..
cmake --build . --config Release -- -m:6
```
