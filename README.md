## TensorRT C++ Tutorial (Needs to be updated, Ignore currently)
This project demonstrates how to use the TensorRT C++ API for high performance GPU inference. It covers how to do the following:
- How to install TensorRT on Ubuntu 20.04
- How to generate a TRT engine file optimized for your GPU
- How to specify a simple optimization profile
- How to read / write data from / into GPU memory
- How to run synchronous inference
- How to work with models with dynamic batch sizes


## Getting Started
The following instructions assume you are using Ubuntu 20.04.
You will need to supply your own onnx model for this sample code. Ensure to specify a dynamic batch size when exporting the onnx model. 

### Prerequisites
- `sudo apt install build-essential`
- `sudo apt install python3-pip`
- `pip3 install cmake`
- Download TensorRT from here: https://developer.nvidia.com/nvidia-tensorrt-8x-download
- Extract, and then navigate to the `CMakeLists.txt` file and replace the `TODO` with the path to your TensorRT installation

### Building the library
- `mkdir build && cd build`
- `cmake ..`
- `make -j$(nproc)`
