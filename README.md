# Unity-TensorRT

This repository allows one to use TensorRT in Unity. Currently, it is at a pre-alpha level (i.e, it barely just works, and could be significantly optimized).

## Current abilities
* Load an ONNX model and perform inference on it, using Unity Texture2D's as input and output. This means there's no GPU<->CPU copies, the only copy is from texture memory to linear memory and back (to run the model), which is only on the GPU.

## Features to add
* Serialize and deserialize TensorRT engines to speed up startup time.
* Allow multiple models to be used at once.
* Benchmark vs https://github.com/keijiro/Pix2Pix .

## Limitations
* Currently only supports DirectX 11 (easy to solve) and Windows 10 (harder to solve).
* Dispose might not clean up all memory.

## How to use
* If you're using CUDA 9.0 and Windows 10, you can use the [Unity project](tensor rt/) as is. It uses the `unitytrtcuda.dll` included in the [unitytrtcuda/x64/Release](unitytrtcuda/x64/Release). 
* If not, you'll have to build from source.

### Build from source
* Install Visual Studio 2017 and **Visual Studio v140 build tools** (Re-run the installer, click on modify, and check the box for it if you already installed Visual Studio).
* Install CUDA. It should also install the Visual studio build tools for CUDA. You can check this worked by if you click `New Project` it gives you an option to use the CUDA template for a project.
* [Download TensorRT](https://developer.nvidia.com/tensorrt). It's free, but you have to sign-up. This code was built with TensorRT 5.03, not sure how it'll fare with earlier or later versions.
* Open the Visual Studio project in the [unitytrtcuda](unitytrtcuda) folder.  Modify the project's include directories and the linker's Additional Libraries to point at your downloaded TensorRT.
* Compile. This should produce a DLL. Copy this DLL, the TensorRT DLLs, as well as the copied CUDA DLLs in the output folder, to the Unity project's Assets/Plugins folder. Note that you have to restart unity for DLL changes to kick in.

