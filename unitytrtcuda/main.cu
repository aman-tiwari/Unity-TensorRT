
//#include "gsl-lite.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include "Unity/IUnityInterface.h"
#include "Unity/IUnityGraphics.h"

#include <cuda_d3d11_interop.h>

//#include "stdafx.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

static IUnityGraphics *unityGraphics = nullptr;

cudaError_t registerTextureToCuda(void* texture, cudaGraphicsResource_t* resource) {
	switch (unityGraphics->GetRenderer()) {
	//case kUnityGfxRendererOpenGL:
	//	return cudaGraphicsGLRegisterImage(&resource, (GLuint)texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	case kUnityGfxRendererD3D11:
		return cudaGraphicsD3D11RegisterResource(resource, (ID3D11Resource*)texture, cudaGraphicsRegisterFlagsNone);
	//case kUnityGfxRendererD3D9:
	//	return cudaGraphicsD3D9RegisterResource(&resource, (IDirect3DResource9*)texture, cudaGraphicsRegisterFlagsNone);
	default:
		return cudaErrorNotYetImplemented;
	}
}

namespace nv = nvinfer1;

typedef void(*FuncPtr)(int severity, const char *);

FuncPtr Debug = nullptr;

struct Logger : nv::ILogger {
	virtual void log(nv::ILogger::Severity severity, const char* msg) override
	{
		if (Debug == nullptr) std::cerr << (int)severity << ": " << msg << std::endl;
		else {
			std::string ret{ std::to_string((int)severity) + ": " + msg };
			Debug((int)severity, ret.c_str());
		}
	}
};

Logger logger;
#define LOG(X) logger.log(nv::ILogger::Severity::kERROR, (X).c_str());


static bool logIfErr(cudaError err, int line) {
	if (err != cudaSuccess) {
		std::string msg{  };
		msg += "line: ";
		msg += std::to_string(line);
		logger.log(nv::ILogger::Severity::kERROR, msg.c_str());
		logger.log(nv::ILogger::Severity::kERROR, cudaGetErrorName(err));
		logger.log(nv::ILogger::Severity::kERROR, cudaGetErrorString(err));
		return true;
	}
	return false;
}


#define check(X) if(logIfErr(X, __LINE__)) { return false; }

template<typename T>
struct Destroy {

	void operator()(T* t) {
		if (t != nullptr) {
			t->destroy();
		}
	}
};

class CudaStream {
	cudaStream_t stream;
public:
	CudaStream() {
		cudaStreamCreate(&stream);
	}

	operator cudaStream_t() {
		return stream;
	}

	virtual ~CudaStream() {
		cudaStreamDestroy(stream);
	}
};


class CudaEvent
{
public:
	CudaEvent()
	{
		cudaEventCreate(&mEvent);
	}

	operator cudaEvent_t()
	{
		return mEvent;
	}

	virtual ~CudaEvent()
	{
		cudaEventDestroy(mEvent);
	}

private:
	cudaEvent_t mEvent;
};

constexpr size_t MAX_WORKSPACE_SIZE = 1ULL << 30; // 1 GB

template<class T>
using nv_unique = std::unique_ptr<T, ::Destroy<T>>;


struct State {
	nv_unique<nv::ICudaEngine> engine{ nullptr };
	int inputBinding = 0;
	int outputBinding = 0;
	int inputSize = 1;
	int outputSize = 1;
	void* bindings[2];
	cudaGraphicsResource_t resources[2];
};

static State state;

std::string extents_to_str(cudaExtent const& extents) {
	std::stringstream res;
	res << "Extents[depth: ";
	res << extents.depth;
	res << ", width:" << extents.width;
	res << ", height: " << extents.height;
	res << "]";
	return res.str();

}

std::string channeldesc_to_str(cudaChannelFormatDesc const& desc) {
	std::stringstream res;
	res << "ChannelDesc[F:";
	res << (desc.f == cudaChannelFormatKindFloat
		? "Float"
		: desc.f == cudaChannelFormatKindUnsigned
		? "Unsigned"
		: desc.f == cudaChannelFormatKindSigned
		? "Signed"
		: desc.f == cudaChannelFormatKindNone
		? "None"
		: "Unknown");
	res << "," << desc.w << "," << desc.x << "," << desc.y << "," << desc.z;
	res << "]";
	return res.str();
}

std::string vec_to_str(std::vector<float> const& vec) {
	std::stringstream res;
	res << "vec [";
	for (auto& elem : vec) res << elem << ",";
	res << "]";
	return res.str();
}

std::string dims_to_str(nv::Dims dims) {
	std::stringstream res;
	res << "Num Dims: ";
	res << dims.nbDims;
	res << "[";

	for (int i = 0; i < dims.nbDims; i++) {
		res << dims.d[i];
		if (i != dims.nbDims - 1) res << ",";
	}
	res << "]";
	return res.str();
}

struct mapped_resources {
	int count_;
	cudaGraphicsResource_t* resources_;
	cudaStream_t stream_;
	mapped_resources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream)
		: count_(count), resources_(resources), stream_(stream) {
	}
	cudaError_t map() {
		return cudaGraphicsMapResources(count_, resources_, stream_);
	}
	~mapped_resources() {
		cudaGraphicsUnmapResources(count_, resources_, stream_);
	}
};


nv_unique<nv::ICudaEngine> createCudaEngine(std::string onnxModelPath, int batchSize = 1)
{
	nv_unique<nv::IBuilder> builder{ nv::createInferBuilder(logger) };
	nv_unique<nv::INetworkDefinition> network{ builder->createNetwork() };
	nv_unique<nvonnxparser::IParser> parser{ nvonnxparser::createParser(*network, logger) };

	if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nv::ILogger::Severity::kINFO)))
	{
		LOG(std::string("ERROR: could not parse input engine."));
		return nullptr;
	}

	// Build TensorRT engine optimized based on for batch size of input data provided.
	builder->setMaxBatchSize(batchSize);
	// Allow TensorRT to use fp16 mode kernels internally.
	// Note that Input and Output tensors will still use 32 bit float type by default.
	builder->setFp16Mode(builder->platformHasFastFp16());
	builder->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
	nv_unique<nv::ICudaEngine> ret{ builder->buildCudaEngine(*network) };
	return std::move(ret); // Build and return TensorRT engine.
}

extern "C" {

	UNITY_INTERFACE_EXPORT void setDebugFunction(FuncPtr fp) {
		Debug = fp;
	}

	UNITY_INTERFACE_EXPORT bool loadModel(char* path) {
		auto new_engine = createCudaEngine(std::string(path));
		if (new_engine.get() == nullptr) {
			return false;
		}

		std::swap(state.engine, new_engine);

		if (state.engine->getNbBindings() != 2) {
			LOG(std::string("only models with 2 bindings are supported for now"));
			return false;
		}
		auto initmsg = std::string("engine created, nbBindings: ") + std::to_string(state.engine->getNbBindings());
		LOG(initmsg);

		for (int i = 0; i < state.engine->getNbBindings(); i++) {
			if (state.engine->bindingIsInput(i)) {
				state.inputBinding = i;
			}
			else {
				state.outputBinding = i;
			}

		}

		auto stateset = std::string("state set");
		LOG(stateset);

		state.inputSize = 1;
		LOG(std::string("inputs:"));

		nv::Dims inputDims{ state.engine->getBindingDimensions(state.inputBinding) };
		LOG(dims_to_str(inputDims));
		
		for (int j = 0; j < inputDims.nbDims; j++) {
			state.inputSize *= inputDims.d[j];
		}

		LOG(std::string("output:"));
		state.outputSize = 1;
		nv::Dims outputDims{ state.engine->getBindingDimensions(state.outputBinding) };
		LOG(dims_to_str(outputDims));

		for (int j = 0; j < outputDims.nbDims; j++) {
			state.outputSize *= outputDims.d[j];
		}

		cudaDeviceSynchronize();
		return true;
	}

	UNITY_INTERFACE_EXPORT bool bindTextures(void* inputTexture, void* outputTexture) {
		if (state.resources[state.inputBinding] != nullptr) {
			check(cudaGraphicsUnregisterResource(state.resources[state.inputBinding]));
			state.resources[state.inputBinding] = nullptr;
		}
		if (state.resources[state.outputBinding] != nullptr) {
			check(cudaGraphicsUnregisterResource(state.resources[state.outputBinding]));
			state.resources[state.outputBinding] = nullptr;
		}

		cudaGraphicsResource_t inputRes;
		check(cudaGraphicsD3D11RegisterResource(&inputRes,
			(ID3D11Resource*)inputTexture, cudaGraphicsRegisterFlagsNone));
		state.resources[state.inputBinding] = inputRes;

		cudaGraphicsResource_t outputRes;
		check(cudaGraphicsD3D11RegisterResource(&outputRes,
			(ID3D11Resource*)outputTexture, cudaGraphicsRegisterFlagsNone));
		state.resources[state.outputBinding] = outputRes;

		CudaStream stream;
		mapped_resources resources(2, state.resources, stream);
		check(resources.map());

		cudaStreamSynchronize(stream);

		cudaArray_t inputArr;
		check(cudaGraphicsSubResourceGetMappedArray(&inputArr, inputRes, 0, 0));
		cudaChannelFormatDesc idesc;
		cudaExtent iextent;
		unsigned int iflags;
		check(cudaArrayGetInfo(&idesc, &iextent, &iflags, inputArr));
		size_t inputTexSize = iextent.width * iextent.height;

		LOG(std::string("input tex (as cudaArray) info:"));
		LOG(channeldesc_to_str(idesc));
		LOG(extents_to_str(iextent));
		LOG(std::to_string(iflags));

		if (inputTexSize != state.inputSize) {
			LOG(std::string("Expected input texture with ")
				+ std::to_string(state.inputSize)
				+ "size but got"
				+ std::to_string(inputTexSize));
			return false;
		}

		check(cudaMalloc(&state.bindings[state.inputBinding], state.inputSize * sizeof(float)));

		cudaArray_t outputArr;
		check(cudaGraphicsSubResourceGetMappedArray(&outputArr, outputRes, 0, 0));
		cudaChannelFormatDesc odesc;
		cudaExtent oextent;
		unsigned int oflags;
		check(cudaArrayGetInfo(&odesc, &oextent, &oflags, outputArr));
		size_t outputTexSize = oextent.width * oextent.height;

		LOG(std::string("output tex (as cudaArray) info:"));
		LOG(channeldesc_to_str(odesc));
		LOG(extents_to_str(oextent));
		LOG(std::to_string(oflags));

		if (outputTexSize != state.outputSize) {
			LOG(std::string("Expected output texture with ")
				+ std::to_string(state.outputSize)
				+ "size but got"
				+ std::to_string(outputTexSize));
			return false;
		}

		check(cudaMalloc(&state.bindings[state.outputBinding], state.outputSize * sizeof(float)));
		return true;
	}

	UNITY_INTERFACE_EXPORT bool inferOnTextures() {

		CudaStream stream;
		mapped_resources resources(2, state.resources, stream);
		check(resources.map());

		cudaStreamSynchronize(stream);
		check(cudaPeekAtLastError());

		//LOG(std::string("mapped resources"));

		cudaGraphicsResource_t inputRes = state.resources[state.inputBinding];
		cudaGraphicsResource_t outputRes = state.resources[state.outputBinding];

		nv_unique<nv::IExecutionContext>context { state.engine->createExecutionContext() };
		//LOG(std::string("created execution context"));

		cudaArray_t inputArr;
		check(cudaGraphicsSubResourceGetMappedArray(&inputArr, inputRes, 0, 0));
		cudaChannelFormatDesc idesc;
		cudaExtent iextent;
		unsigned int iflags;
		check(cudaArrayGetInfo(&idesc, &iextent, &iflags, inputArr));
		size_t inputTexSize = iextent.width * iextent.height;

		cudaArray_t outputArr;
		check(cudaGraphicsSubResourceGetMappedArray(&outputArr, outputRes, 0, 0));
		cudaChannelFormatDesc odesc;
		cudaExtent oextent;
		unsigned int oflags;
		check(cudaArrayGetInfo(&odesc, &oextent, &oflags, outputArr));
		size_t outputTexSize = oextent.width * oextent.height;
		
		check(cudaMemcpy2DFromArrayAsync(state.bindings[state.inputBinding], iextent.width * sizeof(float),
			inputArr, 0, 0, iextent.width * sizeof(float), iextent.height, cudaMemcpyDeviceToDevice, stream));

		check(cudaMemcpy2DFromArrayAsync(state.bindings[state.outputBinding], oextent.width * sizeof(float),
			outputArr, 0, 0, oextent.width * sizeof(float), oextent.height, cudaMemcpyDeviceToDevice, stream));

		context->enqueue(1, state.bindings, stream, nullptr);

		check(cudaStreamSynchronize(stream));
		return true;
	}

	UNITY_INTERFACE_EXPORT bool bindingSize(int bindingIdx, uint64_t* numDims, uint64_t* dimsArr, bool* isInput) {
		if (bindingIdx < 0 || bindingIdx >= state.engine->getNbBindings()) {
			LOG(std::string("invalid binding index requested"));
			return false;
		}
		*isInput = state.engine->bindingIsInput(bindingIdx);
		nv::Dims dims{ state.engine->getBindingDimensions(bindingIdx) };
		*numDims = dims.nbDims;
		for (size_t j = 0; j < dims.nbDims; j++) {
			dimsArr[j] = dims.d[j];
		}
		return true;
	}

	UNITY_INTERFACE_EXPORT bool dispose() {
		nv_unique<nv::ICudaEngine> nulleng{ nullptr };
		std::swap(state.engine, nulleng);
		check(cudaFree(state.bindings[0]));
		check(cudaFree(state.bindings[1]));
		auto err = cudaGraphicsUnmapResources(2, state.resources);
		if (err != cudaErrorUnknown) {
			check(err);
		}
		check(cudaDeviceSynchronize());
		return true;
	}
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces *unityInterfaces) {
	unityGraphics = unityInterfaces->Get<IUnityGraphics>();
}


extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload() {
	dispose();
}
