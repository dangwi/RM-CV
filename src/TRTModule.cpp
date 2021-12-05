#include "TRTModule.hpp"
#include <iostream>
#include "cuda_runtime_api.h"
#include <fstream>


TRTModule::TRTModule(int input_h, int input_w):INPUT_H(input_h),INPUT_W(input_w)
{
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};

    //read trt file
    size_t size{0};
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else{
        std::cerr << "trt file not found!" << std::endl;
        exit(0);
    }

    //build engine
    runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    auto out_dims = engine->getBindingDimensions(1);
    output_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }
    preds = new float[output_size];
    

    inputIndex = engine->getBindingIndex(INPUT_NAME);
    assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    outputIndex = engine->getBindingIndex(OUTPUT_NAME);
    assert(engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);

    // Create GPU buffers on device
    cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&buffers[outputIndex], output_size*sizeof(float));
    cudaStreamCreate(&stream);

    std::cout << "trt init suceess" << std::endl;

}


TRTModule::~TRTModule()
{
    delete preds;
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}


float* TRTModule::operator()(const float* input) const
{
    cudaMemcpyAsync(buffers[inputIndex], input , 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueue(1, buffers, stream, nullptr);
    cudaMemcpyAsync(preds, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return preds;
}


int TRTModule::get_outsize() 
{
    return output_size;
}