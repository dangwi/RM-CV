#ifndef _TRTMODULE_HPP_
#define _TRTMODULE_HPP_

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <sstream>


using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg)throw() override
    {
        //suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};


class TRTModule{
private:
    const int INPUT_W;
    const int INPUT_H;
    const int DEVICE = 0;
    const char* INPUT_NAME = "input_0";
    const char* OUTPUT_NAME = "output_0";
    const std::string engine_file_path = "model_trt.engine";

    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;

    void* buffers[2];
    int inputIndex;
    int outputIndex;
    int output_size=1;

    Logger logger;


public:
    TRTModule(int, int);
    ~TRTModule();
    float* operator()(const float* input) const;
    int get_outsize();

    float* preds;
};

#endif
