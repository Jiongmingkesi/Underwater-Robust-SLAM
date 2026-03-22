#include "Extractors/superpoint_onnx.h"
#include <fstream>
#include <iostream>
#include <numeric>

#define CHECK_CUDA(call)                                                                                    \
    {                                                                                                       \
        cudaError_t err = call;                                                                             \
        if (err != cudaSuccess)                                                                             \
        {                                                                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        }                                                                                                   \
    }


struct KeyPointScore
{
    int index;
    float score;
};

SuperPointOnnxRunner::SuperPointOnnxRunner(unsigned int threads) : num_threads(threads)
{
    stream = nullptr;
    for (int i = 0; i < 4; i++)
        device_buffers[i] = nullptr;
}

SuperPointOnnxRunner::~SuperPointOnnxRunner()
{
    if (stream)
        cudaStreamDestroy(stream);
    for (int i = 0; i < 4; i++)
    {
        if (device_buffers[i])
            cudaFree(device_buffers[i]);
    }
}

int SuperPointOnnxRunner::InitOrtEnv(Configuration cfg)
{
    std::cout << "< - * -------- INITIAL TENSORRT SUPERPOINT START -------- * ->" << std::endl;

    // 1. Create Runtime
    runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        std::cerr << "[ERROR] Failed to create TensorRT Runtime" << std::endl;
        return EXIT_FAILURE;
    }

    // 2. Load Engine File
    
    std::string engine_path = "TensorRT_model/superpoint_400_fp16.engine";
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good())
    {
        std::cerr << "[ERROR] Could not read engine file: " << engine_path << std::endl;
        return EXIT_FAILURE;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> modelStream(size);
    file.read(modelStream.data(), size);
    file.close();

    // 3. Deserialize Engine
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(modelStream.data(), size, nullptr));
    if (!engine)
    {
        std::cerr << "[ERROR] Failed to deserialize engine" << std::endl;
        return EXIT_FAILURE;
    }

    context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
        return EXIT_FAILURE;

    CHECK_CUDA(cudaStreamCreate(&stream));

    // 4. Get Binding Indices (Ensure names match your ONNX export)
    inputIndex = engine->getBindingIndex("image");
    kptsIndex = engine->getBindingIndex("keypoints");
    scoresIndex = engine->getBindingIndex("scores");
    descIndex = engine->getBindingIndex("descriptors");

    // 5. Allocate Device Memory (Max Allocation)
    
    size_t max_pixels = 1920 * 1080;
    int max_kpts = 4096;

    CHECK_CUDA(cudaMalloc(&device_buffers[inputIndex], max_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_buffers[kptsIndex], max_kpts * 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_buffers[scoresIndex], max_kpts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_buffers[descIndex], max_kpts * 256 * sizeof(float)));

    // Resize CPU buffers
    cpu_kpts.resize(max_kpts * 2);
    cpu_scores.resize(max_kpts);
    cpu_desc.resize(max_kpts * 256);

    std::cout << "[INFO] TensorRT SuperPoint initialized successfully." << std::endl;
    return EXIT_SUCCESS;
}

cv::Mat SuperPointOnnxRunner::Extractor_PreProcess(Configuration cfg, const cv::Mat &Image, float &scale)
{
    cv::Mat tempImage = Image.clone();
    cv::Mat resultImage = NormalizeImage(tempImage);
    if (cfg.extractorType == "superpoint")
    {
        resultImage = RGB2Grayscale(resultImage);
    }
    return resultImage;
}

int SuperPointOnnxRunner::Extractor_Inference(Configuration cfg, const cv::Mat &image)
{
    try
    {
        auto time_start = std::chrono::high_resolution_clock::now();

        // 1. Set Input Dimensions (SuperPoint usually static or dynamic H/W)
        context->setBindingDimensions(inputIndex, nvinfer1::Dims4(1, 1, image.rows, image.cols));

        // 2. Copy Input to GPU
        size_t inputSize = 1 * 1 * image.rows * image.cols * sizeof(float);
        CHECK_CUDA(cudaMemcpyAsync(device_buffers[inputIndex], image.ptr<float>(), inputSize, cudaMemcpyHostToDevice, stream));

        // 3. Inference
        void *bindings[4];
        bindings[inputIndex] = device_buffers[inputIndex];
        bindings[kptsIndex] = device_buffers[kptsIndex];
        bindings[scoresIndex] = device_buffers[scoresIndex];
        bindings[descIndex] = device_buffers[descIndex];

        if (!context->enqueueV2(bindings, stream, nullptr))
        {
            std::cerr << "[ERROR] TensorRT Execute Failed!" << std::endl;
            return EXIT_FAILURE;
        }

        // 4. Get Output Dimensions (Number of keypoints detected)
        auto kptsDims = context->getBindingDimensions(kptsIndex);
        int num_points = kptsDims.d[1]; // [1, N, 2]

        // 5. Copy Output back to CPU
        CHECK_CUDA(cudaMemcpyAsync(cpu_kpts.data(), device_buffers[kptsIndex], num_points * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(cpu_scores.data(), device_buffers[scoresIndex], num_points * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(cpu_desc.data(), device_buffers[descIndex], num_points * 256 * sizeof(float), cudaMemcpyDeviceToHost, stream));

        cudaStreamSynchronize(stream);

        auto time_end = std::chrono::high_resolution_clock::now();
        extractor_timer += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] Inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

void SuperPointOnnxRunner::Extractor_PostProcess(
    Configuration cfg,
    std::vector<cv::KeyPoint> &vKeyPoints,
    cv::Mat &Descriptors)
{
    // Retrieve actual number of points from the last inference context
    auto kptsDims = context->getBindingDimensions(kptsIndex);
    int num_raw = kptsDims.d[1];

    // === Top-K Logic (Sort and Select Top 400) ===
    int k = 400;

    // 1. Collect indices and scores
    std::vector<KeyPointScore> candidates;
    candidates.reserve(num_raw);
    for (int i = 0; i < num_raw; ++i)
    {
        // Optional: Filter minimal score if needed, e.g. > 0.005
        candidates.push_back({i, cpu_scores[i]});
    }

    // 2. Sort descending
    std::sort(candidates.begin(), candidates.end(), [](const KeyPointScore &a, const KeyPointScore &b)
              { return a.score > b.score; });

    // 3. Select Top K
    int num_final = std::min((int)candidates.size(), k);

    vKeyPoints.clear();
    vKeyPoints.reserve(num_final);

    if (num_final > 0)
    {
        Descriptors.create(num_final, 256, CV_32F);

        for (int i = 0; i < num_final; ++i)
        {
            int idx = candidates[i].index;

            // Fill KeyPoint
            cv::KeyPoint kp;
            kp.pt.x = cpu_kpts[idx * 2];
            kp.pt.y = cpu_kpts[idx * 2 + 1];
            kp.response = cpu_scores[idx];
            kp.size = 1.0f;
            kp.octave = 0;
            vKeyPoints.push_back(kp);

            // Fill Descriptor (Copy 256 floats)
            std::copy(
                cpu_desc.begin() + idx * 256,
                cpu_desc.begin() + (idx + 1) * 256,
                Descriptors.ptr<float>(i));
        }
    }
    else
    {
        Descriptors = cv::Mat();
    }
}

float SuperPointOnnxRunner::GetMatchThresh() { return this->matchThresh; }
void SuperPointOnnxRunner::SetMatchThresh(float thresh) { this->matchThresh = thresh; }
double SuperPointOnnxRunner::GetTimer(std::string name)
{
    if (name == "extractor")
        return this->extractor_timer;
    return this->matcher_timer;
}
std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> SuperPointOnnxRunner::GetKeypointsResult()
{
    return this->keypoints_result;
}