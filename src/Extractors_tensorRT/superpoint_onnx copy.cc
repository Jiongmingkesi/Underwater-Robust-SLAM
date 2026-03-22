#include "Extractors/superpoint_onnx.h"
#include <fstream>
#include <iostream>
#include <numeric>

// Helper to check CUDA errors
#define CHECK_CUDA(call)                                                                                    \
    {                                                                                                       \
        cudaError_t err = call;                                                                             \
        if (err != cudaSuccess)                                                                             \
        {                                                                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                            \
        }                                                                                                   \
    }

// Struct for sorting keypoints
struct KeyPointScore
{
    int index;
    float score;
};

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
    std::string engine_path = cfg.extractorPath; // Should be TensorRT_model/superpoint_400_fp16.engine
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

    // 4. Create Context
    context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        std::cerr << "[ERROR] Failed to create execution context" << std::endl;
        return EXIT_FAILURE;
    }

    // 5. Create CUDA Stream
    cudaStreamCreate(&stream);

    // 6. Allocate Device Buffers
    // Assumption: Engine bindings order: "image", "keypoints", "scores", "descriptors"
    // Use names to be safe
    inputIndex = engine->getBindingIndex("image");
    kptsIndex = engine->getBindingIndex("keypoints");
    scoresIndex = engine->getBindingIndex("scores");
    descIndex = engine->getBindingIndex("descriptors");

    // Input Dimensions (Fixed 1x1x500x800 based on file usage or dynamic?)
    // Assuming fixed max size for allocation, usually defined in creating engine.
    // If the engine is dynamic, we must set binding dimensions.
    // Let's assume input is 1x1x500x800 float32
    size_t inputSize = 1 * 1 * 500 * 800 * sizeof(float);

    // Output Dimensions: Engine usually outputs max number of keypoints allowed by graph
    // E.g. [1, MaxN, 2]. We will filter to 400 later.
    // We allocate enough for what the engine *might* output (e.g. 1024 or more).
    // Safe max allocation.
    int max_possible_kpts = 2048; // Safe upper bound for allocation

    // Check actual dims if possible (only works for static dims)
    auto out_dims = engine->getBindingDimensions(kptsIndex);
    if (out_dims.d[1] > 0)
        max_possible_kpts = out_dims.d[1];

    size_t kptsSize = max_possible_kpts * 2 * sizeof(float);
    size_t scoresSize = max_possible_kpts * sizeof(float);
    size_t descSize = max_possible_kpts * 256 * sizeof(float);

    CHECK_CUDA(cudaMalloc(&device_buffers[inputIndex], inputSize));
    CHECK_CUDA(cudaMalloc(&device_buffers[kptsIndex], kptsSize));
    CHECK_CUDA(cudaMalloc(&device_buffers[scoresIndex], scoresSize));
    CHECK_CUDA(cudaMalloc(&device_buffers[descIndex], descSize));

    // Prepare CPU buffers
    cpu_kpts.resize(max_possible_kpts * 2);
    cpu_scores.resize(max_possible_kpts);
    cpu_desc.resize(max_possible_kpts * 256);

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
        // Set Binding Dimensions if dynamic (assuming fixed 500x800 for SP usually,
        // but explicit setting is safer for dynamic profiles)
        nvinfer1::Dims4 inputDims(1, 1, image.rows, image.cols);
        context->setBindingDimensions(inputIndex, inputDims);

        size_t inputSize = 1 * 1 * image.rows * image.cols * sizeof(float);

        // Copy Input to Device
        CHECK_CUDA(cudaMemcpyAsync(device_buffers[inputIndex], image.ptr<float>(), inputSize, cudaMemcpyHostToDevice, stream));

        // Inference
        // Since bindings are void* array, we need to arrange them correctly
        void *bindings[4];
        bindings[inputIndex] = device_buffers[inputIndex];
        bindings[kptsIndex] = device_buffers[kptsIndex];
        bindings[scoresIndex] = device_buffers[scoresIndex];
        bindings[descIndex] = device_buffers[descIndex];

        bool status = context->enqueueV2(bindings, stream, nullptr);
        if (!status)
        {
            std::cerr << "[ERROR] TensorRT execution failed." << std::endl;
            return EXIT_FAILURE;
        }

        // Get Output Dimensions to know how much to copy
        // For SuperPoint with NonMaxSuppression inside, the output dimension might be dynamic or fixed padded.
        // Assuming output 0 dimension is batch, 1 is num_points.
        auto kptsDims = context->getBindingDimensions(kptsIndex);
        int num_points = kptsDims.d[1];

        // Copy Outputs to Host
        CHECK_CUDA(cudaMemcpyAsync(cpu_kpts.data(), device_buffers[kptsIndex], num_points * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(cpu_scores.data(), device_buffers[scoresIndex], num_points * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaMemcpyAsync(cpu_desc.data(), device_buffers[descIndex], num_points * 256 * sizeof(float), cudaMemcpyDeviceToHost, stream));

        cudaStreamSynchronize(stream);

        auto time_end = std::chrono::high_resolution_clock::now();
        // Timing logic if needed
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] TRT Inference failed: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

void SuperPointOnnxRunner::Extractor_PostProcess(
    Configuration cfg,
    std::vector<cv::KeyPoint> &vKeyPoints,
    cv::Mat &Descriptors)
{
    // Retrieve dimensions from the last inference context (assuming single threaded runner usage)
    auto kptsDims = context->getBindingDimensions(kptsIndex);
    int num_raw_points = kptsDims.d[1];

    // 1. Filter and Sort (Top-K = 400)
    int k = 400; // Target number
    std::vector<KeyPointScore> valid_indices;
    valid_indices.reserve(num_raw_points);

    // Collect valid indices (scores > threshold)
    float threshold = 0.005f; // Basic threshold
    for (int i = 0; i < num_raw_points; i++)
    {
        if (cpu_scores[i] > threshold)
        {
            valid_indices.push_back({i, cpu_scores[i]});
        }
    }

    // Sort descending by score
    std::sort(valid_indices.begin(), valid_indices.end(),
              [](const KeyPointScore &a, const KeyPointScore &b)
              {
                  return a.score > b.score;
              });

    // Clamp to K
    int num_final = std::min((int)valid_indices.size(), k);

    // 2. Fill Outputs
    vKeyPoints.clear();
    vKeyPoints.reserve(num_final);

    Descriptors.create(num_final, 256, CV_32F);

    for (int i = 0; i < num_final; i++)
    {
        int idx = valid_indices[i].index;

        // Construct KeyPoint
        cv::KeyPoint kp;
        kp.pt.x = cpu_kpts[idx * 2];
        kp.pt.y = cpu_kpts[idx * 2 + 1];
        kp.response = cpu_scores[idx];
        kp.size = 1.0f; // Standard size
        kp.octave = 0;
        vKeyPoints.push_back(kp);

        // Construct Descriptor
        // Copy 256 floats
        std::copy(
            cpu_desc.begin() + idx * 256,
            cpu_desc.begin() + (idx + 1) * 256,
            Descriptors.ptr<float>(i));
    }
}

SuperPointOnnxRunner::SuperPointOnnxRunner(unsigned int threads) : num_threads(threads) {}

SuperPointOnnxRunner::~SuperPointOnnxRunner()
{
    cudaStreamDestroy(stream);
    cudaFree(device_buffers[inputIndex]);
    cudaFree(device_buffers[kptsIndex]);
    cudaFree(device_buffers[scoresIndex]);
    cudaFree(device_buffers[descIndex]);
}