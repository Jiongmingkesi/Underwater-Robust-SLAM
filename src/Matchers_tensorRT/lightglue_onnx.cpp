#include <thread>
#include <fstream>
#include "Matchers/lightglue_onnx.h"

#define CHECK_CUDA(call)                                                                                    \
    {                                                                                                       \
        cudaError_t err = call;                                                                             \
        if (err != cudaSuccess)                                                                             \
        {                                                                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        }                                                                                                   \
    }

int LightGlueDecoupleOnnxRunner::InitOrtEnv(Configuration cfg)
{
    std::cout << "< - * -------- INITIAL TENSORRT LIGHTGLUE START -------- * ->" << std::endl;

    runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

    // LightGlue Engine Path
    std::string engine_path = "TensorRT_model/superpoint_lightglue_fp16_dynamic.engine";
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good())
    {
        std::cerr << "[ERROR] Could not read LG engine file: " << engine_path << std::endl;
        return EXIT_FAILURE;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> modelStream(size);
    file.read(modelStream.data(), size);
    file.close();

    engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(modelStream.data(), size, nullptr));
    if (!engine)
        return EXIT_FAILURE;

    context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    cudaStreamCreate(&stream);

    idx_kpts0 = engine->getBindingIndex("kpts0");
    idx_kpts1 = engine->getBindingIndex("kpts1");
    idx_desc0 = engine->getBindingIndex("desc0");
    idx_desc1 = engine->getBindingIndex("desc1");
    idx_matches0 = engine->getBindingIndex("matches0");
    idx_mscores0 = engine->getBindingIndex("mscores0");

    std::cout << "[INFO] TensorRT LightGlue initialized." << std::endl;
    return EXIT_SUCCESS;
}

std::vector<cv::Point2f> LightGlueDecoupleOnnxRunner::Matcher_PreProcess(std::vector<cv::Point2f> kpts, int h, int w)
{
    return NormalizeKeypoints(kpts, h, w);
}

std::vector<cv::Point2f> LightGlueDecoupleOnnxRunner::Matcher_PreProcess(std::vector<cv::KeyPoint> kpts, int h, int w)
{
    std::vector<cv::Point2f> pts;
    for (auto &kp : kpts)
        pts.push_back(kp.pt);
    return NormalizeKeypoints(pts, h, w);
}

void LightGlueDecoupleOnnxRunner::Matcher_Inference(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, float *desc0, float *desc1)
{
    int N = kpts0.size();
    int M = kpts1.size();

    
    cpu_matches0.clear();
    cpu_mscores0.clear();
    if (N == 0 || M == 0)
        return;

    auto time_start = std::chrono::high_resolution_clock::now();

    
    context->setBindingDimensions(idx_kpts0, nvinfer1::Dims3(1, N, 2));
    context->setBindingDimensions(idx_kpts1, nvinfer1::Dims3(1, M, 2));
    context->setBindingDimensions(idx_desc0, nvinfer1::Dims3(1, N, 256));
    context->setBindingDimensions(idx_desc1, nvinfer1::Dims3(1, M, 256));

    
    
    size_t max_elems = std::max(N, M) * 2;

    
    nvinfer1::DataType match_type = engine->getBindingDataType(idx_matches0);
    size_t match_byte_size = (match_type == nvinfer1::DataType::kINT32) ? 4 : 8;

    CHECK_CUDA(cudaMalloc(&d_kpts0, N * 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kpts1, M * 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_desc0, N * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_desc1, M * 256 * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_matches0, max_elems * match_byte_size));
    CHECK_CUDA(cudaMalloc(&d_mscores0, max_elems * sizeof(float)));

    
    float *h_kpts0 = new float[N * 2];
    float *h_kpts1 = new float[M * 2];
    for (int i = 0; i < N; i++)
    {
        h_kpts0[2 * i] = kpts0[i].x;
        h_kpts0[2 * i + 1] = kpts0[i].y;
    }
    for (int i = 0; i < M; i++)
    {
        h_kpts1[2 * i] = kpts1[i].x;
        h_kpts1[2 * i + 1] = kpts1[i].y;
    }

    CHECK_CUDA(cudaMemcpyAsync(d_kpts0, h_kpts0, N * 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_kpts1, h_kpts1, M * 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_desc0, desc0, N * 256 * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_desc1, desc1, M * 256 * sizeof(float), cudaMemcpyHostToDevice, stream));

    
    void *bindings[6];
    bindings[idx_kpts0] = d_kpts0;
    bindings[idx_kpts1] = d_kpts1;
    bindings[idx_desc0] = d_desc0;
    bindings[idx_desc1] = d_desc1;
    bindings[idx_matches0] = d_matches0;
    bindings[idx_mscores0] = d_mscores0;

    if (!context->enqueueV2(bindings, stream, nullptr))
    {
        std::cerr << "[ERROR] TensorRT enqueueV2 failed!" << std::endl;
    }

    
    nvinfer1::Dims match_dims = context->getBindingDimensions(idx_matches0);
    nvinfer1::Dims score_dims = context->getBindingDimensions(idx_mscores0);

    
    bool is_sparse = false;
    int num_entries = 0; 

    
    if (match_dims.nbDims >= 1 && match_dims.d[match_dims.nbDims - 1] == 2)
    {
        is_sparse = true;
        
        if (match_dims.nbDims >= 2)
            num_entries = match_dims.d[match_dims.nbDims - 2];
        else
            num_entries = match_dims.d[0] / 2;
    }
    else
    {
        
        is_sparse = false;
        
        if (match_dims.nbDims >= 1 && match_dims.d[match_dims.nbDims - 1] > 0)
            num_entries = match_dims.d[match_dims.nbDims - 1];
        else
            num_entries = N; 
    }

    
    static bool first_run = true;
    if (first_run)
    {
        std::cout << "[DEBUG TRT] Output Format: " << (is_sparse ? "SPARSE [K, 2]" : "DENSE [N]")
                  << " | DataType: " << (match_type == nvinfer1::DataType::kINT32 ? "INT32" : "INT64")
                  << " | NumEntries: " << num_entries << std::endl;
        first_run = false;
    }

    if (num_entries > 0)
    {
        
        std::vector<float> temp_scores(num_entries);
        CHECK_CUDA(cudaMemcpyAsync(temp_scores.data(), d_mscores0, num_entries * sizeof(float), cudaMemcpyDeviceToHost, stream));

        
        int match_copy_count = is_sparse ? num_entries * 2 : num_entries;
        std::vector<int64_t> temp_matches_64(match_copy_count);

        if (match_type == nvinfer1::DataType::kINT32)
        {
            std::vector<int32_t> temp_matches_32(match_copy_count);
            CHECK_CUDA(cudaMemcpyAsync(temp_matches_32.data(), d_matches0, match_copy_count * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);
            for (int i = 0; i < match_copy_count; i++)
                temp_matches_64[i] = static_cast<int64_t>(temp_matches_32[i]);
        }
        else
        {
            CHECK_CUDA(cudaMemcpyAsync(temp_matches_64.data(), d_matches0, match_copy_count * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);
        }

        
        if (is_sparse)
        {
            
            cpu_matches0 = temp_matches_64;
            cpu_mscores0 = temp_scores;
        }
        else
        {
            
            
            cpu_matches0.reserve(num_entries * 2);
            cpu_mscores0.reserve(num_entries);

            for (int i = 0; i < num_entries; i++)
            {
                int64_t match_idx = temp_matches_64[i];
                
                
                if (match_idx > -1)
                {
                    cpu_matches0.push_back(i);         // queryIdx
                    cpu_matches0.push_back(match_idx); // trainIdx
                    cpu_mscores0.push_back(temp_scores[i]);
                }
            }
        }
    }
    else
    {
        cudaStreamSynchronize(stream);
    }

    
    cudaFree(d_kpts0);
    cudaFree(d_kpts1);
    cudaFree(d_desc0);
    cudaFree(d_desc1);
    cudaFree(d_matches0);
    cudaFree(d_mscores0);
    delete[] h_kpts0;
    delete[] h_kpts1;

    auto time_end = std::chrono::high_resolution_clock::now();
    matcher_timer += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
}

int LightGlueDecoupleOnnxRunner::Matcher_PostProcess_fused(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, std::vector<int> &vnMatches12)
{
    int count = 0;
    int num_pairs = cpu_mscores0.size(); 

    
    if (cpu_matches0.size() < num_pairs * 2)
        return 0;

    
    std::fill(vnMatches12.begin(), vnMatches12.end(), -1);

    for (int i = 0; i < num_pairs; i++)
    {
        int64_t queryIdx = cpu_matches0[i * 2];
        int64_t trainIdx = cpu_matches0[i * 2 + 1];
        float score = cpu_mscores0[i];

        if (score > this->matchThresh)
        {
            if (queryIdx >= 0 && queryIdx < (int64_t)kpts0.size() &&
                trainIdx >= 0 && trainIdx < (int64_t)kpts1.size())
            {
                vnMatches12[queryIdx] = static_cast<int>(trainIdx);
                count++;
            }
        }
    }
    return count;
}

float LightGlueDecoupleOnnxRunner::GetMatchThresh() { return this->matchThresh; }
void LightGlueDecoupleOnnxRunner::SetMatchThresh(float thresh) { this->matchThresh = thresh; }
double LightGlueDecoupleOnnxRunner::GetTimer(std::string name) { return matcher_timer; }
std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueDecoupleOnnxRunner::GetKeypointsResult() { return keypoints_result; }

LightGlueDecoupleOnnxRunner::LightGlueDecoupleOnnxRunner(unsigned int threads) : num_threads(threads) {}
LightGlueDecoupleOnnxRunner::~LightGlueDecoupleOnnxRunner()
{
    if (stream)
        cudaStreamDestroy(stream);
}