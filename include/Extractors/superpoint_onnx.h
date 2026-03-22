#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "Matchers/transform.h"
#include "Matchers/Configuration.h"


class SuperPointOnnxRunner
{
public:
	const unsigned int num_threads;

    Ort::Env env0;
    Ort::SessionOptions session_options0;
    Ort::Session* ExtractorSession;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<char*> ExtractorInputNodeNames;
    std::vector<std::vector<int64_t>> ExtractorInputNodeShapes;
    std::vector<char*> ExtractorOutputNodeNames;
    std::vector<std::vector<int64_t>> ExtractorOutputNodeShapes;


    float matchThresh = 0.0f;
    long long extractor_timer = 0.0f;
    long long matcher_timer = 0.0f;

    float lastmatch = 0;
    std::vector<float> scales = {1.0f , 1.0f};

    std::vector<std::vector<Ort::Value>> extractor_outputtensors; 


    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> keypoints_result;
    
public:
    cv::Mat Extractor_PreProcess(Configuration cfg , const cv::Mat& srcImage , float& scale);
    int Extractor_Inference(Configuration cfg , const cv::Mat& image);
    void Extractor_PostProcess(Configuration cfg , std::vector<Ort::Value> tensor, std::vector<cv::KeyPoint>& vKeyPoints, cv::Mat &Descriptors);

public:
    explicit SuperPointOnnxRunner(unsigned int num_threads = 1);
    ~SuperPointOnnxRunner();

    float GetMatchThresh();
    void SetMatchThresh(float thresh);
    double GetTimer(std::string name);

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> GetKeypointsResult();

    int InitOrtEnv(Configuration cfg);
    
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> InferenceImage(Configuration cfg , \
            const cv::Mat& srcImage, const cv::Mat& destImage);
};
