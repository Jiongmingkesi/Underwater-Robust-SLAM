#include "Matchers/Configuration.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "Extractors/SPextractor.h"
#include <opencv2/core/eigen.hpp>
#include "Extractors/superpoint_onnx.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;

    SPextractor::SPextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                             float _iniThFAST, float _minThFAST) : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
                                                                   iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {
        // === TensorRT Init ===
        if (mModelstr == "onnx") 
        {
            Configuration cfg;
            cfg.device = "cuda";
            
            cfg.extractorPath = "TensorRT_model/superpoint_400_fp16.engine";
            cfg.extractorType = "superpoint";

            featureExtractor = new SuperPointOnnxRunner();
            featureExtractor->InitOrtEnv(cfg); 
        }

        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0] = 1.0f;
        mvLevelSigma2[0] = 1.0f;

        for (int i = 1; i < nlevels; i++)
        {
            mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
            mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for (int i = 0; i < nlevels; i++)
        {
            mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
            mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
        }

        mvImagePyramid.resize(nlevels);
        mnFeaturesPerLevel.resize(nlevels);
    }

    int SPextractor::operator()(InputArray _image, vector<KeyPoint> &_keypoints,
                                cv::Mat &_descriptors)
    {
        if (_image.empty())
            return 0;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1);

        Mat descriptors;
        int res = -1;
        if (nlevels == 1)
            res = ExtractSingleLayer(image, _keypoints, _descriptors);
        else
        {
            ComputePyramid(image);
            res = ExtractMultiLayers(image, _keypoints, descriptors);
        }
        return res;
    }

    int SPextractor::ExtractSingleLayer(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &Descriptors)
    {
        if (mModelstr == "onnx")
        {
            Configuration cfg;
            cv::Mat image_copy = image.clone();
            cv::Mat inputImage = NormalizeImage(image_copy);

            featureExtractor->lastmatch = lastmatchnum;
            
            featureExtractor->Extractor_Inference(cfg, inputImage);
            
            featureExtractor->Extractor_PostProcess(cfg, vKeyPoints, Descriptors);
        }
        return vKeyPoints.size();
    }

    int SPextractor::ExtractMultiLayers(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &Descriptors)
    {
        
        ComputePyramid(image);
        return 0;
    }

    void SPextractor::ComputePyramid(cv::Mat image)
    {
        for (int level = 0; level < nlevels; ++level)
        {
            float scale = mvInvScaleFactor[level];
            Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
            Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
            Mat temp(wholeSize, image.type());
            mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            if (level != 0)
            {
                resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
                copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101 + BORDER_ISOLATED);
            }
            else
            {
                copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101);
            }
        }
    }

} // namespace ORB_SLAM