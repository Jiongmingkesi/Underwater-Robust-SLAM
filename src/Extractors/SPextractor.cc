/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/

#include "Matchers/Configuration.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "Extractors/SPextractor.h"
#include "ORBextractor.h"
#include <opencv2/core/eigen.hpp>

#include "Extractors/super_point.h"
#include <utility>
#include <unordered_map>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;


const float factorPI = (float)(CV_PI/180.f);

SPextractor::SPextractor(int _nfeatures, float _scaleFactor, int _nlevels,
         float _iniThFAST, float _minThFAST):
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
   
    if(mModelstr == "onnx")
    {   Configuration cfg;
        cfg.device = "cuda";
        
        cfg.extractorPath = "onnxmodel/superpoint_inuse.onnx";
        cfg.extractorType = "superpoint";
        featureExtractor = new SuperPointOnnxRunner();
        featureExtractor->InitOrtEnv(cfg);

    }

    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }
    for(int i = 0; i < mvLevelSigma2.size(); i++)
    {
        cout<<mvLevelSigma2[i]<<" assd";
    }
    cout<<endl;
    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

}





int SPextractor::operator()( InputArray _image,  vector<KeyPoint>& _keypoints,
                      cv::Mat& _descriptors)
{
    if(_image.empty())
        return 0;

    Mat image = _image.getMat();
    
    assert(image.type() == CV_8UC1 );

    Mat descriptors;
    
    int res = -1;
    if (nlevels == 1) 
        res = ExtractSingleLayer(image, _keypoints, _descriptors);
    else{
        ComputePyramid(image);
        res = ExtractMultiLayers(image, _keypoints, descriptors);
    }
    // Pre-compute the scale pyramid




    // else




    //     // // preprocess the resized image

    //     // // Compute the descriptors


    //     // Scale keypoint coordinates
    //     // And add the keypoints to the output
    return res;
}

int SPextractor::ExtractSingleLayer(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints, cv::Mat &Descriptors)
{
    if(mModelstr == "onnx"){
        Configuration cfg;
        cv::Mat image_copy = image.clone();
        cv::Mat inputImage = NormalizeImage(image_copy);
        featureExtractor->lastmatch = lastmatchnum;
        featureExtractor->Extractor_Inference(cfg , inputImage);
        featureExtractor->Extractor_PostProcess(cfg , std::move(featureExtractor->extractor_outputtensors[0]),vKeyPoints,Descriptors);
    }
    else{
    }





    return vKeyPoints.size();
}

int SPextractor::ExtractMultiLayers(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints, cv::Mat &Descriptors)
{
    ComputePyramid(image);
    int nKeyPoints = 0;
    vector<vector<cv::KeyPoint>> allKeypoints(nlevels);
    vector<cv::Mat> allDescriptors(nlevels);
    for (int level = 0; level < nlevels; ++level)
    {
        if(level == 0)
        {
        }
        else
        {
            
        }
        nKeyPoints +=allKeypoints[level].size();
    }
    vKeyPoints.clear();
    vKeyPoints.reserve(nKeyPoints);
    for (int level = 0; level < nlevels; ++level)
    {
        for(auto keypoint : allKeypoints[level])
        {
            keypoint.octave = level;
            keypoint.pt *= mvScaleFactor[level];
            vKeyPoints.emplace_back(keypoint);
        }
    }
    cv::vconcat(allDescriptors.data(), allDescriptors.size(), Descriptors);
    
    return vKeyPoints.size();
}
//                       OutputArray _descriptors)







void SPextractor::ComputePyramid(cv::Mat image)
{
    std::cout<<" compute pyramid !"<<std::endl;
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        Mat temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if( level != 0 )
        {
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101+BORDER_ISOLATED);            
        }
        else
        {
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);            
        }
    }

}

} //namespace ORB_SLAM
