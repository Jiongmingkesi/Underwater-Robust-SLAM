
#include "iostream"
#include <opencv2/core/core.hpp>
#include <onnxruntime_cxx_api.h>
#include "Matchers/Configuration.h"
#include "Matchers/SPmatcher.h"
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
using namespace std;

namespace ORB_SLAM3
{
    const float SPmatcher::TH_HIGH = 1.4;
    const float SPmatcher::TH_LOW = 1.2;
    const int SPmatcher::HISTO_LENGTH = 30;

SPmatcher::SPmatcher(float thre)
{
    Configuration cfg;
    cfg.device = "cuda";
    cfg.extractorPath = "";
    cfg.extractorType = "";
    featureMatcher = new LightGlueDecoupleOnnxRunner();
    featureMatcher->InitOrtEnv(cfg);
    featureMatcher->SetMatchThresh(thre);
    std::string mode = "LightGlueDecoupleOnnxRunner";
}

void SPmatcher::plotspmatch(cv::Mat frame1,cv::Mat frame2, std::vector<cv::KeyPoint> kpts1, std::vector<cv::KeyPoint> kpts2,  std::vector<int> vmatches12){
    vector<cv::DMatch> vmatches;
    for(int i=0 ; i < vmatches12.size(); i++)
    {
        int idx = vmatches12[i];
        if(idx > 0)
        {
            cv::DMatch match;
            match.queryIdx = i;
            match.trainIdx = idx;
            vmatches.push_back(match);
        }
    }
    cv::Mat matched_img;
    cv::drawMatches(frame1,kpts1,frame2,kpts2,vmatches,matched_img, cv::Scalar(0, 255, 0));
    cv::imshow("Matched Features", matched_img);
    cv::waitKey(0);

}

int SPmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
{
    GeometricCamera* pCamera;
    Sophus::SE3f Tcw;
    Eigen::Vector3f Ow;

    if(bRight){
        Tcw = pKF->GetRightPose();
        Ow = pKF->GetRightCameraCenter();
        pCamera = pKF->mpCamera2;
    }
    else{
        Tcw = pKF->GetPose();
        Ow = pKF->GetCameraCenter();
        pCamera = pKF->mpCamera;
    }

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    int nFused = 0;
    const int nMPs = vpMapPoints.size();
    int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;
    for(int i = 0; i < nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
        {
            count_notMP++;
            continue;
        }
        if(pMP->isBad())
        {
            count_bad++;
            continue;
        }
        else if(pMP->IsInKeyFrame(pKF))
        {
            count_isinKF++;
            continue;
        }

        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        Eigen::Vector3f p3Dc = Tcw*p3Dw;

        if(p3Dc(2)<0.0f)
        {
            count_negdepth++;
            continue;
        }
        const float invz = 1/p3Dc(2);

        const Eigen::Vector2f uv = pCamera->project(p3Dc);

        if(!pKF->IsInImage(uv(0),uv(1)))
        {
            count_notinim++;
            continue;
        }

        const float ur = uv(0) - bf*invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw - Ow;
        const float dist3D = PO.norm();

        if(dist3D < minDistance || dist3D > maxDistance)
        {
            count_dist++;
            continue;
        }

        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn) < 0.5*dist3D)
        {
            count_normal++;
            continue;
        }

        int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius,bRight);

        if(vIndices.empty())
        {
            count_notidx++;
            continue;
        }

        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = 10;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
        {
            size_t idx = *vit;
            const cv::KeyPoint &kp = (pKF -> NLeft == -1) ? pKF->mvKeysUn[idx]
                                                            : (!bRight) ? pKF -> mvKeys[idx]
                                                                        : pKF -> mvKeysRight[idx];
            const int &kpLevel = kp.octave;
            if(kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;
            
            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = uv(0)-kpx;
                const float ey = uv(1)-kpy;
                
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                
                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = uv(0) - kpx;
                const float ey = uv(1) - kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            if(bRight) idx += pKF->NLeft;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const float dist = DescriptorDistance_sp(dMP, dKF);

            if(dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist <= TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                {
                    if(pMPinKF->Observations() > pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
        else
            count_thcheck++;

    }

    return nFused;

}

int SPmatcher::Fuse(KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
 {
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
    Eigen::Vector3f Ow = Tcw.inverse().translation();

    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;
    
    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        Eigen::Vector3f p3Dc = Tcw * p3Dw;

        // Depth must be positive
        if(p3Dc(2)<0.0f)
            continue;

        // Project into Image
        
        const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

        // Point must be inside the image
        
        if(!pKF->IsInImage(uv(0),uv(1)))
            continue;

        // Depth must be inside the scale pyramid of the image
        
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        Eigen::Vector3f PO = p3Dw-Ow;
        const float dist3D = PO.norm();

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        Eigen::Vector3f Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;


            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance_sp(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                
                
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }
    
    return nFused;
 }

int SPmatcher::MatchingPoints_onnx(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, float* desc0,float* desc1){
    int rows = 500;
    int cols = 800;
    auto normal_kpts0 = featureMatcher->Matcher_PreProcess(kpts0 , rows , cols);
    auto normal_kpts1 = featureMatcher->Matcher_PreProcess(kpts1 , rows , cols);
    Configuration cfg;
    std::vector<Ort::Value> output = featureMatcher->Matcher_Inference(normal_kpts0, normal_kpts1, desc0, desc1);
    int size;
    std::vector<int> vnMatches12;
    vnMatches12 = std::vector<int>(normal_kpts0.size(), -1);
    size = featureMatcher->Matcher_PostProcess_fused(output, kpts0 , kpts1, vnMatches12);
    return size;
}

int SPmatcher::MatchingPoints_onnx(std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, cv::Mat desc0,cv::Mat desc1, std::vector<int>& vnMatches12){
    vnMatches12.resize(kpts0.size(),-1);
    int rows = 500;
    int cols = 800;

    auto normal_kpts0 = featureMatcher->Matcher_PreProcess(kpts0 , rows , cols);
    auto normal_kpts1 = featureMatcher->Matcher_PreProcess(kpts1 , rows , cols);

    int rows0 = desc0.rows;
    int cols0 = desc0.cols;
    float* descriptors_data0 = new float[rows0 * cols0];
    for (int i = 0 ; i < rows0; i++){
        const float* row_data = desc0.ptr<float>(i);
        for (int j = 0; j < cols0; j++){
            descriptors_data0[i*cols0 + j] = row_data[j];
        }
    }

    int rows1 = desc1.rows;
    int cols1 = desc1.cols;
    float* descriptors_data1 = new float[rows1 * cols1];
    for (int i = 0 ; i < rows1; i++){
        const float* row_data = desc1.ptr<float>(i);
        for (int j = 0; j < cols1; j++){
            descriptors_data1[i*cols1 + j] = row_data[j];
        }
    }

    Configuration cfg;
    std::vector<Ort::Value> output = featureMatcher->Matcher_Inference(normal_kpts0, normal_kpts1, descriptors_data0, descriptors_data1);
    int size;
    size = featureMatcher->Matcher_PostProcess_fused(output, kpts0 , kpts1, vnMatches12);
    return size;
}

int SPmatcher::MatchingPoints_onnx(std::vector<cv::KeyPoint> kpts0, const std::vector<cv::KeyPoint> kpts1, cv::Mat desc0, const cv::Mat desc1, std::vector<int>& vnMatches12){
    vnMatches12.resize(kpts0.size(),-1);
    int rows = 500;
    int cols = 800;
    std::vector<cv::Point2f> kpts_pf0, kpts_pf1;
    for(const cv::KeyPoint& keypoint : kpts0){
        kpts_pf0.emplace_back(keypoint.pt);
    }

    for(const cv::KeyPoint& keypoint : kpts1){
        kpts_pf1.emplace_back(keypoint.pt);
    }
    auto normal_kpts0 = featureMatcher->Matcher_PreProcess(kpts0 , rows , cols);
    auto normal_kpts1 = featureMatcher->Matcher_PreProcess(kpts1 , rows , cols);

    int rows0 = desc0.rows;
    int cols0 = desc0.cols;
    float* descriptors_data0 = new float[rows0 * cols0];
    for (int i = 0 ; i < rows0; i++){
        const float* row_data = desc0.ptr<float>(i);
        for (int j = 0; j < cols0; j++){
            descriptors_data0[i*cols0 + j] = row_data[j];
        }
    }

    int rows1 = desc1.rows;
    int cols1 = desc1.cols;
    float* descriptors_data1 = new float[rows1 * cols1];
    for (int i = 0 ; i < rows1; i++){
        const float* row_data = desc1.ptr<float>(i);
        for (int j = 0; j < cols1; j++){
            descriptors_data1[i*cols1 + j] = row_data[j];
        }
    }

    Configuration cfg;
    std::vector<Ort::Value> output = featureMatcher->Matcher_Inference(normal_kpts0, normal_kpts1, descriptors_data0, descriptors_data1);
    int size  = featureMatcher->Matcher_PostProcess_fused(output, kpts_pf0 , kpts_pf1, vnMatches12);
    return size;
}


int SPmatcher::MatchingPoints_onnx(Frame &f1, Frame &f2, vector<int> &vnMatches12)
{   
    bool outlier_rejection=false;
    vnMatches12.resize(f1.mvKeys.size(),-1);
    
    
    int rows = f2.imgLeft.rows;
    int cols = f2.imgLeft.cols;
    std::vector<cv::Point2f> kpts1, kpts2;

    for(const cv::KeyPoint& keypoint : f1.mvKeys){
        kpts1.emplace_back(keypoint.pt);
    }

    for(const cv::KeyPoint& keypoint : f2.mvKeys){
        kpts2.emplace_back(keypoint.pt);
    }



    auto normal_kpts1 = featureMatcher->Matcher_PreProcess(kpts1 , rows , cols);
    auto normal_kpts2 = featureMatcher->Matcher_PreProcess(kpts2 , rows , cols);

    
    int rows1 = f1.mDescriptors.rows;
    int cols1 = f1.mDescriptors.cols;
    float* descriptors_data1 = new float[rows1 * cols1];
    for (int i = 0 ; i < rows1; i++){
        const float* row_data = f1.mDescriptors.ptr<float>(i);
        for (int j = 0; j < cols1; j++){
            descriptors_data1[i*cols1 + j] = row_data[j];
        }
    }


    int rows2 = f2.mDescriptors.rows;
    int cols2 = f2.mDescriptors.cols;
    float* descriptors_data2 = new float[rows2 * cols2];
    for (int i = 0 ; i < rows2; i++){
        const float* row_data = f2.mDescriptors.ptr<float>(i);
        for (int j = 0; j < cols2; j++){
            descriptors_data2[i*cols2 + j] = row_data[j];
        }
    }
    std::vector<Ort::Value> output = featureMatcher->Matcher_Inference(normal_kpts1, normal_kpts2, descriptors_data1, descriptors_data2);
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> output_end;
    int size  = featureMatcher->Matcher_PostProcess_fused(output, kpts1 , kpts2, vnMatches12);

    
    std::cout << "[SPmatcher] ONNX Matches: " << size << std::endl;

    return size;
}




        






    
    

Eigen::Matrix<double, 259, Eigen::Dynamic> SPmatcher::NormalizeKeypoints(const Eigen::Matrix<double, 259, Eigen::Dynamic> &features, int width, int height)
{
    Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features;
    norm_features.resize(259, features.cols());
    norm_features = features;
    for (int col = 0; col <features.cols(); ++col) {
        norm_features(1, col) = 
            (features(1, col) - width / 2) / (std::max(width, height) * 0.7);
        norm_features(2, col) = 
            (features(2, col) - height / 2) /(std::max(width, height) * 0.7);
    }
    return norm_features;
}

void SPmatcher::ConvertMatchesToVector(const std::vector<cv::DMatch>& matches, std::vector<int>& vnMatches12){
            
            vnMatches12 = std::vector<int>(matches.size(), -1);
            
            
            for (size_t i = 0; i < matches.size(); ++i){
                int idxF1 = matches[i].queryIdx;
                int idxF2 = matches[i].trainIdx;

                vnMatches12[idxF1] = idxF2;
            }
}

Eigen::Matrix<double, 259, Eigen::Dynamic> SPmatcher::ConvertToEigenMatrix(const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors)
{
    int numPoints = static_cast<int>(keypoints.size());
    Eigen::Matrix<double, 259, Eigen::Dynamic> pointsAndDescriptors(259, numPoints);

    for(int i = 0; i < numPoints; ++i)
    {
        pointsAndDescriptors(0, i) = keypoints[i].pt.x;
        pointsAndDescriptors(1, i) = keypoints[i].pt.y;
        pointsAndDescriptors(2, i) = keypoints[i].response;

        cv::Mat descriptor = descriptors.row(i);
        for(int j = 0; j < descriptors.cols; ++j){
            pointsAndDescriptors(3 + j, i) = static_cast<double>(descriptor.at<uchar>(0, j));
        }
    }
    return pointsAndDescriptors;
}

int SPmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12)
{
    int nmatches = 0;
    vnMatches12 = vector<int>(F1.mvKeys.size(), -1);
    
    vector<int> vMatchedDistance(F2.mvKeys.size(), INT_MAX);
    vector<int> vnMatches21(F2.mvKeys.size(), -1);

    nmatches =  MatchingPoints_onnx(F1, F2, vnMatches12);
    
    return nmatches;
    
}

int SPmatcher::SearchByProjection(Frame &CurrentFrame, Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;
    const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    const Eigen::Vector3f twc = Tcw.inverse().translation();
    
    vector<int> vmatches(CurrentFrame.N,-1);

    const Sophus::SE3f Tlw = LastFrame.GetPose();
    const Eigen::Vector3f tlc = Tlw * twc; 
    
    
    const bool bForward = tlc(2)>CurrentFrame.mb && !bMono;     
    const bool bBackward = -tlc(2)>CurrentFrame.mb && !bMono;   

    for(int i=0; i<LastFrame.N; i++)
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

        if(pMP)
        {
            if(!LastFrame.mvbOutlier[i])
            {
                    
                Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                Eigen::Vector3f x3Dc = Tcw * x3Dw;

                const float xc = x3Dc(0);
                const float yc = x3Dc(1);
                const float invzc = 1.0/x3Dc(2);

                if(invzc<0)
                    continue;

                    
                Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                if(uv(0)<CurrentFrame.mnMinX || uv(0)>CurrentFrame.mnMaxX)
                    continue;
                if(uv(1)<CurrentFrame.mnMinY || uv(1)>CurrentFrame.mnMaxY)
                    continue;
                    
                int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                    : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                    // Search in a window. Size depends on scale
                    
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave]; 

                    
                vector<size_t> vIndices2;

                    
                    
                    
                    
                if(bForward)  
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave);
                else if(bBackward)  
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave);
                else  
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                float bestDist = 256;
                int bestIdx2 = -1;

                    
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;

                        
                    if(CurrentFrame.mvpMapPoints[i2])
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

                    if(CurrentFrame.Nleft == -1 && CurrentFrame.mvuRight[i2]>0)
                    {
                            
                        const float ur = uv(0) - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const float dist = DescriptorDistance_sp(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                    
                if(bestDist<=TH_HIGH)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    vmatches[bestIdx2] = i;
                    nmatches++;

                        
                }
                if(CurrentFrame.Nleft != -1){
                    Eigen::Vector3f x3Dr = CurrentFrame.GetRelativePoseTrl() * x3Dc;
                    Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dr);

                    int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                        : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                        // Search in a window. Size depends on scale
                    float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                    vector<size_t> vIndices2;

                    if(bForward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave, -1,true);
                    else if(bBackward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave, true);
                    else
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave-1, nLastOctave+1, true);

                    const cv::Mat dMP = pMP->GetDescriptor();

                    float bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                    {
                        const size_t i2 = *vit;
                        if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft])
                            if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft]->Observations()>0)
                                continue;

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2 + CurrentFrame.Nleft);

                        const float dist = DescriptorDistance_sp(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=TH_HIGH)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2 + CurrentFrame.Nleft]=pMP;
                        nmatches++;
                    }
                }
            }
        }
    }

    return nmatches;

}

int SPmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist)
{
    int nmatches = 0;
    std::vector<int> vnMatches12;
    nmatches = MatchingPoints_onnx(CurrentFrame.mvKeys, pKF->mvKeys, CurrentFrame.mDescriptors, pKF->mDescriptors, vnMatches12);
    const Sophus::SE3f Tcw = CurrentFrame.GetPose();
    Eigen::Vector3f Ow = Tcw.inverse().translation();
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
    for(size_t i = 0, iend = vnMatches12.size(); i<iend; i++)
    {   
        MapPoint* pMP = vpMPs[vnMatches12[i]];
        if(!pMP||pMP->isBad()||sAlreadyFound.count(pMP))
        {
            nmatches--;
            continue;
        }
        if(!CurrentFrame.mvpMapPoints[i]){
            CurrentFrame.mvpMapPoints[i]=pMP;
        }
    }

    
    //                 //Project



    //                 // Compute predicted scale level


    //                 // Depth must be inside the scale pyramid of the image

    

    //                 // Search in a window
    

    



    





        return nmatches;
    
}

int SPmatcher::SearchBySP(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches)
{
    const std::vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    vpMapPointMatches = std::vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

    std::vector<cv::DMatch> matches;
    vector<int> vnMatches12;
    MatchingPoints_onnx(F,F, vnMatches12);
    int nmatches = 0;
    for(int i = 0; i < static_cast<int>(matches.size()); ++i)
    {
        int realIdxKF = matches[i].queryIdx;
        int bestIdxF = matches[i].trainIdx;

        if(matches[i].distance > TH_HIGH)
            continue;
        
        MapPoint* pMP = vpMapPointsKF[realIdxKF];
        if(!pMP)
            continue;
        if(pMP->isBad())
            continue;
        vpMapPointMatches[bestIdxF] = pMP;
        nmatches++;
    }

    return nmatches;
}

int SPmatcher::SearchBySP(Frame &F, const std::vector<MapPoint*> &vpMapPoints)
{
    std::cout << vpMapPoints.size() <<std::endl;
    std::cout << F.mDescriptors.rows << std::endl;

    std::vector<cv::Mat> MPdescriptorAll;
    std::vector<int> select_indice;
    for(size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];

        if(!pMP)
            continue;
        if(!pMP->mbTrackInView)
            continue;
        if(pMP->isBad())
            continue;
        const cv::Mat MPdescriptor = pMP->GetDescriptor();
        MPdescriptorAll.push_back(MPdescriptor);
        select_indice.push_back(iMP);
    }
    cv::Mat MPdescriptors;
    MPdescriptors.create(MPdescriptorAll.size(), 32, CV_8U);

    for (int i=0; i<static_cast<int>(MPdescriptorAll.size()); i++)
    {
        for(int j=0; j<32; j++)
        {
            MPdescriptors.at<unsigned char>(i, j) = MPdescriptorAll[i].at<unsigned char>(j);
        }
    }

    std::vector<cv::DMatch> matches;
    cv::BFMatcher desc_matcher(cv::NORM_HAMMING, true);
    desc_matcher.match(MPdescriptors, F.mDescriptors, matches, cv::Mat());

    int nmatches = 0;
    for(int i = 0; i < static_cast<int>(matches.size()); ++i)
    {
        int realIdxMap = select_indice[matches[i].queryIdx];
        int bestIdxF = matches[i].trainIdx;

        if(matches[i].distance > TH_HIGH)
            continue;
        if(F.mvpMapPoints[bestIdxF])
            if(F.mvpMapPoints[bestIdxF]->Observations()>0)
                continue;
        
        MapPoint* pMP = vpMapPoints[realIdxMap];
        F.mvpMapPoints[bestIdxF] = pMP;
        nmatches++;
    }
}

int SPmatcher::SearchBySP(Frame &CurrentFrame, Frame &LastFrame)
{
    vector<cv::Point2f> vbPrevMatched;
    vector<int> vnMatches1;
    int size = MatchingPoints_onnx(CurrentFrame, LastFrame, vnMatches1);
    int nmatches = 0;
    for(int i = 0; i < static_cast<int>(vnMatches1.size());++i)
    {
        int IdxLF = vnMatches1[i];
        int IdxCF = i;
        if(IdxLF != -1){
            MapPoint* pMP = LastFrame.mvpMapPoints[IdxLF];
            if(!pMP)
                continue;
            if(pMP->isBad())
                continue;
            if(!LastFrame.mvbOutlier[IdxLF])
                CurrentFrame.mvpMapPoints[IdxCF] = pMP;
             nmatches++;
             vnMatches1[i]=-1;
        }
        
    }
    
    return nmatches;
}

int SPmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const int th){
    int nmatches = 0;
    int nullnum = 0;
    int numObservations = 0;
    const bool bFactor = th != 1.0;
    for(size_t iMP=0;  iMP<vpMapPoints.size(); iMP++){
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView)
            continue;

        if(pMP->isBad())
            continue;
        const int &nPredictedLevel = 0;
        float r = RadiusByViewingCos(pMP->mTrackViewCos);
        if(bFactor)
            r*=th;
        int num = 0;
        const vector<size_t> vIndices = F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);
        if(vIndices.empty())
        {
            nullnum++;
            continue;
        }
           
        const cv::Mat MPdescriptor = pMP->GetDescriptor();
        float bestDist=256;
        int bestLevel= -1;
        float bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            
            const size_t idx = *vit;
            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0){

                    if(num == 0){
                        numObservations++;
                    }
                    num++;
                    continue;
                }

                    
            
            if(F.mvuRight[idx]>0)
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row(idx);
            const float dist = DescriptorDistance_sp(MPdescriptor,d);

            if(dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx = idx;
            }
            else if(dist < bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2 = dist;
            }
        }

        if(bestDist <= TH_HIGH)
        {
            F.mvpMapPoints[bestIdx] = pMP;
            nmatches++;
        }
    }
    return nmatches;
}
int SPmatcher::SearchByProjection1(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
{
        int nmatches=0, left = 0, right = 0;

        
        const bool bFactor = th!=1.0;

        
        for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
        {
            MapPoint* pMP = vpMapPoints[iMP];
            if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
                continue;

            if(bFarPoints && pMP->mTrackDepth>thFarPoints)
                continue;

            if(pMP->isBad())
                continue;

            if(pMP->mbTrackInView)
            {
                
                const int &nPredictedLevel = 0;

                // The size of the window will depend on the viewing direction
                
                float r = RadiusByViewingCos(pMP->mTrackViewCos);

                
                if(bFactor)
                    r*=th;

                
                const vector<size_t> vIndices =
                        F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,      
                                            r*F.mvScaleFactors[nPredictedLevel],    
                                            nPredictedLevel-1,nPredictedLevel);     
                
                if(!vIndices.empty()){
                    const cv::Mat MPdescriptor = pMP->GetDescriptor();

                    
                    float bestDist=256;
                    int bestLevel= -1;
                    float bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // Get best and second matches with near keypoints
                    
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;

                        
                        if(F.mvpMapPoints[idx])
                            if(F.mvpMapPoints[idx]->Observations()>0)
                                continue;

                        

                        const cv::Mat &d = F.mDescriptors.row(idx);

                        
                        const float dist = DescriptorDistance_sp(MPdescriptor,d);

                        
                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            bestLevel = F.mvKeysUn[idx].octave;
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            bestLevel2 = F.mvKeysUn[idx].octave;
                            bestDist2=dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    
                    
                    if(bestDist<=TH_HIGH)
                    {
                        
                        
                            
                            
                        F.mvpMapPoints[bestIdx] = pMP;
                        nmatches++;

                    }
                }
            }

            if(F.Nleft != -1 && pMP->mbTrackInViewR){
                const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
                if(nPredictedLevel != -1){
                    float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                    const vector<size_t> vIndices =
                            F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

                    if(vIndices.empty())
                        continue;

                    const cv::Mat MPdescriptor = pMP->GetDescriptor();

                    float bestDist=256;
                    int bestLevel= -1;
                    float bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // Get best and second matches with near keypoints
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;

                        if(F.mvpMapPoints[idx + F.Nleft])
                            if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
                                continue;


                        const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);

                        const float dist = DescriptorDistance_sp(MPdescriptor,d);

                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            bestLevel = F.mvKeysRight[idx].octave;
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            bestLevel2 = F.mvKeysRight[idx].octave;
                            bestDist2=dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    if(bestDist<=TH_HIGH)
                    {
                        if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                            continue;

                        if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                            F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                            nmatches++;
                            left++;
                        }


                        F.mvpMapPoints[bestIdx + F.Nleft]=pMP;
                        nmatches++;
                        right++;
                    }
                }
            }
        }
        return nmatches;
}
int SPmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                           vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
{
    int nmatches = 0;
    vector<int> vnMatches12 = vector<int>((*pKF1).mvKeys.size(), -1);
    



    nmatches = MatchingPoints_onnx(pKF1->mvKeys, pKF2->mvKeys, pKF1->mDescriptors, pKF2->mDescriptors,  vnMatches12);
    for(size_t i = 0, iend = vnMatches12.size(); i<iend; i++)
    {
        if(vnMatches12[i]<0)
        
            continue;
        MapPoint* pMP1 = pKF1->GetMapPoint(i);
        if(pMP1)
        {   //vnMatches12[i] = -1;
            continue;
        }
        MapPoint* pMP2 = pKF2->GetMapPoint(vnMatches12[i]);
        if(pMP2)
        {
            continue;
        }
        vMatchedPairs.push_back(make_pair(i,vnMatches12[i]));
    }
    return nmatches;
}

int SPmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12,vector<cv::DMatch>& vmatches)
    {

        
        const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeat3Vec;
        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        const cv::Mat &Descriptors1 = pKF1->mDescriptors;

        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
        const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeat3Vec;
        const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
        const cv::Mat &Descriptors2 = pKF2->mDescriptors;

        
        vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
        vector<bool> vbMatched2(vpMapPoints2.size(),false);

        

        int nmatches = 0;

        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it != f1end && f2it != f2end)
        {
            
            if(f1it->first == f2it->first)
            {
                
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
                        continue;
                    }

                    MapPoint* pMP1 = vpMapPoints1[idx1];
                    if(!pMP1)
                        continue;
                    if(pMP1->isBad())
                        continue;

                    const cv::Mat &d1 = Descriptors1.row(idx1);

                    float bestDist1=256;
                    int bestIdx2 =-1 ;
                    float bestDist2=256;

                    
                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        const size_t idx2 = f2it->second[i2];

                        if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
                            continue;
                        }

                        MapPoint* pMP2 = vpMapPoints2[idx2];

                        
                        if(vbMatched2[idx2] || !pMP2)
                            continue;

                        if(pMP2->isBad())
                            continue;

                        const cv::Mat &d2 = Descriptors2.row(idx2);

                        float dist = DescriptorDistance_sp(d1,d2);
                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }

                    
                    if(bestDist1<TH_LOW)
                    {
                        if(static_cast<float>(bestDist1)<0.8*static_cast<float>(bestDist2))
                        {
                            vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2]=true;
                            nmatches++;
                            cv::DMatch match;
                            match.queryIdx = idx1;
                            match.trainIdx = bestIdx2;
                            match.distance = bestDist1;
                            vmatches.push_back(match);
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        return nmatches;
    }

int SPmatcher::SearchByBoWSP(KeyFrame* pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12, vector<cv::DMatch> & vmatches)
{
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    int nmatches = 0;
    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<int> vnMatches12 = vector<int>((*pKF1).mvKeys.size(), -1);
    nmatches = MatchingPoints_onnx(pKF1->mvKeys, pKF2->mvKeys, pKF1->mDescriptors, pKF2->mDescriptors,  vnMatches12);
    for(size_t i = 0, iend = vnMatches12.size(); i<iend; i++)
    {
        if(vnMatches12[i]<0)
            continue;
        
        MapPoint* pMP1 = pKF1->GetMapPoint(i);
        if(!pMP1||pMP1->isBad())
        {
            continue;
        }
        MapPoint* pMP2 = pKF2->GetMapPoint(vnMatches12[i]);
        if(!pMP2||pMP2->isBad())
        {
            continue;
        }
        vpMatches12[i] = pMP2;
        cv::DMatch match;
        match.queryIdx = i;
        match.trainIdx = vnMatches12[i];
        vmatches.push_back(match);
    }
    return vmatches.size();
}
int SPmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                                       std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
{
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];
            KeyFrame* pKFi = vpPointsKFs[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0)
                continue;

            // Project into Image
            const float invz = 1/p3Dc(2);
            const float x = p3Dc(0)*invz;
            const float y = p3Dc(1)*invz;

            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // Point must be inside the image
            if(!pKF->IsInImage(u,v))
                continue;

            // Depth must be inside the scale invariance region of the point
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist = PO.norm();

            if(dist<minDistance || dist>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            float bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;
                if(vpMatched[idx])
                    continue;



                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                const float dist = DescriptorDistance_sp(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW)
            {
                vpMatched[bestIdx] = pMP;
                vpMatchedKF[bestIdx] = pKFi;
                nmatches++;
            }

        }

        return nmatches;
}
int SPmatcher::SearchByBoWSP(KeyFrame* pKF, Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
    const std::vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    vpMapPointMatches = std::vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));
    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> kpts0 = F.mvKeys;
    std::vector<cv::KeyPoint> kpts1 = pKF->mvKeys;
    std::vector<cv::Point2f> pts0;
    std::vector<cv::Point2f> pts1;
    for(int i = 0 ; i < kpts0.size(); i++)
    {
        pts0.push_back(kpts0[i].pt);
    }
    for(int i = 0 ; i < kpts1.size(); i++)
    {
        pts1.push_back(kpts1[i].pt);
    }
    cv::Mat desc0 = F.mDescriptors;
    cv::Mat desc1 = pKF->mDescriptors;
    std::vector<int> vnMatches01;
    int nmatches_ori = MatchingPoints_onnx(pts0, pts1, desc0, desc1, vnMatches01);

    int nmatches =0;
    for(int i = 0 ; i < vnMatches01.size(); i++){
        int IdxKF, IdxF;
        if(vnMatches01[i] != -1){
            IdxKF = vnMatches01[i];
            IdxF = i;
            MapPoint* pMP = vpMapPointsKF[IdxKF];
            if(!pMP)
                continue;
            if(pMP->isBad())
                continue;
            vpMapPointMatches[IdxF]=pMP;
            nmatches++;
        }
        else{
            continue;
        }
    }
    




        

    return nmatches;
}

float SPmatcher::RadiusByViewingCos(const float &viewCos)
{
    
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


int SPmatcher::SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    
    
    // Camera 1 & 2 from world
    Sophus::SE3f T1w = pKF1->GetPose();
    Sophus::SE3f T2w = pKF2->GetPose();

    //Transformation between cameras
    
    Sophus::Sim3f S21 = S12.inverse();
    //Camera 2 from world

    //Transformation between cameras

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = get<0>(pMP->GetIndexInKeyFrame(pKF2));
            if(idx2>=0 && idx2<N2)
                
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        
        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        
        Eigen::Vector3f p3Dc1 = T1w * p3Dw;
        
        Eigen::Vector3f p3Dc2 = S21 * p3Dc1;

        // Depth must be positive
        if(p3Dc2(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2(2);
        const float x = p3Dc2(0)*invz;
        const float y = p3Dc2(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = p3Dc2.norm();

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const float dist = DescriptorDistance_sp(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        Eigen::Vector3f p3Dw = pMP->GetWorldPos();
        Eigen::Vector3f p3Dc2 = T2w * p3Dw;
        Eigen::Vector3f p3Dc1 = S12 * p3Dc2;

        // Depth must be positive
        if(p3Dc1(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1(2);
        const float x = p3Dc1(0)*invz;
        const float y = p3Dc1(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = p3Dc1.norm();

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        float bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const float dist = DescriptorDistance_sp(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}






        


        

        

        



        


            



//             else

int SPmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3f &Scw, const vector<MapPoint*> &vpPoints,
                                       vector<MapPoint*> &vpMatched, int th, float ratioHamming)
{
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

        int nmatches=0;

        // For each Candidate MapPoint Project and Match
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0)
                continue;

            // // Project into Image

            const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

            // Point must be inside the image
            if(!pKF->IsInImage(uv(0),uv(1)))
                continue;

            // Depth must be inside the scale invariance region of the point
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist = PO.norm();

            if(dist<minDistance || dist>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist)
                continue;

            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;
                if(vpMatched[idx])
                    continue;



                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                const int dist = DescriptorDistance_sp(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW*ratioHamming)
            {
                vpMatched[bestIdx] = pMP;
                nmatches++;
            }

        }

        return nmatches;  
}

float SPmatcher::DescriptorDistance_sp(const cv::Mat &a, const cv::Mat &b)
{
    float dist = (float)cv::norm(a, b, cv::NORM_L2);

    return dist;
}

}

