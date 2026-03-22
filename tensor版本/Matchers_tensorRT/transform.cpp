#include "Matchers/transform.h"
#include "opencv2/opencv.hpp"
cv::Mat NormalizeImage(cv::Mat& Image)
{
    cv::Mat normalizedImage = Image.clone();

    if (Image.channels() == 3) {
        cv::cvtColor(normalizedImage, normalizedImage, cv::COLOR_BGR2RGB);
        normalizedImage.convertTo(normalizedImage, CV_32F, 1.0 / 255.0);
    } else if (Image.channels() == 1) {
        Image.convertTo(normalizedImage, CV_32F, 1.0 / 255.0);
    } else {
        throw std::invalid_argument("[ERROR] Not an image");
    }

    return normalizedImage;
}

std::vector<cv::Point2f> NormalizeKeypoints(std::vector<cv::Point2f> kpts, int h , int w)
{
    cv::Size size(w, h);
    cv::Point2f shift(static_cast<float>(w) / 2, static_cast<float>(h) / 2);
    float scale = static_cast<float>((std::max)(w, h)) / 2;

    std::vector<cv::Point2f> normalizedKpts;
    for (const cv::Point2f& kpt : kpts) {
        cv::Point2f normalizedKpt = (kpt - shift) / scale;
        normalizedKpts.push_back(normalizedKpt);
    }

    return normalizedKpts;
}

//     // Resize an image to a fixed size, or according to max or min edge.






cv::Mat RGB2Grayscale(cv::Mat& Image) {
    cv::Mat resultImage;
    cv::cvtColor(Image, resultImage, cv::COLOR_RGB2GRAY);

    return resultImage;
}