#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat NormalizeImage(cv::Mat& Image);

std::vector<cv::Point2f> NormalizeKeypoints(std::vector<cv::Point2f> kpts, int h , int w);


cv::Mat RGB2Grayscale(cv::Mat& Image);