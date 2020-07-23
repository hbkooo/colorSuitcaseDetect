#ifndef BAGGAGE_FEATURE_H
#define BAGGAGE_FEATURE_H

#include <opencv2/opencv.hpp>
#include "TrtNet.h"
#include <vector>
#include <chrono>

class BaggageFeature
{
public:

	//Load from engine file
	explicit BaggageFeature(const std::string& engineName);
	std::vector<float> get_feature(cv::Mat& img);
//	float* get_feature(cv::Mat& img);
private:
    int outputCount;
	std::unique_ptr<Tn::trtNet> net;
	std::vector<float> prepareImage(cv::Mat& img);
};

#endif