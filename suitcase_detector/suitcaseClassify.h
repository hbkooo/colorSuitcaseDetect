#ifndef SUITCASECLASSIFY_H
#define SUITCASECLASSIFY_H

#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <TrtNet.h>
#include "YoloLayer.h"

const std::vector<std::string> COLOR = {"black", "blue", "brown", "gray", "green",
                                        "orange", "other", "pink", "purple", "red",
                                        "white", "yellow"};

class SuitcaseClassify {
public:
    SuitcaseClassify(const std::string &prototxt, const std::string &caffemodel, const std::string &saveName);
	SuitcaseClassify(const std::string &engineName);

    ~SuitcaseClassify();

    // get 12 scores of different coclor
    std::vector<float> classify(cv::Mat &img);

    // get max score color name and max score
    std::string classifyColor(cv::Mat &img, float &score);

private:
    std::vector<float> prepareImage(cv::Mat& img);
    cv::Mat SetMean(const std::string &mean_value, int inputW, int inputH, int channels_);

private:
    int outputCount;
    int class_num;
    std::unique_ptr<Tn::trtNet> net;

};

#endif // MASKCLASSIFY_H
