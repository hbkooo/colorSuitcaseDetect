#ifndef SUITCASE_DETECTOR_H
#define SUITCASE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include "TrtNet.h"
#include <vector>
#include <chrono>
#include "YoloLayer.h"
#include <list>
#include <map>
#include <set>

#include "suitcaseClassify.h"

namespace Yolo {
    struct Bbox {
        int classId;
        int left;
        int right;
        int top;
        int bot;
        float score;
    };
}


struct SUITCASE_BOX {
    std::string label;  // suitcase/backpack
    std::string color;
    int left;
    int top;
    int right;
    int bot;
    float score;
};

//extern const std::vector<std::string> COLOR;

class SuitcaseDetector {
public:
    //Load from caffe model
    SuitcaseDetector(const std::string &prototxt, const std::string &caffemodel, const std::string &saveName);

    ~SuitcaseDetector() { net.reset(nullptr); }

    //Load from engine file
    explicit SuitcaseDetector(const std::string &engineName);

    // 初始化行李颜色分类器
    void InitSuitcaseClassifier(const std::string &prototxt, const std::string &caffemodel,
            const std::string &saveName);

    // 检测行李箱
    std::vector<SUITCASE_BOX> DetectSuitcase(cv::Mat &img, float threshold = 0.5f);

private:
    std::vector<float> prepareImage(cv::Mat &img);

    std::vector<Yolo::Bbox>
    postProcessImg(cv::Mat &img, std::vector<Yolo::Detection> detections, int classid, float threshold);

    void DoNms(std::vector<Yolo::Detection> &detections, float nmsThresh);

    std::vector<Yolo::Bbox> Detect(cv::Mat &img, float threshold = 0.5f);

private:
    int outputCount;
    const int class_num = 80;
    std::unique_ptr<Tn::trtNet> net;
    bool init_flag = false;

    std::unique_ptr<SuitcaseClassify> classifier;

    std::map<int, std::string> LABELS = {
            {24, "backpack"},
            {28, "suitcase"},
            {26, "handbag"},
//            {0,  "person"}
    };

};


#endif
