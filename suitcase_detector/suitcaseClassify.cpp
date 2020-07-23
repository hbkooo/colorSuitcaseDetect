#include "suitcaseClassify.h"
#include <cuda_runtime_api.h>

using namespace cv;
using namespace std;
using namespace Tn;

//void imageROIResize8U3C(void *src, int srcWidth, int srcHeight, cv::Rect imgROI, void *dst, int dstWidth, int dstHeight);
//void convertBGR2RGBfloat(void *src, void *dst, int width, int height, cudaStream_t stream);
//void imageSplit(const void *src, float *dst, int width, int height, cudaStream_t stream);

//######################################################################
// classify
//######################################################################

SuitcaseClassify::SuitcaseClassify(const std::string &prototxt, const std::string &caffemodel,
                                           const std::string &saveName) {
	
	vector<vector<float>> calibData;
	class_num = 12;     // coco的80类
    net.reset(new trtNet(prototxt, caffemodel, class_num, {"prob"}, calibData));
    cout << "save Engine..." << saveName << endl;
    net->saveEngine(saveName);
    outputCount = net->getOutputSize() / sizeof(float);	
}

SuitcaseClassify::SuitcaseClassify(const std::string &engineName) {
    net.reset(new trtNet(engineName));
    outputCount = net->getOutputSize() / sizeof(float);
}

SuitcaseClassify::~SuitcaseClassify() {
//    delete trtNet;
    //free(cpuBuffers);
	net.reset(nullptr);
}

vector<float> SuitcaseClassify::prepareImage(cv::Mat& img)
{
    int c = 3;
    int h = 32;   //net h
    int w = 32;   //net w

    cv::Mat resize;
    cv::resize(img.clone(), resize, cv::Size(w, h));
    resize.convertTo(resize, CV_32FC3);

    cv::Mat mean_ = SetMean("94.5168,100.976,119.113", w, h, c);  // 104,117,123
    cv::subtract(resize, mean_, resize);

    //HWC TO CHW
    vector<Mat> input_channels(c);
    cv::split(resize, input_channels);

    vector<float> result(h * w * c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        data += channelLength;
    }

    return result;
}

std::vector<float> SuitcaseClassify::classify(Mat &img) {
    std::vector<float> results;
    if (img.empty()) {
        return results;
    }

    unique_ptr<float[]> outputData(new float[outputCount]);
    vector<float> curInput = prepareImage(img);

    net->doInference(curInput.data(), outputData.get(), 1);

    auto output = outputData.get();
    results.resize(outputCount);

//    cout << outputCount << endl;

    for(int i = 0; i < results.size(); i++) {
        results[i] = output[i];
    }

    outputData.reset(nullptr);
    return results;

}

std::string SuitcaseClassify::classifyColor(Mat &img, float &score) {
    std::vector<float> result = classify(img);
    if(result.size() <= 0)
        return COLOR[0];

    assert(result.size() <= COLOR.size());

    int maxId = 0;
    score = result[maxId];
    for(int i = 0; i < result.size(); i++) {
        if(result[i] > score) {
            maxId = i;
            score = result[i];
        }
    }
    return COLOR[maxId];
}

cv::Mat SuitcaseClassify::SetMean(const string &mean_value, int inputW, int inputH, int channels_) {
    //mean_value
    cv::Mat mean_;

    cv::Scalar channel_mean;
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
        float value = std::atof(item.c_str());
        values.push_back(value);
    }

    std::vector<cv::Mat> channels;
    for (int i = 0; i < channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(inputH, inputW, CV_32FC1,
                        cv::Scalar(values[i]));
        channels.push_back(channel);
    }
    cv::merge(channels, mean_);
    return mean_;
}

