#include "baggage_feature.h"

using namespace Tn;
using namespace std;
using namespace cv;

BaggageFeature::BaggageFeature(const std::string& engineName){
	net.reset(new trtNet(engineName));
    outputCount = net->getOutputSize()/sizeof(float);
}

vector<float> BaggageFeature::prepareImage(cv::Mat& img)
{
    int c = 3;
    int h = 384;   //net h
    int w = 128;   //net w

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, CV_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(w,h), 0, 0, cv::INTER_CUBIC);  // INTER_AREA

    cv::Mat img_float;
    if (c == 3)
        resized.convertTo(img_float, CV_32FC3,1/255.0);
    else
        resized.convertTo(img_float, CV_32FC1,1/255.0);
    
    cv::Mat img_normalized;
    cv::subtract(img_float,cv::Scalar(0.485,0.456,0.406),img_normalized);

    cv::Mat img_normalized1;
    cv::divide(img_normalized,cv::Scalar(0.229,0.224,0.225),img_normalized1);
    //std::cout<<"img:"<<img_float<<std::endl;
    //HWC TO CHW
    vector<Mat> input_channels(c);
    cv::split(img_normalized1, input_channels);

    vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

std::vector<float>  BaggageFeature::get_feature(cv::Mat& img){
//    float*  BaggageFeature::get_feature(cv::Mat& img){
    unique_ptr<float[]> outputData(new float[outputCount]);
    // std::cout<<"outputCount:"<<outputCount<<std::endl;
    vector<float> curInput = prepareImage(img);
    //前向
    net->doInference(curInput.data(), outputData.get(), 1);


    //得到输出
    auto output = outputData.get();
    std::vector<float>  result ;
     for(int i = 0;i<outputCount;i++){
         result.push_back(output[i]);

     }

    return result;
//    return output;
}

