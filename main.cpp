#include<iostream>
#include "suitcase_detector/suitcase_detector.h"

using namespace std;
using namespace cv;

string labels[] = { "black",  "blue", "brown", "gray", "green", "orange", "other",
        "pink", "purple", "red", "white", "yellow" };

bool fileIsExist2(const char *filePath)
{
    std::ifstream _file(filePath, std::ios::in);
    if(_file) {
        _file.close();
        return true;
    }
    else
        return false;
}

void testYolov3(string prototxt, string caffemodel, string engine,
        string classfyP, string classifyC, string classifyE, string img_path) {
    cout << "[info] : test yolov3 ...\n";
    SuitcaseDetector *detector = nullptr;
    cout << "start load detector ...\n";
    if(fileIsExist2(engine.c_str()))
        detector = new SuitcaseDetector(engine);
    else
        detector = new SuitcaseDetector(prototxt, caffemodel, engine);

    detector->InitSuitcaseClassifier(classfyP, classifyC, classifyE);

    cout << "loading detector over\n";

    cout << "detect image : " << img_path << endl;
    Mat img = imread(img_path);

    std::vector<SUITCASE_BOX> boxes = detector->DetectSuitcase(img);

    cout << "detect result : \n";
    for (const auto& item : boxes) {

        cout << "label : " << item.label << ", color : " << item.color
             << ", left : " << item.left << ", right : " << item.right
             << ", top : " << item.top << ", bottom : " << item.bot
             << ", score : " << item.score << endl;
        rectangle(img,Point(item.left, item.top),Point(item.right, item.bot),cv::Scalar(0, 0, 255),3,8,0);
        putText(img, item.label, Point(item.left, item.top), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 255, 255), 2, 2, 0);
    }
    std::cout << "\nsave result ../result.jpg\n\n";
    imwrite("../result.jpg", img);

    delete detector;
    detector = nullptr;
}

void testClassify(string prototxt, string caffemodel, string engine, string img_path){

    cout << "[info] : test classify ...\n";
    SuitcaseClassify *classify = nullptr;

    if(fileIsExist2(engine.c_str()))
        classify = new SuitcaseClassify(engine);
    else
        classify = new SuitcaseClassify(prototxt, caffemodel, engine);

    Mat img = imread(img_path);
    vector<float> result = classify->classify(img);
    cout << "result score : \n";
    int maxLabel = 0;
    float maxScore = result[0];
    for(int i = 0; i < result.size(); i++) {
        if(result[i] > maxScore) {
            maxLabel = i;
            maxScore = result[i];
            // cout << "in if : update maxscore : " << maxScore << ", result[i] : " << result[i] << endl;
        }
        cout << labels[i] << " : "  << result[i] << endl;
    }
    cout << "classify result : " << labels[maxLabel] << " : "  << maxScore << endl;

    delete classify;
    classify = nullptr;


}

int main(int argc, char** argv){
	
	std::string prototxt = "../suitcase.prototxt";
	std::string caffemodel = "../suitcase.caffemodel";
	std::string engine = "../suitcase.engine";
    std::string Cprototxt = "../classify.prototxt";
    std::string Ccaffemodel = "../classify.caffemodel";
    std::string Cengine = "../classify.engine";
	
	if(argc <= 2){
		cout << "use ./main img_path [detect|classify] \n";
		return -1;
	}

	if(argc > 2 && !strcmp(argv[2], "classify")) {

	}
	std::string img_path = argv[1];

	std::cout << "prototxt : " << prototxt << endl
	            << "caffemodel : " << caffemodel << endl
	            << "engine : " << engine << endl
	            << "image_path : " << img_path << endl;

	if(!strcmp(argv[2], "classify"))
        testClassify(prototxt,caffemodel, engine, img_path);
	else
	    testYolov3(prototxt,caffemodel, engine, Cprototxt,Ccaffemodel, Cengine, img_path);

	
	return 0;
}
