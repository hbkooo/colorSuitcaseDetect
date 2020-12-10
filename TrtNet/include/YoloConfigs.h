#ifndef _YOLO_CONFIGS_H_
#define _YOLO_CONFIGS_H_


namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.5f;
    // static constexpr int CLASS_NUM = 2; //No more need

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };

    struct Bbox {
        // 用于结果判定
        int classId;
        // 检测结果区域的矩形左上角x
        int left;
        // 检测结果区域的矩形左上角y
        int right;
        // 检测结果区域的矩形高度
        int top;
        // 检测结果区域的矩形宽度
        int bot;
        //置信度
        float score;
    };

    //YOLO 608
    

    //YOLO 416
    
}

#endif
