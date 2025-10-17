#ifndef YOLOX_CPU_DETECTOR_H
#define YOLOX_CPU_DETECTOR_H

#pragma once

#include <QImage>
#include <QRectF>
#include <vector>
#include <memory>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

struct YXBox
{
    QRectF rect;
    float  score;
    int    cls;   // person(0)만 사용
};

class YoloXNanoLiteCpu
{
public:
    YoloXNanoLiteCpu();
    ~YoloXNanoLiteCpu() = default;

    bool init(const QString& modelPath, int inputSize = 416, int threads = 4);
    bool ready() const;

    // qimgIn에 대해 person 박스만 반환
    std::vector<YXBox> detect(const QImage& qimgIn, float scoreTh = 0.25f, float nmsIouTh = 0.45f);

private:
    // 전처리/후처리
    static cv::Mat letterbox(const cv::Mat& src, int dstW, int dstH, float& scale, cv::Point& pad);
    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);
    static std::vector<YXBox> nms(const std::vector<YXBox>& boxes, float iouTh);

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "yolox_cpu"};
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;

    std::string inputName_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> outputNamePtrs_; // Run 호출용

    int inW_ = 416;
    int inH_ = 416;

    // 입력 dtype/레이아웃 메타
    ONNXTensorElementDataType inputType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    bool nhwc_ = false; // true면 [N,H,W,C], false면 [N,C,H,W]

    std::vector<float>  inputBuf_;    // FP32 CHW
    std::vector<uint8_t> inputBufU8_; // UINT8 CHW
};

#endif // YOLOX_CPU_DETECTOR_H
