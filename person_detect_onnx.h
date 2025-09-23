#ifndef PERSON_DETECT_ONNX_H
#define PERSON_DETECT_ONNX_H

#pragma once
#include <vector>
#include <QString>

struct PDBox
{
    float x, y, w, h;  // bounding box
    float score;       // confidence
    int   label;       // class ID (COCO 기준: person=0)
};

struct PDParams
{
    int   inputW = 640;
    int   inputH = 640;
    bool  letterbox = true;
    float confThresh = 0.25f;
    float nmsIoU     = 0.45f;
    bool  rgbInput   = true;  // true: RGB888, false: Grayscale
};

class PersonDetectorONNX
{
public:
    bool init(const QString& modelPath, const PDParams& params, int threads = 1);
    bool detect(const uchar* input, int w, int h, int strideBytes, std::vector<PDBox>& outBoxes);

private:
    void preprocess(const uchar* input, int w, int h, int strideBytes, float* outTensor);
    void nms(const std::vector<PDBox>& in, std::vector<PDBox>& out);
    void scaleCoords(PDBox& box);

private:
    PDParams P_;
    void* session_ = nullptr;
    void* env_ = nullptr;
    void* memoryInfo_ = nullptr;
};

#endif // PERSON_DETECT_ONNX_H
