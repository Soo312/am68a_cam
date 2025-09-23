#ifndef POSE_ESTIMATE_ONNX_H
#define POSE_ESTIMATE_ONNX_H

#pragma once
#include <vector>
#include <QString>
#include <onnxruntime_cxx_api.h>
#include "cameraworker.h"

struct PoseKpt { float x, y, c; };          // 좌표 + confidence
struct PosePerson {
    float x, y, w, h, score;                // bbox
    std::vector<PoseKpt> kpts;              // 17개
};

/*
struct PoseParams {
    int   inputW = 640, inputH = 640;
    bool  letterbox = true;
    float confDet = 0.25f;                  // bbox 신뢰도
    float confKpt = 0.2f;                   // 키포인트 표시 임계
    float nmsIoU  = 0.45f;
    bool  rgbInput = true;                  // RGB888 입력
};
*/

class PoseEstimatorONNX {
public:
    bool init(const QString& onnxPath, const PoseParams& P, int threads=2);
    bool infer(const uchar* data, int w, int h, int strideBytes, std::vector<PosePerson>& out);

private:
    void preprocess(const uchar* src, int w, int h, int stride, float* dst);
    void nms(const std::vector<PosePerson>& in, std::vector<PosePerson>& out);
    void scaleBack(PosePerson& p);

private:
    PoseParams P_;
    void* env_ = nullptr;        // Ort::Env*
    void* sess_ = nullptr;       // Ort::Session*
    void* mem_  = nullptr;       // Ort::MemoryInfo*
    std::vector<const char*> in_names_, out_names_;
    std::vector<Ort::AllocatedStringPtr> in_keep_, out_keep_;
};

#endif // POSE_ESTIMATE_ONNX_H
