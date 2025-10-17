#ifndef UTILHEADER_H
#define UTILHEADER_H

#pragma once
#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QThread>
#include <atomic>
#include <QImage>
#include <QObject>
#include <QDir>
#include <qdatetime.h>
#include <memory>


namespace Ui { class CameraWorker; }
namespace Arena { class ISystem; class IDevice; class IImage;}

class PoseEstimatorONNX;

struct PoseParams {
    int   inputW = 640, inputH = 640;
    bool  letterbox = true;
    float confDet = 0.25f;
    float confKpt = 0.20f;
    float nmsIoU  = 0.45f;
    bool  rgbInput = true;


    //** 추가: 후처리 필터
    int   minBoxPx      = 48;     // 최소 한 변 픽셀 (소물체 오검 방지)
    float minAreaRatio  = 0.010f; // bbox 면적/이미지 면적 최소 비율(1% 미만 버림)
    float maxAreaRatio  = 0.85f;  // 화면 대부분 차지하는 박스(배경/검은영역) 컷
    int   minKeypts     = 6;      // 신뢰 가능한 키포인트 수
    float minAvgKptConf = 0.35f;  // 키포인트 평균 conf 하한
};
struct PoseKpt
{
    float x;
    float y;
    float conf;
};

struct Letterbox
{
    int netW, netH;
    float scale;
    float padX;
    float padY;
};
#endif // UTILHEADER_H
