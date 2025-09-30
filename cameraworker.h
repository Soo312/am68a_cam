#ifndef CAMERAWORKER_H
#define CAMERAWORKER_H

#pragma once
#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QThread>
#include <atomic>
#include <QImage>
#include <QObject>
//#include "pose_estimate_onnx.h"

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


class CaptureWorker : public QObject
{
    Q_OBJECT

public:
    explicit CaptureWorker(QString deviceHint, bool isToF=false,int camIdx = -1, QObject* parent=nullptr);
    ~CaptureWorker();
    QImage bayerRG8ToRgbQImage(Arena::IImage* pImg);
public slots:
    void start();
    void stop();

signals:
    //void frameReady(const QImage& img, const QImage& img2);
    void frameReady(int camidx,QImage img);
    void frameReadyC16(int camidx,QImage& falseColor,QImage& grayPose);
    void errorOccurred(const QString& msg);
    void poseRequest(const QImage& img);


public slots:
    void onFrameRaw(Arena::IImage* img);
    void captureOnceC16();
    void requestSnap()
    {
         snapPending_.store(true, std::memory_order_release);
    }
    //void setPoseStreaming(bool on, int inferStride = 2);

private:
    bool openDevice();
    void closeDevice();
    QImage toQImage2D(Arena::IImage* pImg);
    std::atomic_bool captureOnceFlag_{false};
    std::atomic<bool> snapPending_{false};
    //void snapPose(const QImage& inimg); // ← 네가 원하는 "내부 함수" (GetImage + Gray8 + YOLO + 저장)
    std::atomic<bool> poseStreaming_{false};
    int inferStride_ = 2;
    int streamFrameIdx_ = 0;


private:
    QString hint_;
    bool isToF_ = false;
    std::atomic<bool> running_{false};
    Arena::ISystem* sys_ = nullptr;
    Arena::IDevice* dev_ = nullptr;
    // ▼ 가시광 표시용 더블버퍼
    QImage visBuf_[2];
    int    visIdx_ = 0;
    float scaleX_ = 1.0;
    float offX_  = 1.0;
    float scaleY_ = 1.0;
    float offY_  = 1.0;
    float scaleZ_ = 1.0;
    float offZ_  = 1.0;

public:
    void setSystem(Arena::ISystem* s);
    int camIdx_ = -1;
    //포즈 추론
      PoseEstimatorONNX* pose_ = nullptr;
      bool poseReady_ = false;
      PoseParams pPose_;
    QString captureDir_ = "/home/CameraWorker/captures";

};


class PoseWorker : public QObject
{
    Q_OBJECT
public :
    explicit PoseWorker(QObject* parent = nullptr);
    ~PoseWorker() override = default;

    PoseEstimatorONNX* pose_ = nullptr;
    PoseParams pPose_;
    std::atomic<bool> busy_{false};

    bool poseReady_ = false;
    QString captureDir_ = "/home/CameraWorker/captures";

signals:
    void poseDone(const QImage& canvas);

public slots:
    void snapPose(const QImage& inimg);



};

class CameraWorker : public QMainWindow {
  Q_OBJECT
public:
  explicit CameraWorker(QWidget* parent=nullptr);
  ~CameraWorker();
  void doCaptureAndPose();
protected:
  void keyPressEvent(QKeyEvent* ev) override;

private slots:
    void onStart();
    void onSnapshot();
    void onFrame(int camidx, const QImage& img);



public slots:
  void requestCapture();
  void handleTermKey(char ch);
  void onFrameReady(int camIdx, const QImage& qimg);
  void onPoseDone(const QImage& img);

signals:
    void frameReady(int camIdx, const QImage& qimg);
    void requestSnapPose();
    void onSnapPose(const QImage& qimg);
    void requestPoseImage(const QImage& qimg);

private:
  Ui::CameraWorker* ui;
  QThread vis_thread_;
  QThread tof_thread_;
  QThread pose_thread_;
  //CaptureWorker* worker_ = nullptr;
  CaptureWorker* tof_worker_ = nullptr;
  CaptureWorker* vis_worker_ = nullptr;
  PoseWorker* pose_worker_ = nullptr;
  QImage lastFrame_;

  std::atomic<bool> capturePending_{false};

//포즈 추론
  PoseEstimatorONNX* pose_ = nullptr;
  bool poseReady_ = false;
  PoseParams pPose_;
  QString captureDir_ = "/home/CameraWorker/captures";
public:
  Arena::ISystem* sys_ = nullptr;
};



#endif // CAMERAWORKER_H
