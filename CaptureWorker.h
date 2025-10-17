#ifndef CAPTUREWORKER_H
#define CAPTUREWORKER_H

#include <utilHeader.h>



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


#endif // CAPTUREWORKER_H
