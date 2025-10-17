#ifndef CAMERAWORKER_H
#define CAMERAWORKER_H

#include <CaptureWorker.h>
#include <PoseWorker.h>

class CameraWorker : public QMainWindow {
  Q_OBJECT
public:
  explicit CameraWorker(QWidget* parent=nullptr);
  ~CameraWorker();
  void doCaptureAndPose();

  //스트리밍
  void startPoseStreaming();
  void stopPoseStreaming();
  void togglePoseStreaming();

protected:
  void keyPressEvent(QKeyEvent* ev) override;

private slots:
    void onStart();
    void onSnapshot();
    void onFrame(int camidx, const QImage& img);
    void yoloFrameReady(const int idx, const QImage& yimg);
    void requestUsingyolo();

signals:
    void frameReady(int camIdx, const QImage& qimg);
    void requestSnapPose();
    void onSnapPose(const QImage& qimg);
    void requestPoseImage(const QImage& qimg);
    void requestSaveImage(const QImage& qimg);

    void requestStartBatch();

    void runPose(int camidx, QImage frame);


public slots:
  void requestCapture();
  void handleTermKey(char ch);
  void onFrameReady(int camIdx, const QImage& qimg);
  void onPoseDone(const QImage& img);



private:
  Ui::CameraWorker* ui;
  QThread vis_thread_;
  QThread tof_thread_;
  QThread pose_thread_;
  QThread save_thread_;
  //CaptureWorker* worker_ = nullptr;
  CaptureWorker* tof_worker_ = nullptr;
  CaptureWorker* vis_worker_ = nullptr;
  PoseWorker* pose_worker_ = nullptr;
  QImage lastFrame_;
  QElapsedTimer poseTick_;

  std::atomic<bool> capturePending_{false};


//포즈 추론
  PoseWorker* pose_ = nullptr;
  std::atomic<bool> aiEnabled_{true};
  std::atomic<bool> aiBusy_{false};
  QImage lastOverlay_[2];   // camidx 0/1 별 최신 오버레이 프레임 보관
  std::atomic<bool> haveOverlay_[2] = {false, false};

  bool poseReady_ = false;
  PoseParams pPose_;
  QString captureDir_ = "/home/CameraWorker/captures";

  QTimer* poseStreamTimer_ = nullptr;
  std::atomic<bool> poseStreaming_{false};
  QString poseStreamDir_ ="/home/CameraWorker/caputres/1001";
  int poseStreamSeq_ = 0;
  bool is_after_rec_ = true;

  QStringList batchFiles_;
  int batchIdx_ = -1;
  QString batchOutDir_;

  bool isusing_yolo = false;

public:
  Arena::ISystem* sys_ = nullptr;
};



#endif // CAMERAWORKER_H
