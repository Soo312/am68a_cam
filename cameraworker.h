#ifndef CAMERAWORKER_H
#define CAMERAWORKER_H

#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QThread>
#include <atomic>
#include <QImage>
#include <QVector3D>
#include <QMutex>
#include <QMutexLocker>
#include <vector>



//3D 출력
#include "cpu_pointcloud_view.h"
#include  <QShortcut>

QT_BEGIN_NAMESPACE
namespace Ui { class CameraWorker; }
namespace Arena { class ISystem; class IDevice; class IImage;}
QT_END_NAMESPACE

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
    void errorOccurred(const QString& msg);
    void pointCloudReady(int camidx, std::vector<QVector3D>& point);
    void frameReadyABCY16(int camidx,
                          QByteArray raw,
                          size_t width,
                          size_t height,
                          size_t sizeFilled,
                          size_t strideBytes);

public :
    void onFrameRaw(Arena::IImage* img);
    bool takeLatestPointCloud
    (
        const QVector<QVector3D>*& outPts,
        const QVector<quint16>*&   outConf,
        int& outW,
        int& outH
    );

private:
    bool openDevice();
    void closeDevice();
    QImage toQImage2D(Arena::IImage* pImg);
private:
    QString hint_;
    bool isToF_ = false;
    std::atomic<bool> running_{false};
    Arena::ISystem* sys_ = nullptr;
    Arena::IDevice* dev_ = nullptr;
    // ▼ 가시광 표시용 더블버퍼
    QImage visBuf_[2];
    int    visIdx_ = 0;



public:
    void setSystem(Arena::ISystem* s);
    int camIdx_ = -1;

    // 포인트클라우드 더블버퍼 (ToF 전용)
private:
    QVector<QVector3D> pcBuf_[2];
    QVector<quint16>   confBuf_[2];
    int                pcIdx_ = 0;          // write index
    std::atomic<bool>  pcHasNew_{ false };
    int                lastW_ = 0;
    int                lastH_ = 0;

};

class CameraWorker : public QMainWindow {
  Q_OBJECT
public:
  explicit CameraWorker(QWidget* parent=nullptr);
  ~CameraWorker();
    CPUPointCloudView* pcView() const  // ✅ public getter
    {
        return pcView_;
    }

public slots:
    void handleTermKey(char ch);

private slots:
  void onStart();
  void onSnapshot();
  void onFrame(int camidx, const QImage& img);//2D용

  void onFrameABCY16(int camidx,
                     QByteArray data,
                     size_t width,
                     size_t height,
                     size_t sizeFilled);

  void setPointCloudView(CPUPointCloudView* view){pcView_ = view;}

  //Tof 타이머 슬롯
  void onPcPoll();

private:
  Ui::CameraWorker* ui;
  QThread vis_thread_;
  QThread tof_thread_;
  //CaptureWorker* worker_ = nullptr;
  CaptureWorker* tof_worker_ = nullptr;
  CaptureWorker* vis_worker_ = nullptr;
  QImage lastFrame_;

  CPUPointCloudView* pcView_ = nullptr;
  QTimer pcPollTimer_;

public:
  Arena::ISystem* sys_ = nullptr;
};

#endif // CAMERAWORKER_H
