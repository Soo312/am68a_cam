#ifndef CAMERAWORKER_H
#define CAMERAWORKER_H

#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QThread>
#include <atomic>
#include <QImage>

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

public slots:
    void onFrameRaw(Arena::IImage* img);

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


};

class CameraWorker : public QMainWindow {
  Q_OBJECT
public:
  explicit CameraWorker(QWidget* parent=nullptr);
  ~CameraWorker();
private slots:
  void onStart();
  void onSnapshot();
  void onFrame(int camidx, const QImage& img);

private:
  Ui::CameraWorker* ui;
  QThread vis_thread_;
  QThread tof_thread_;
  //CaptureWorker* worker_ = nullptr;
  CaptureWorker* tof_worker_ = nullptr;
  CaptureWorker* vis_worker_ = nullptr;
  QImage lastFrame_;

public:
  Arena::ISystem* sys_ = nullptr;
};

#endif // CAMERAWORKER_H
