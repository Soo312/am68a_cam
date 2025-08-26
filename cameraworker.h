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


public:
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
  QThread thread_;
  CaptureWorker* worker_ = nullptr;
  QImage lastFrame_;
};

#endif // CAMERAWORKER_H
