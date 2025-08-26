
#include "cameraworker.h"
#include "ui_cameraworker.h"
#include "ImageRenderHelper.h"

#include <QFileDialog>
#include <QDateTime>
#include <QPixmap>

//Arena
#include <Arena/ArenaApi.h>
#include <GenApi/GenApi.h>

#include <Base/GCException.h>
using namespace std;

CaptureWorker::CaptureWorker(QString hint,bool isToF,int camIdx, QObject* p)
    :QObject(p), hint_(std::move(hint)), isToF_(isToF),camIdx_(camIdx){}
CaptureWorker::~CaptureWorker(){stop();closeDevice();}

bool CaptureWorker::openDevice()
{
    try
    {
        auto* sys = Arena::OpenSystem();
        sys_ = sys;
        sys_->UpdateDevices(100);
        std::vector<Arena::DeviceInfo> devs = sys_->GetDevices();
        if(devs.empty()) return false;

        size_t idx = 0;
        Arena::DeviceInfo* target = nullptr;
        if(!hint_.isEmpty())
        {
            for (auto& d : devs)
            {
                const QString ip     = QString::fromLatin1(d.IpAddressStr().c_str());
                const QString serial = QString::fromLatin1(d.SerialNumber().c_str());
                const QString mac    = QString::fromLatin1(d.MacAddressStr().c_str());
                if (ip == hint_ || serial == hint_ || mac == hint_)
                {
                    target = &d;                       // << 인덱스 대신 DeviceInfo 복사
                    break;
                }
            }
        }

        dev_ = sys_->CreateDevice(*target);
        return true;
    }
    catch (...)
    {
        return false;
    }

}

void CaptureWorker::closeDevice()
{
    if (dev_) { try { sys_->DestroyDevice(dev_); } catch (...) {} dev_ = nullptr; }
    if (sys_) { try { Arena::CloseSystem(sys_); } catch (...) {} sys_ = nullptr; }
}

QImage CaptureWorker::toQImage2D(Arena::IImage *pImg)
{
    const int w = (int)pImg->GetWidth();
    const int h = (int)pImg->GetHeight();

    // 현재 포맷이 Mono8이면 바로 사용
    {
        GenApi::CEnumerationPtr pixFmt = dev_->GetNodeMap()->GetNode("PixelFormat");
        GenApi::CEnumEntryPtr eMono8 = pixFmt ? pixFmt->GetEntryByName("Mono8") : nullptr;
        if (eMono8 && (uint64_t)eMono8->GetValue() == pImg->GetPixelFormat()) {
            QImage q((const uchar*)pImg->GetData(), w, h, QImage::Format_Grayscale8);
            return q.copy();
        }
    }

    // 그 외 포맷은 BGR8로 변환 (BGR8의 정수값을 Enum에서 취득)
    uint64_t bgr8Val = 0;
    {
        GenApi::CEnumerationPtr pixFmt = dev_->GetNodeMap()->GetNode("PixelFormat");
        GenApi::CEnumEntryPtr eBgr8 = pixFmt ? pixFmt->GetEntryByName("BGR8") : nullptr;
        if (!eBgr8) {
            qWarning("PixelFormat entry 'BGR8' not available");
            return QImage();
        }
        bgr8Val = (uint64_t)eBgr8->GetValue();
    }

    Arena::IImage* pBgr = nullptr;
    try {
        pBgr = Arena::ImageFactory::Convert(pImg, bgr8Val);
    } catch (...) {
        qWarning("ImageFactory::Convert to BGR8 failed");
        return QImage();
    }

    // QImage는 RGB888, 데이터는 BGR → rgbSwapped()
    QImage q((const uchar*)pBgr->GetData(), w, h, QImage::Format_RGB888);
    QImage rgb = q.rgbSwapped().copy(); // deep copy (버퍼 회수 대비)
    Arena::ImageFactory::Destroy(pBgr);
    return rgb;
}


void CaptureWorker::start()
{
    if(!openDevice())
    {
        emit errorOccurred("Open device failed");
        return;
    }
    running_ = true;

    auto sMap = dev_->GetTLStreamNodeMap();
    auto dMap = dev_->GetNodeMap();


    // 0) (선택) 안전을 위해 Acquisition 멈춰있게 보장
    //    이미 멈춰있으면 예외 안 남
    try {
        if (auto n = GenApi::CCommandPtr(dMap->GetNode("AcquisitionStop")); GenApi::IsWritable(n)) n->Execute();
    } catch (...) {}

    // 1) 트리거/취득 모드 — 스트리밍 기본값
    if (auto n = GenApi::CEnumerationPtr(dMap->GetNode("TriggerMode"));     GenApi::IsWritable(n)) n->FromString("Off");
    if (auto n = GenApi::CEnumerationPtr(dMap->GetNode("AcquisitionMode")); GenApi::IsWritable(n)) n->FromString("Continuous");

    // 2) 스트림 계층: 자동 MTU 협상 끄기
    if (auto n = GenApi::CBooleanPtr(sMap->GetNode("StreamAutoNegotiatePacketSize")); GenApi::IsWritable(n))
        n->SetValue(false);

    // 3) 장치 측 패킷 크기 “작게” 고정 (MTU=1500 환경에서도 100% 붙는 값부터 시도)
    if (auto n = GenApi::CIntegerPtr(dMap->GetNode("GevSCPSPacketSize")); GenApi::IsWritable(n))
        n->SetValue(1400);
    // 1440이 일반 안전값이지만, 스위치/보드 조합에 따라 1400이 더 확실할 때가 많다.
    // 그래도 안 되면 1300 → 1200 → 1000 순으로 내려가며 시도.

    // 4) 인터패킷 딜레이로 혼잡 완화 (필요 시 값 ↑)
    if (auto n = GenApi::CIntegerPtr(dMap->GetNode("GevSCPD")); GenApi::IsWritable(n))
        n->SetValue(5000);
    // 여전히 타임아웃이면 10000, 20000까지도 올려볼 수 있음 (나노초 단위인 경우 多).

    // 5) 패킷 재전송 켜기 (유실 대비)
    if (auto n = GenApi::CBooleanPtr(sMap->GetNode("StreamPacketResendEnable")); GenApi::IsWritable(n)) n->SetValue(true);

    // 6) 버퍼 처리 모드 (프레임 드롭 시 최신 프레임 유지)
    if (auto n = GenApi::CEnumerationPtr(sMap->GetNode("StreamBufferHandlingMode")); GenApi::IsWritable(n)) n->FromString("NewestOnly");

    // 7) (강력 권장) 링크 스루풋 제한으로 초반 폭주 방지
    //    모드가 있으면 켜고 적당한 bps로 제한 (예: 80 Mbps)
    if (auto m = GenApi::CEnumerationPtr(dMap->GetNode("DeviceLinkThroughputLimitMode")); GenApi::IsWritable(m))
        m->FromString("Off");
    if (auto v = GenApi::CIntegerPtr(dMap->GetNode("DeviceLinkThroughputLimit")); GenApi::IsWritable(v))
    {
        int64_t hi = v->GetMax();
        int64_t lo = v->GetMin();
        v->SetValue(std::max<int64_t>(hi,lo));
    }

    //장치 이름 확인
    std::string modelname = "UNKNOWN";
    if(auto node = GenApi::CValuePtr(dMap->GetNode("DeviceModelName")); node && GenApi::IsReadable(node))
    {
        modelname = node->ToString();
    }

    auto setPF = [&](const char* name)->bool {
        auto pf = GenApi::CEnumerationPtr(dMap->GetNode("PixelFormat"));
        if (!pf || !GenApi::IsWritable(pf)) return false;
        auto e = pf->GetEntryByName(name);
        if (!e || !GenApi::IsAvailable(e) || !GenApi::IsReadable(e)) return false;
        pf->FromString(name);
        return true;
    };

    // 8) (2D 카메라 우선) 픽셀포맷 단순화 — 모노8
    if(modelname == "HTR003S-001")
    {
        bool ok = setPF("Coord3D_ABCY16");
    }
    else
    {
        if (auto pf = GenApi::CEnumerationPtr(dMap->GetNode("PixelFormat")); GenApi::IsWritable(pf))
        {
            //TRIO32S-CC 는 BGR8이 적용

            if (pf->GetEntryByName("BGR8"))
                pf->FromString("BGR8");
            //if(pf->GetEntryByName("Mono8"))
            //    pf->FromString("Mono8");
        }
    }

    try
    {
        dev_->StartStream();
    }
    catch (const std::exception& e)
    {
        emit errorOccurred(QString("StartStream: %1").arg(e.what())); // 원문 표시
        closeDevice();
        return;
    }
    catch (const GenICam::GenericException& e)
    {
        emit errorOccurred(QString("StartStream (GenICam): %1").arg(e.what()));
        std::cout << "StartStream (GenICam): " << e.what() << std::endl;
        closeDevice();
        return;
    }
    catch (...)
    {
        emit errorOccurred("StartStream: unknown exception");
        closeDevice();
        return;
    }

    while(running_)
    {
        try
        {
            auto* img = dev_->GetImage(200);
            if(!img) continue;
            /*
            if(!isToF_)
            {
                emit frameReady(camIdx_,toQImage2D(img));
            }
            else
            {
                emit errorOccurred("ToF heatmap not implemented (use Helios example logic).");
            }
            */

            onFrameRaw(img);
            //dev_->RequeueBuffer(img);
        }
        catch (...)
        {
            QThread::msleep(5);
        }
    }
    try { dev_->StopStream(); }
    catch (...) {}
    closeDevice();
}

void CaptureWorker::stop() { running_ = false; }

CameraWorker::CameraWorker(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CameraWorker)
{
    ui->setupUi(this);
    //"192.168.1.150";//HTR003S-001     //"192.168.1.151";//TRI032S-CC
    QString hint = "192.168.1.151";
    bool isToF = false;
    int camIdx = -1;
    if(hint == "192.168.1.151")
    {
        camIdx = 0;
        isToF = false;
    }
    else if(hint == "192.168.1.150")
    {

        camIdx = 1;
        isToF = true;
    }
    worker_ = new CaptureWorker(hint, isToF, camIdx);
    worker_->moveToThread(&thread_);
    connect(&thread_, &QThread::started, worker_, &CaptureWorker::start);
    connect(this, &CameraWorker::destroyed, worker_, &CaptureWorker::stop);
    connect(worker_, &CaptureWorker::frameReady, this, &CameraWorker::onFrame, Qt::QueuedConnection);
    connect(worker_, &CaptureWorker::errorOccurred, this, [this](const QString& m){
      statusBar()->showMessage(m, 3000);
    });

    connect(ui->btnStart,   &QPushButton::clicked, this, &CameraWorker::onStart);
    connect(ui->btnSnapshot,&QPushButton::clicked, this, &CameraWorker::onSnapshot);

}



CameraWorker::~CameraWorker()
{
    if (worker_) {
      worker_->stop();
      thread_.quit(); thread_.wait();
      delete worker_;
    }
    delete ui;
}

void CameraWorker::onStart()
{
    thread_.start();
}

void CameraWorker::onFrame(int camidx ,const QImage& img)
{
    if(img.isNull())return;
    lastFrame_ = img;
    if(camidx == 0)
    {
        ui->videoLabel->setPixmap(QPixmap::fromImage(img).scaled(
                                  ui->videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    else if(camidx == 1)
    {
        ui->videoLabel_2->setPixmap(QPixmap::fromImage(img).scaled(
                                   ui->videoLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

void CaptureWorker::onFrameRaw(Arena::IImage *img)
{
    if(!img)
        return;
    Arena::IImage* copy = nullptr;
    try
    {
        copy = Arena::ImageFactory::Copy(img);
    }
    catch(...)
    {

    }
    if(dev_) dev_->RequeueBuffer(img);
    img = nullptr; // 안전

    QImage qimg;
        bool ok = false;

        if (copy) {
            if (!isToF_) {
                // RGB 경로
                qimg = toQImage2D(copy);
                ok = !qimg.isNull();
            } else {
                // ToF 경로 (깊이 false-color)
                ok = ImageRenderHelper::makeDepthFalseColor(copy, /*min=*/500, /*max=*/1500, qimg);
                // 필요 시: invalid(0) → 검정 처리, 범위 밖 → 0 처리 등 내부에서 보정
            }
        }

        // 4) 복사본 해제
        if (copy) {
            try { Arena::ImageFactory::Destroy(copy); }
            catch (...) {}
            copy = nullptr;
        }

        // 5) 마지막 성공 프레임 캐시 & emit
        //  - static thread_local 로 워커(스레드)별 캐시
        static thread_local QImage lastOk;

        if (ok && !qimg.isNull()) {
            lastOk = qimg;
            emit frameReady(camIdx_, qimg);
        } else if (!lastOk.isNull()) {
            // 실패시에는 직전 프레임 유지(필요 없으면 이 가지 삭제 가능)
            emit frameReady(camIdx_, lastOk);
        }

}

void CameraWorker::onSnapshot()
{
  if (lastFrame_.isNull()) return;
  QString path = QFileDialog::getSaveFileName(this, "Save PNG",
      QDir::homePath() + "/shot_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".png",
      "PNG Images (*.png)");
  if (!path.isEmpty()) lastFrame_.save(path, "PNG");
}
