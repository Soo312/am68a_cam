
#include "cameraworker.h"
#include "ui_cameraworker.h"
#include "ImageRenderHelper.h"

#include <QFileDialog>
#include <QDateTime>
#include <QPixmap>
#include <QDebug>


//Arena
#include <Arena/ArenaApi.h>
#include <GenApi/GenApi.h>

#include <Base/GCException.h>
using namespace std;

CaptureWorker::CaptureWorker(QString hint,bool isToF,int camIdx, QObject* p)
    :QObject(p), hint_(std::move(hint)), isToF_(isToF),camIdx_(camIdx){}
CaptureWorker::~CaptureWorker(){stop();closeDevice();}

QImage CaptureWorker::bayerRG8ToRgbQImage(Arena::IImage *pImg)
{
    if(!pImg)
        return QImage();
    auto nodeMap = dev_ ? dev_->GetNodeMap() : nullptr;
    auto pfEnum  = nodeMap ? GenApi::CEnumerationPtr(nodeMap->GetNode("PixelFormat")) : nullptr;

        auto getVal = [&](const char* name)->uint64_t {
            if (!pfEnum) return 0;
            if (auto e = pfEnum->GetEntryByName(name);
                e && GenApi::IsAvailable(e) && GenApi::IsReadable(e))
                return (uint64_t)e->GetValue();
            return 0;
        };

        const uint64_t vRGB8     = getVal("RGB8");
        const uint64_t vBayerRG8 = getVal("BayerRG8");

        // 안전 확인: 진짜 BayerRG8인지
        if (!(vRGB8 && vBayerRG8) || pImg->GetPixelFormat() != vBayerRG8)
            return QImage();

        const int w = (int)pImg->GetWidth();
        const int h = (int)pImg->GetHeight();

        Arena::IImage* conv = nullptr;
        try
        {
            // SDK가 최적화된 디베이어 수행
            conv = Arena::ImageFactory::Convert(pImg, vRGB8);

            const int bpp  = (int)conv->GetBitsPerPixel();  // 기대: 24
            const int bypp = (bpp + 7) / 8;                 // 3
            const size_t filled = conv->GetSizeFilled();
            const int srcStride = (h > 0 && (filled % (size_t)h) == 0)
                                ? (int)(filled / (size_t)h)
                                : (w * bypp);

            const uchar* src = (const uchar*)conv->GetData();
            // ▼ 더블버퍼의 현재 작업 버퍼
                  QImage& work = visBuf_[visIdx_];
                  if (work.isNull() || work.size() != QSize(w, h) || work.format() != QImage::Format_RGB888) {
                      work = QImage(w, h, QImage::Format_RGB888);
                      if (work.isNull()) { Arena::ImageFactory::Destroy(conv); return QImage(); }
                  }

                  const size_t copyBytes = (size_t)w * (size_t)bypp;
                  for (int y = 0; y < h; ++y) {
                      memcpy(work.scanLine(y), src + (size_t)y * (size_t)srcStride, copyBytes);
                  }

                  Arena::ImageFactory::Destroy(conv);
                  return work;  // 작업 버퍼(주의: emit 시에는 copy로 분리)
              }
              catch (...) {
                  if (conv) { try { Arena::ImageFactory::Destroy(conv); } catch (...) {} }
                  return QImage();
              }
}

static inline QImage BayerRG8_ToGrayQImage(Arena::IImage* img)
{
    if (!img) return QImage();

    const int w    = (int)img->GetWidth();
    const int h    = (int)img->GetHeight();
    const int bpp  = (int)img->GetBitsPerPixel(); // BayerRG8 → 8
    if (w <= 0 || h <= 0 || bpp != 8) return QImage();

    const uchar* src = (const uchar*)img->GetData();
    const size_t filled = img->GetSizeFilled();

    // stride 계산: 패딩이 있으면 filled/height 사용
    const int srcStride =
        (h > 0 && (filled % (size_t)h) == 0)
        ? (int)(filled / (size_t)h)
        : w;  // tight fallback

    QImage out(w, h, QImage::Format_Grayscale8);
    if (out.isNull()) return QImage();

    const size_t copyBytes = (size_t)w; // bypp = 1
    for (int y = 0; y < h; ++y)
        memcpy(out.scanLine(y), src + (size_t)y * (size_t)srcStride, copyBytes);

    return out;
}

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

static inline QImage TightCopyToQImage_StrideAware(Arena::IImage* img)
{
    if (!img)
    {
        return QImage();
    }

    const int w    = static_cast<int>(img->GetWidth());
    const int h    = static_cast<int>(img->GetHeight());
    const int bpp  = static_cast<int>(img->GetBitsPerPixel()); // bits per pixel
    const int bypp = (bpp + 7) / 8;                            // bytes per pixel
    const uchar* src = reinterpret_cast<const uchar*>(img->GetData());
    const size_t filled = img->GetSizeFilled();

    // 소스의 실제 한 줄 길이(패딩 포함)
    const int srcStride =
        (h > 0 && (filled % static_cast<size_t>(h)) == 0)
        ? static_cast<int>(filled / static_cast<size_t>(h))
        : (w * bypp);

    QImage::Format fmt =
        (bypp == 1) ? QImage::Format_Grayscale8 :
        (bypp == 3) ? QImage::Format_BGR888 :
        (bypp == 4) ? QImage::Format_ARGB32 :
                      QImage::Format_Invalid;

    if (fmt == QImage::Format_Invalid)
    {
        return QImage();
    }

    QImage out(w, h, fmt);
    if (out.isNull())
    {
        return QImage();
    }

    // 행마다 소스는 srcStride씩 건너뛰고, 픽셀 유효분(w*bypp)만 복사
    const size_t copyBytes = static_cast<size_t>(w) * static_cast<size_t>(bypp);
    for (int y = 0; y < h; ++y)
    {
        const uchar* srcLine = src + static_cast<size_t>(y) * static_cast<size_t>(srcStride);
        uchar* dstLine = out.scanLine(y);
        memcpy(dstLine, srcLine, copyBytes);
    }

#if QT_VERSION < QT_VERSION_CHECK(5, 14, 0)
    if (bypp == 3)
    {
        // 소스가 BGR인데 Qt<5.14는 RGB888 → 스왑 필요
        out = out.rgbSwapped();
    }
#endif

    return out;
}

static inline int CalcStepBytes(Arena::IImage* img)
{
    const int w   = static_cast<int>(img->GetWidth());
    const int h   = static_cast<int>(img->GetHeight());
    const int bpp = static_cast<int>(img->GetBitsPerPixel()); // bits per pixel
    const int bypp = (bpp + 7) / 8; // ceil to bytes

    const size_t filled = img->GetSizeFilled(); // SDK가 채운 전체 바이트
    if (h > 0 && (filled % static_cast<size_t>(h)) == 0)
    {
        return static_cast<int>(filled / static_cast<size_t>(h)); // 실제 bytesPerLine
    }
    return w * bypp; // fallback
}



QImage CaptureWorker::toQImage2D(Arena::IImage *pImg)
{
    if (!pImg)
    {
        return QImage();
    }

    // 1) 우선 현재 버퍼를 stride-aware로 복사 (패딩 고려)
    QImage out = TightCopyToQImage_StrideAware(pImg);
    if (!out.isNull())
    {
        return out;
    }

    // 2) (옵션) 그래도 깨지면 32bpp(BGRA8)로 변환해서 다시 복사
    //    4바이트 정렬이라 훨씬 안정적
    Arena::IImage* conv = nullptr;
    try
    {
        auto nodeMap = dev_ ? dev_->GetNodeMap() : nullptr;
        uint64_t bgraVal = 0;

        if (nodeMap)
        {
            auto pf = GenApi::CEnumerationPtr(nodeMap->GetNode("PixelFormat"));
            if (pf)
            {
                // 이름은 기기마다 다를 수 있어 "BGRa8", "BGRA8Packed" 둘 다 시도
                if (auto e = pf->GetEntryByName("BGRa8"); e && GenApi::IsAvailable(e) && GenApi::IsReadable(e))
                {
                    bgraVal = static_cast<uint64_t>(e->GetValue());
                }
                else if (auto e2 = pf->GetEntryByName("BGRA8Packed"); e2 && GenApi::IsAvailable(e2) && GenApi::IsReadable(e2))
                {
                    bgraVal = static_cast<uint64_t>(e2->GetValue());
                }
            }
        }

        if (bgraVal == 0)
        {
            // 값 심볼 못 찾으면 표준 상수로도 시도 (없어도 예외 안 나게 try/catch)
            // Lucid에서는 Convert에 PFNC코드 사용 가능. 필요 시 유지/제거.
            // bgraVal = ARENA_PFNC_BGRa8; // (환경에 맞게 상수 정의 되어있다면)
        }

        if (bgraVal != 0)
        {
            conv = Arena::ImageFactory::Convert(pImg, bgraVal);
            QImage out2 = TightCopyToQImage_StrideAware(conv);
            Arena::ImageFactory::Destroy(conv);
            conv = nullptr;

            if (!out2.isNull())
            {
                return out2;
            }
        }
    }
    catch (...)
    {
        if (conv)
        {
            try { Arena::ImageFactory::Destroy(conv); } catch (...) {}
            conv = nullptr;
        }
    }

    return QImage();
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
        n->SetValue(1500);
    // 1440이 일반 안전값이지만, 스위치/보드 조합에 따라 1400이 더 확실할 때가 많다.
    // 그래도 안 되면 1300 → 1200 → 1000 순으로 내려가며 시도.

    // 4) 인터패킷 딜레이로 혼잡 완화 (필요 시 값 ↑)
    if (auto n = GenApi::CIntegerPtr(dMap->GetNode("GevSCPD")); GenApi::IsWritable(n))
        n->SetValue(40000);
    // 여전히 타임아웃이면 10000, 20000까지도 올려볼 수 있음 (나노초 단위인 경우 多).

    // 5) 패킷 재전송 켜기 (유실 대비)
    if (auto n = GenApi::CBooleanPtr(sMap->GetNode("StreamPacketResendEnable")); GenApi::IsWritable(n)) n->SetValue(true);

    // 6) 버퍼 처리 모드 (프레임 드롭 시 최신 프레임 유지)
    if (auto n = GenApi::CEnumerationPtr(sMap->GetNode("StreamBufferHandlingMode")); GenApi::IsWritable(n))
        n->FromString("NewestOnly");

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


    // Enable 먼저
    if (auto en = GenApi::CBooleanPtr(dMap->GetNode("AcquisitionFrameRateEnable"));
        en && GenApi::IsWritable(en)) {
        en->SetValue(true);
    }

    // FrameRate 설정 (있을 때만)
    if (auto fr = GenApi::CFloatPtr(dMap->GetNode("AcquisitionFrameRate"));
        fr && GenApi::IsWritable(fr)) {
        double min = fr->GetMin();
        double max = fr->GetMax();
        double target = 14;
        if (target < min) target = min;
        if (target > max) target = max;
        fr->SetValue(target);
        qDebug() << "FrameRate set to" << target;
    } else {
        qDebug() << "AcquisitionFrameRate not available on this model.";
    }

    try
    {
        if (auto en = GenApi::CBooleanPtr(sMap->GetNode("StreamBufferCountManual"));
            en && GenApi::IsWritable(en))
        {
            en->SetValue(true);
        }

        if (auto cnt = GenApi::CIntegerPtr(sMap->GetNode("StreamBufferCount"));
            cnt && GenApi::IsWritable(cnt))
        {
            int64_t lo = cnt->GetMin();
            int64_t hi = cnt->GetMax();
            int64_t want = 10;
            if (want < lo) want = lo;
            if (want > hi) want = hi;
            cnt->SetValue(want);
        }
    }
    catch (...) {}

    try
    {
        if (auto n = GenApi::CBooleanPtr(dMap->GetNode("ChunkModeActive"));
            n && GenApi::IsWritable(n))
        {
            n->SetValue(false);
        }

        // 혹시 개별 Chunk 항목 켜져 있으면 전부 Off (있을 때만)
        const char* chunkKeys[] = {
            "ChunkEnable", "ChunkTimestampEnable", "ChunkFrameIDEnable",
            "ChunkExposureTimeEnable", "ChunkGainEnable"
        };
        for (auto key : chunkKeys)
        {
            if (auto c = GenApi::CBooleanPtr(dMap->GetNode(key));
                c && GenApi::IsWritable(c))
            {
                c->SetValue(false);
            }
        }
    }
    catch (...) {}

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

    // --- Stream(TLStream) ---
    if (auto n = GenApi::CBooleanPtr(sMap->GetNode("StreamAutoNegotiatePacketSize"));
        n && GenApi::IsWritable(n))
    {
        n->SetValue(true); // 기존 true -> false 필수
    }

    if (auto n = GenApi::CBooleanPtr(sMap->GetNode("StreamPacketResendEnable"));
        n && GenApi::IsWritable(n))
    {
        n->SetValue(true);
    }

    if (auto n = GenApi::CEnumerationPtr(sMap->GetNode("StreamBufferHandlingMode"));
        n && GenApi::IsWritable(n))
    {
        n->FromString("NewestOnly");
    }

    // 버퍼 수 수동 설정 + 12~16부터 시작 (지연↓, 필요시 24)
    if (auto en = GenApi::CBooleanPtr(sMap->GetNode("StreamBufferCountManual"));
        en && GenApi::IsWritable(en))
    {
        en->SetValue(true);
    }
    if (auto cnt = GenApi::CIntegerPtr(sMap->GetNode("StreamBufferCount"));
        cnt && GenApi::IsWritable(cnt))
    {
        int64_t want = 16; // 12~16 권장 (불안하면 24)
        if (want < cnt->GetMin()) want = cnt->GetMin();
        if (want > cnt->GetMax()) want = cnt->GetMax();
        cnt->SetValue(want);
    }

    // --- Device ---
    if (auto n = GenApi::CIntegerPtr(dMap->GetNode("GevSCPSPacketSize"));
        n && GenApi::IsWritable(n))
    {
        try
        {
            n->SetValue(1300); // 안되면 1400 -> 1300
        }
        catch (...)
        {
            n->SetValue(1200);
        }
    }

    if (auto n = GenApi::CIntegerPtr(dMap->GetNode("GevSCPD"));
        n && GenApi::IsWritable(n))
    {
        n->SetValue(30000); // 15µs 시작 (지지직이면 20000까지)
    }

    // 링크 스루풋 제한: bytes/sec 단위 (예: 350 Mbps → 43,750,000 B/s)
    if (auto mode = GenApi::CEnumerationPtr(dMap->GetNode("DeviceLinkThroughputLimitMode"));
        mode && GenApi::IsWritable(mode))
    {
        mode->FromString("Off");
    }
    if (auto v = GenApi::CIntegerPtr(dMap->GetNode("DeviceLinkThroughputLimit"));
        v && GenApi::IsWritable(v))
    {
        /*
        const double wantMbps = 250.0; // 300~450 사이에서 테스트
        int64_t wantBytesPerSec = static_cast<int64_t>((wantMbps * 1'000'000.0) / 8.0);
        if (wantBytesPerSec > v->GetMax()) wantBytesPerSec = v->GetMax();
        if (wantBytesPerSec < v->GetMin()) wantBytesPerSec = v->GetMin();
        v->SetValue(wantBytesPerSec);
*/
        v->SetValue(115000000);
    }

    // 8) (2D 카메라 우선) 픽셀포맷 단순화 — 모노8
    if(modelname == "HTR003S-001")
    {
        //2D
        bool ok = setPF("Coord3D_C16"); //Coord3D_C16 은 2D에서 HeatMap만 보여줄때 적절 Coord3D_ABCY16은 3D 모델링할때 적절
        if (!ok) ok = setPF("Range");       // 또는 "Coord3D_Z16", "Confidence16" 등 장치 메뉴 확인
        if (!ok) ok = setPF("Coord3D_C16"); // 반복 시도 가능
        //3D 인데 프레임끊김이 좀심함
        /*
        bool ok = setPF("Coord3D_ABCY16"); //Coord3D_C16 은 2D에서 HeatMap만 보여줄때 적절 Coord3D_ABCY16은 3D 모델링할때 적절
        if (!ok) ok = setPF("Range");       // 또는 "Coord3D_Z16", "Confidence16" 등 장치 메뉴 확인
        if (!ok) ok = setPF("Coord3D_ABCY16"); // 반복 시도 가능
        */

        auto w = GenApi::CIntegerPtr(dMap->GetNode("Width"));
        auto h = GenApi::CIntegerPtr(dMap->GetNode("Height"));
        if (w && GenApi::IsWritable(w)) w->SetValue(640);
        if (h && GenApi::IsWritable(h)) h->SetValue(480);
    }
    else
    {
        if (auto pf = GenApi::CEnumerationPtr(dMap->GetNode("PixelFormat")); GenApi::IsWritable(pf))
        {
            //TRIO32S-CC 는 BayerRG8 적용

            if (pf->GetEntryByName("BayerRG8"))
                pf->FromString("BayerRG8");
        }
        auto w = GenApi::CIntegerPtr(dMap->GetNode("Width"));
        auto h = GenApi::CIntegerPtr(dMap->GetNode("Height"));
        //if (w && GenApi::IsWritable(w)) w->SetValue(2048);
        //if (h && GenApi::IsWritable(h)) h->SetValue(1500);

        auto width = w->GetValue("Width");
        auto height = h->GetValue("Height");

                qDebug() << "width : "<< width << " , height : " << height <<endl;
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

            /* Debug Log
            static int frameCounter = 0;
                   if ((++frameCounter % 30) == 0)
                   {   // FPS≈30일 때 30프레임=1초 주기
                       auto pf = GenApi::CEnumerationPtr(dev_->GetNodeMap()->GetNode("PixelFormat"));
                       QString pfName = "N/A";
                       if (pf && GenApi::IsReadable(pf))
                       {
                           GenICam::gcstring gc = pf->ToString();
                           pfName = QString::fromLatin1(gc.c_str());   // ✅ 안전 변환
                       }

                       auto s = dev_->GetTLStreamNodeMap();
                       auto delivered = GenApi::CIntegerPtr(s->GetNode("StreamDeliveredFrameCount"));
                       auto dropped   = GenApi::CIntegerPtr(s->GetNode("StreamLostFrameCount"));

                       qDebug() << "[PF]" << pfName
                                << "delivered=" << (delivered && GenApi::IsReadable(delivered) ? (long long)delivered->GetValue() : -1)
                                << "dropped="   << (dropped   && GenApi::IsReadable(dropped)   ? (long long)dropped->GetValue()   : -1);
                   }
             */
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

    ui->videoLabel->setScaledContents(true);
    ui->videoLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui->videoLabel->setAlignment(Qt::AlignCenter);

    ui->videoLabel_2->setScaledContents(true);
    ui->videoLabel_2->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui->videoLabel_2->setAlignment(Qt::AlignCenter);


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
    if (!thread_.isRunning())
    {
        thread_.start();
    }
}


void CameraWorker::onFrame(int camidx ,const QImage& img)
{
    if(img.isNull())return;
    lastFrame_ = img;
    if(camidx == 0)
    {
        /*
        ui->videoLabel->setPixmap(QPixmap::fromImage(img).scaled(
                                  ui->videoLabel->size(),
                                      Qt::KeepAspectRatio,
                                      Qt::FastTransformation));
                                      */
        ui->videoLabel->setPixmap(QPixmap::fromImage(img));
    }

    else if(camidx == 1)
    {
        /*
        ui->videoLabel_2->setPixmap(QPixmap::fromImage(img).scaled(
                                   ui->videoLabel_2->size(),
                                        Qt::KeepAspectRatio,
                                        Qt::FastTransformation));
                                        */
        ui->videoLabel_2->setPixmap(QPixmap::fromImage(img));
    }
}

void CaptureWorker::onFrameRaw(Arena::IImage *img)
{
    if (!img)
         return;

     // 1) 복사
     Arena::IImage* copy = nullptr;
     try { copy = Arena::ImageFactory::Copy(img); } catch (...) {}

     // 2) 원본 즉시 반납
     if (dev_) dev_->RequeueBuffer(img);
     img = nullptr;

     // 3) 변환
     QImage qimg;
     bool ok = false;

     if (copy)
     {
         if (!isToF_)
         {
             // 가시광: BayerRG8로 수신 후 컬러화
             qimg = bayerRG8ToRgbQImage(copy);
             if (qimg.isNull())
             {
                 // 혹시 현재가 이미 RGB/BGR8인 경우 등: 기존 타이트 카피로 폴백
                 qimg = TightCopyToQImage_StrideAware(copy);
             }
             ok = !qimg.isNull();
         }
         else
         {
             // ToF: 기존 false color
             ok = ImageRenderHelper::makeDepthFalseColor(copy, 300, 6000, qimg);
         }

         try { Arena::ImageFactory::Destroy(copy); } catch (...) {}
         copy = nullptr;
     }

     // 4) emit
     static thread_local QImage lastOk;
     if (ok && !qimg.isNull())
     {
         lastOk = qimg;
         emit frameReady(camIdx_, qimg.copy());

         visIdx_ ^= 1;
     }
     /*
     else if (!lastOk.isNull())
     {
         emit frameReady(camIdx_, lastOk);
     }
     */
}

void CameraWorker::onSnapshot()
{
  if (lastFrame_.isNull()) return;
  QString path = QFileDialog::getSaveFileName(this, "Save PNG",
      QDir::homePath() + "/shot_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".png",
      "PNG Images (*.png)");
  if (!path.isEmpty()) lastFrame_.save(path, "PNG");
}
