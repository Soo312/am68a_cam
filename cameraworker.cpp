
#include "cameraworker.h"
#include "ui_cameraworker.h"
#include "ImageRenderHelper.h"

#include <QFileDialog>
#include <QDateTime>
#include <QPixmap>
#include <QDebug>
#include <QTimer>
#include <QPainter>
#include <QKeyEvent>


//Arena
#include <Arena/ArenaApi.h>
#include <GenApi/GenApi.h>

#include <Base/GCException.h>

#include "pose_estimate_onnx.h"
#include "person_detect_onnx.h"

using namespace std;

// COCO 17 keypoints 연결 (필요에 맞게 수정 가능)
static const int EDGES[][2] = {
    {5,7},{7,9},   // left arm: shoulder-elbow-wrist
    {6,8},{8,10},  // right arm
    {11,13},{13,15}, // left leg: hip-knee-ankle
    {12,14},{14,16}, // right leg
    {5,6}, {11,12},  // shoulders, hips
    {5,11}, {6,12}   // torso diagonals (선택)
};
static constexpr int EDGE_COUNT = sizeof(EDGES)/sizeof(EDGES[0]);

static QImage mono16ToGray16_QImage(Arena::IImage* img)
{
    if (!img || img->GetBitsPerPixel() != 16) return QImage();

    const int w = (int)img->GetWidth();
    const int h = (int)img->GetHeight();

    const int bpp  = (int)img->GetBitsPerPixel();  // 기대: 24
    const int bypp = (bpp + 7) / 8;                 // 3
    const size_t filled = img->GetSizeFilled();
    const int srcStride = (h > 0 && (filled % (size_t)h) == 0)
                        ? (int)(filled / (size_t)h)
                        : (w * bypp);

    const uint8_t* base = static_cast<const uint8_t*>(img->GetData());

#if QT_VERSION >= QT_VERSION_CHECK(5, 13, 0)
    QImage out(w, h, QImage::Format_Grayscale16);
    const int dstBpl = out.bytesPerLine();            // 보통 w*2
    for (int y = 0; y < h; ++y)
    {
        const void* srcLine = base + (size_t)y * srcStride;
        void*       dstLine = out.scanLine(y);
        // 픽셀 유효 바이트(w*2)만 복사 (dstBpl이 더 크면 남는 건 0 유지)
        std::memcpy(dstLine, srcLine, (size_t)w * 2);
        // 필요시: 남는 바이트가 있으면 0으로 채우기 (생략 가능)
        // if (dstBpl > w*2) std::memset((uint8_t*)dstLine + w*2, 0, dstBpl - w*2);
    }
    return out;
#else
    // Qt 5.12 이하: 16비트를 그대로 보여줄 수 없으므로 8비트로 단순 축소(손실)
    QImage out(w, h, QImage::Format_Grayscale8);
    for (int y = 0; y < h; ++y)
    {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(base + (size_t)y * srcBpl);
        uint8_t* dst = out.scanLine(y);
        for (int x = 0; x < w; ++x) dst[x] = uint8_t(src[x] >> 8); // 상위바이트만 (간단)
    }
    return out;
#endif
}

static inline Letterbox makeLetterbox(int srcW, int srcH, int netW, int netH)
{
    Letterbox lb{};
    float rW = float(netW) / srcW;
    float rH = float(netH) / srcH;
    lb.scale = std::min(rW, rH);
    lb.padX  = (netW - srcW * lb.scale) * 0.5f;
    lb.padY  = (netH - srcH * lb.scale) * 0.5f;
    lb.netW  = netW;
    lb.netH  = netH;
    return lb;
}

static inline QPointF unletterboxPoint(const QPointF& pNet, const Letterbox& lb)
{
    return QPointF((pNet.x() - lb.padX) / lb.scale,
                   (pNet.y() - lb.padY) / lb.scale);
}

static inline QRectF unletterboxRect(const QRectF& rNet, const Letterbox& lb)
{
    QPointF tl = unletterboxPoint(rNet.topLeft(), lb);
    QPointF br = unletterboxPoint(rNet.bottomRight(), lb);
    return QRectF(tl, br);
}



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
        //auto* sys = Arena::OpenSystem();
        //sys_ = sys;
        if(!sys_)
        {
            emit errorOccurred("System not set");
            return false;
        }

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
    //if (sys_) { try { Arena::CloseSystem(sys_); } catch (...) {} sys_ = nullptr; }
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

void CaptureWorker::setSystem(Arena::ISystem *s)
{
    sys_ = s;
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
        double target = 15;
        if (target < min) target = min;
        if (target > max) target = max;
        fr->SetValue(target);
        qDebug() << "FrameRate set to" << target;
    } else {
        qDebug() << "AcquisitionFrameRate not available on this model.";
    }

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
         int64_t want = isToF_ ? 16 : 12;
         want = std::clamp(want, cnt->GetMin(), cnt->GetMax());
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
        GenApi::CEnumerationPtr op = dMap->GetNode("Scan3dOperatingMode");

        if (GenApi::IsReadable(op->GetEntryByName("Distance8300mmMultiFreq")))
            op->FromString("Distance8300mmMultiFreq");

        auto w = GenApi::CIntegerPtr(dMap->GetNode("Width"));
        auto h = GenApi::CIntegerPtr(dMap->GetNode("Height"));
        if (w && GenApi::IsWritable(w)) w->SetValue(640);
        if (h && GenApi::IsWritable(h)) h->SetValue(480);

        GenApi::CEnumerationPtr sel = dMap->GetNode("Scan3dCoordinateSelector");
        GenApi::CFloatPtr sc = dMap->GetNode("Scan3dCoordinateScale");
        GenApi::CFloatPtr of = dMap->GetNode("Scan3dCoordinateOffset");
        if (sel && sc && of) {
            sel->FromString("CoordinateA");
            scaleX_ = (float)sc->GetValue();
            offX_ = (float)of->GetValue();
            sel->FromString("CoordinateB");
            scaleY_ = (float)sc->GetValue();
            offY_ = (float)of->GetValue();
            sel->FromString("CoordinateC");
            scaleZ_ = (float)sc->GetValue();
            offZ_ = (float)of->GetValue();
        }

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
            auto* img = dev_->GetImage(100);
            if(!img) continue;

            onFrameRaw(img);

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
    QThread::currentThread()->setObjectName("UIThread");

    ui->setupUi(this);

    ui->videoLabel->setScaledContents(true);
    ui->videoLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui->videoLabel->setAlignment(Qt::AlignCenter);

    ui->videoLabel_2->setScaledContents(true);
    ui->videoLabel_2->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui->videoLabel_2->setAlignment(Qt::AlignCenter);


    //"192.168.1.150";//HTR003S-001     //"192.168.1.151";//TRI032S-CC

    /*
    QString hint = "192.168.1.150";
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
    */
    sys_ = Arena::OpenSystem();

    vis_worker_ = new CaptureWorker("192.168.1.151", /*isToF=*/false, /*camIdx=*/0);
    tof_worker_  = new CaptureWorker("192.168.1.150", /*isToF=*/true,  /*camIdx=*/1);
    pose_worker_ = new PoseWorker();

    pose_worker_->moveToThread(&pose_thread_);
    pose_thread_.start();

    vis_worker_->moveToThread(&vis_thread_);
    connect(&vis_thread_, &QThread::started, vis_worker_, &CaptureWorker::start);
    connect(this, &CameraWorker::destroyed, vis_worker_, &CaptureWorker::stop);
    connect(vis_worker_, &CaptureWorker::frameReady, this,
            &CameraWorker::onFrame, Qt::QueuedConnection);
    connect(vis_worker_, &CaptureWorker::errorOccurred, this, [this](const QString& m){
      statusBar()->showMessage(m, 3000);
    });

    tof_thread_.setObjectName("ToFThread");
    tof_worker_->moveToThread(&tof_thread_);
    connect(&tof_thread_, &QThread::started, tof_worker_, &CaptureWorker::start);
    connect(this, &CameraWorker::destroyed, tof_worker_, &CaptureWorker::stop);
    connect(tof_worker_, &CaptureWorker::frameReady, this, &CameraWorker::onFrameReady, Qt::QueuedConnection);
    connect(tof_worker_, &CaptureWorker::errorOccurred, this, [this](const QString& m){
      statusBar()->showMessage(m, 3000);
    });
    //connect(this, &CameraWorker::requestSnapPose, tof_worker_,
    //        &CaptureWorker::requestSnap, Qt::DirectConnection);
    connect(tof_worker_, &CaptureWorker::poseRequest,
            pose_worker_, &PoseWorker::snapPose,
            Qt::QueuedConnection);





    pose_worker_->pPose_ = pPose_;
    pose_worker_->pose_ = new PoseEstimatorONNX();


    connect(&pose_thread_, &QThread::finished, pose_worker_,
            &QObject::deleteLater);


    qRegisterMetaType<QImage>("QImage");

    connect(this, &CameraWorker::requestPoseImage,
            pose_worker_, &PoseWorker::snapPose,
            Qt::QueuedConnection);

    //connect(this, &CameraWorker::onSnapPose,
    //        pose_worker_, &PoseWorker::snapPose, Qt::QueuedConnection);

    connect(pose_worker_, &PoseWorker::poseDone,
            this, &CameraWorker::onPoseDone,
            Qt::QueuedConnection);


    vis_worker_->setSystem(sys_);
    tof_worker_->setSystem(sys_);
    //AI 파라미터
    {
        if (!pose_) pose_ = new PoseEstimatorONNX();

        //추가: 캡처 폴더 준비
        QDir().mkpath(captureDir_);

        //추가: 포즈 파라미터 & 초기화
        pPose_.inputW     = 640;
        pPose_.inputH     = 640;
        pPose_.letterbox  = true;
        pPose_.confDet    = 0.25f;   // bbox/obj 임계
        pPose_.confKpt    = 0.25f;   // 키포인트 표시 임계
        pPose_.nmsIoU     = 0.50f;
        pPose_.rgbInput   = true;    // Gray8 → RGB(복제)로 넣을 것

        poseReady_ = pose_->init("/home/CameraWorker/models/yolo11m-pose.onnx", pPose_, /*threads*/2);
    }
    connect(ui->btnCapture, &QPushButton::clicked, this, &CameraWorker::requestCapture);

    connect(ui->btnStart,   &QPushButton::clicked, this, &CameraWorker::onStart);
    connect(ui->btnSnapshot,&QPushButton::clicked, this, &CameraWorker::onSnapshot);

}



CameraWorker::~CameraWorker()
{
    if (tof_worker_)
    {
      tof_worker_->stop();
      tof_thread_.quit();
      tof_thread_.wait();
      delete tof_worker_;
    }
    if (vis_worker_)
    {
      vis_worker_->stop();
      vis_thread_.quit();
      vis_thread_.wait();
      delete vis_worker_;
    }
    pose_thread_.quit();
    pose_thread_.wait();
    delete ui;
}

void CameraWorker::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_K)
    {
        qDebug() << "K key pressed";
        emit requestCapture();  // 또는 너의 캡처 로직 직접 호출
    }
    else
    {
        QMainWindow::keyPressEvent(event); // 다른 키는 기본 처리
    }
}

static QString tsNow()
{
    return QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss_zzz");
}

void CameraWorker::doCaptureAndPose()
{
    if (lastFrame_.isNull())
    {
        qWarning() << "[K] lastFrame_ is null. Skip.";
        return;
    }

    // 1. 캡처 디렉토리 준비
    QDir().mkpath(captureDir_);

    // 2. 파일 경로
    const QString base     = captureDir_ + "/" + tsNow();
    const QString rawPath  = base + "_raw.png";
    const QString posePath = base + "_pose.png";

    // 3. 이미지 저장
    if (lastFrame_.save(rawPath))
        qInfo() << "[K] saved raw:" << rawPath;
    else
        qWarning() << "[K] save raw failed:" << rawPath;

    // 4. 이미지 포맷 확인
    QImage inferImg = lastFrame_;
    if (pPose_.rgbInput)
    {
        if (inferImg.format() != QImage::Format_RGB888)
            inferImg = inferImg.convertToFormat(QImage::Format_RGB888);
    }
    else
    {
        if (inferImg.format() != QImage::Format_Grayscale8)
            inferImg = inferImg.convertToFormat(QImage::Format_Grayscale8);
    }

    // 5. 포즈 추론
    std::vector<PosePerson> persons;
    bool ok = pose_->infer(
        inferImg.bits(),
        inferImg.width(),
        inferImg.height(),
        inferImg.bytesPerLine(),
        persons
    );

    {
        // 원본(src) = qimgIn 크기, 네트(net) = pPose_.inputW/H (정사각)
        const Letterbox lb = makeLetterbox(lastFrame_.width(), lastFrame_.height(),
                                           pPose_.inputW, pPose_.inputH);

        // (선택) 모델이 이미 원본 좌표로 돌려주는지 빠르게 감지하고 싶다면:
        // bool looksNet = false;
        // for (auto& per : persons) {
        //     if (per.x > qimgIn.width()+8 || per.y > qimgIn.height()+8) { looksNet = true; break; }
        // }
        // if (!looksNet) { /* 이미 원본 좌표면 아래 보정 블록을 건너뛰어도 됨 */ }

        // 1) bbox/kpts 언레터박스
        for (auto& per : persons)
        {
            QRectF rNet(per.x, per.y, per.w, per.h);
            const QRectF rSrc = unletterboxRect(rNet, lb);
            per.x = rSrc.x(); per.y = rSrc.y();
            per.w = rSrc.width(); per.h = rSrc.height();

            for (auto& k : per.kpts)
            {
                const QPointF ps = unletterboxPoint(QPointF(k.x, k.y), lb);
                k.x = ps.x(); k.y = ps.y();
            }
        }

        // 2) "손 위" 강화: 박스 상단 20% 확장(프레임 밖 오버런 방지)
        const float topGrow = 0.20f;
        const float W = float(lastFrame_.width());
        const float H = float(lastFrame_.height());
        for (auto& per : persons)
        {
            per.y = std::max(0.0f, per.y - per.h * topGrow);
            per.h = std::min(H - per.y, per.h * (1.0f + topGrow));
            per.x = std::clamp(per.x, 0.0f, std::max(0.0f, W - 1.0f));
            per.w = std::min(W - per.x, per.w);
        }
    }

    if (!ok)
    {
        qWarning() << "[K] pose infer failed.";
        return;
    }

    qInfo() << "[K] persons =" << persons.size();

    // 6. 스켈레톤 오버레이 그리기
    QImage vis = lastFrame_.convertToFormat(QImage::Format_RGB888);

    {
        QPainter p(&vis);
        p.setRenderHint(QPainter::Antialiasing, true);
        QPen pen(Qt::green); pen.setWidth(3); p.setPen(pen);
        QFont f = p.font(); f.setPointSize(10); p.setFont(f);

        for (size_t pi = 0; pi < persons.size(); ++pi)
        {
            const auto& person = persons[pi];
            const auto& kpts = person.kpts; // 네 구조에 맞게 조정 필요

            // 연결 선
            for (int i = 0; i < EDGE_COUNT; ++i)
            {
                int a = EDGES[i][0], b = EDGES[i][1];
                if (a < (int)kpts.size() && b < (int)kpts.size()
                    && kpts[a].conf > 0.2f && kpts[b].conf > 0.2f)
                {
                    p.drawLine(QPointF(kpts[a].x, kpts[a].y),
                               QPointF(kpts[b].x, kpts[b].y));
                }
            }

            // 관절 점
            for (const auto& k : kpts)
            {
                if (k.conf > 0.2f)
                    p.drawEllipse(QPointF(k.x, k.y), 3, 3);
            }

            // (선택) 바운딩 박스 그리기
            // if (person.bbox.width > 0)
            // {
            //     QRectF rc(person.bbox.x, person.bbox.y,
            //              person.bbox.width, person.bbox.height);
            //     p.drawRect(rc);
            //     p.drawText(rc.topLeft() + QPointF(2,-2), QString("person %1").arg(pi));
            // }
        }
    }

    // 7. 결과 이미지 저장
    if (vis.save(posePath))
        qInfo() << "[K] saved pose overlay:" << posePath;
    else
        qWarning() << "[K] save pose failed:" << posePath;

    // 8. 뷰 갱신 (선택)
    // lastFrame_ = vis;
    // emit frameReady(-1, vis);
}

void CameraWorker::onStart()
{

    if (!tof_thread_.isRunning())
        tof_thread_.start();
    /*QTimer::singleShot(600, this, [this]{
        if (!vis_thread_.isRunning())
            vis_thread_.start();
    });*/
   //vis_thread_.start();

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

void CameraWorker::onPoseDone(const QImage &img)
{
    lastFrame_ = img;
    emit frameReady(1,img);
}

void CameraWorker::handleTermKey(char ch)
{
    switch (ch)
    {
        case 'k': case 'K':
            //doCaptureAndPose();
            requestCapture();
            emit requestSnapPose();
            emit requestPoseImage(lastFrame_.copy());
    case 'v' : case 'V':

        break;
        break;
            case 'q' : case 'Q':
            //1분 캡처+ai넣어야할곳

            break;


        case 'x': case 'X':
            QCoreApplication::quit();
            return;
        default:
            return;
    }
}

void CameraWorker::requestCapture()
{
    capturePending_.store(true, std::memory_order_release);
}



void CameraWorker::onFrameReady(int camIdx, const QImage &qimgIn)
{
    // 1) 평소처럼 UI로 먼저 전달(원하면 순서 바꿔도 OK)
    lastFrame_ = qimgIn;

    emit frameReady(camIdx, qimgIn);
    onFrame(1,qimgIn);


    //emit savedCapture(annPath);
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

            float rMin = 0.6;
            float rMax = 8.0;
            float zScale = scaleZ_ / 1000;
            uint16_t zMin = 0.6 / zScale;
            uint16_t zMax = 8.0 / zScale;



             ok = ImageRenderHelper::makeDepthFalseColor(copy, zMin, zMax, qimg);

             QImage gimg;
             //ImageRenderHelper::depthC16ToGray8(copy,zMin,zMax,1.0,true,gimg);

            if(!qimg.isNull())
            {
                emit frameReady(camIdx_, qimg);

                if (snapPending_.exchange(false, std::memory_order_acq_rel))
                {
                    //snapPose(qimg);
                    //snapPose(gimg);

                    //emit poseRequest(qimg);
                    //emit poseRequest(gimg);
                }


                visIdx_ ^= 1;
            }
         }

         try { Arena::ImageFactory::Destroy(copy); } catch (...) {}
         copy = nullptr;
     }

     // 4) emit
     static thread_local QImage lastOk;
     if (ok && !qimg.isNull())
     {
         lastOk = qimg;

     }

}

void CaptureWorker::captureOnceC16()
{
    captureOnceFlag_.store(true);
}

void CameraWorker::onSnapshot()
{
  if (lastFrame_.isNull()) return;
  QString path = QFileDialog::getSaveFileName(this, "Save PNG",
      QDir::homePath() + "/shot_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".png",
      "PNG Images (*.png)");
  if (!path.isEmpty()) lastFrame_.save(path, "PNG");
}

PoseWorker::PoseWorker(QObject *parent)
    : QObject(parent)
{

}



void PoseWorker::snapPose(const QImage& inImg)
{
    qInfo() << "[snapPose] thread=" << QThread::currentThread()
            << " id=" << (quintptr)QThread::currentThreadId()
            << " name=" << QThread::currentThread()->objectName();

    //QImage gray;
    //ImageRenderHelper::depthC16ToGray8(pImg, /*zMin*/ zMin, /*zMax*/ zMax,1.0,true, gray);
    //gray = mono16ToGray16_QImage(pImg);

    if (inImg.isNull())
    {
        qWarning() << "[SnapPose] depthC16ToGray8 failed";
        return;
    }

    //AI 파라미터
    {
        if (!pose_) pose_ = new PoseEstimatorONNX();

        //추가: 캡처 폴더 준비
        QDir().mkpath(captureDir_);

        //추가: 포즈 파라미터 & 초기화
        pPose_.inputW     = 640;
        pPose_.inputH     = 640;
        pPose_.letterbox  = true;
        pPose_.confDet    = 0.20f;   // bbox/obj 임계
        pPose_.confKpt    = 0.20f;   // 키포인트 표시 임계
        pPose_.nmsIoU     = 0.55f;
        pPose_.rgbInput   = true;    // Gray8 → RGB(복제)로 넣을 것

        poseReady_ = pose_->init("/home/CameraWorker/models/yolo11m-pose.onnx", pPose_, /*threads*/2);
    }

    // 1. 캡처 디렉토리 준비
        QDir().mkpath(captureDir_);

        // 2. 파일 경로
        const QString base     = captureDir_ + "/" + tsNow();
        const QString rawPath  = base + "_raw2.png";
        const QString posePath = base + "_pose2.png";

        // 3. 이미지 저장 (원본 Gray)
        if (inImg.save(rawPath))
            qInfo() << "[K] saved raw:" << rawPath;
        else
            qWarning() << "[K] save raw failed:" << rawPath;

        // 4. 이미지 포맷 확인 (모델 입력용)
        QImage inferImg = inImg;
        if (pPose_.rgbInput)
        {
            // 모델이 RGB를 기대하면 Gray→RGB888(채널 반복)
            if (inferImg.format() != QImage::Format_RGB888)
                inferImg = inferImg.convertToFormat(QImage::Format_RGB888);
        }
        else
        {
            if (inferImg.format() != QImage::Format_Grayscale8)
                inferImg = inferImg.convertToFormat(QImage::Format_Grayscale8);
        }

        // 5. 포즈 추론
        std::vector<PosePerson> persons;
        bool ok = (pose_ != nullptr) ? pose_->infer(
                        inferImg.bits(),
                        inferImg.width(),
                        inferImg.height(),
                        inferImg.bytesPerLine(),
                        persons)
                                     : false;



        {
            // 원본(src) = qimgIn 크기, 네트(net) = pPose_.inputW/H (정사각)
            const Letterbox lb = makeLetterbox(inImg.width(), inImg.height(),
                                               pPose_.inputW, pPose_.inputH);

            // (선택) 모델이 이미 원본 좌표로 돌려주는지 빠르게 감지하고 싶다면:
            // bool looksNet = false;
            // for (auto& per : persons) {
            //     if (per.x > qimgIn.width()+8 || per.y > qimgIn.height()+8) { looksNet = true; break; }
            // }
            // if (!looksNet) { /* 이미 원본 좌표면 아래 보정 블록을 건너뛰어도 됨 */ }

            // 1) bbox/kpts 언레터박스
            for (auto& per : persons)
            {
                QRectF rNet(per.x, per.y, per.w, per.h);
                const QRectF rSrc = unletterboxRect(rNet, lb);
                per.x = rSrc.x(); per.y = rSrc.y();
                per.w = rSrc.width(); per.h = rSrc.height();

                for (auto& k : per.kpts)
                {
                    const QPointF ps = unletterboxPoint(QPointF(k.x, k.y), lb);
                    k.x = ps.x(); k.y = ps.y();
                }
            }

            // 2) "손 위" 강화: 박스 상단 20% 확장(프레임 밖 오버런 방지)
            const float topGrow = 0.20f;
            const float W = float(inImg.width());
            const float H = float(inImg.height());
            for (auto& per : persons)
            {
                per.y = std::max(0.0f, per.y - per.h * topGrow);
                per.h = std::min(H - per.y, per.h * (1.0f + topGrow));
                per.x = std::clamp(per.x, 0.0f, std::max(0.0f, W - 1.0f));
                per.w = std::min(W - per.x, per.w);
            }
        }

        if (!ok)
        {
            qWarning() << "[K] pose infer failed.";
            return;
        }

        qInfo() << "[K] persons =" << persons.size();

        // 6. 스켈레톤 오버레이 그리기 (시각화는 Gray 위에 RGB로)
        QImage vis = inImg.convertToFormat(QImage::Format_RGB888);
        {
            QPainter p(&vis);
            p.setRenderHint(QPainter::Antialiasing, true);
            QPen pen(Qt::green);
            pen.setWidth(3);
            p.setPen(pen);
            QFont f = p.font();
            f.setPointSize(10);
            p.setFont(f);

            for (size_t pi = 0; pi < persons.size(); ++pi)
            {
                const auto& person = persons[pi];
                const auto& kpts = person.kpts;

                for (int i = 0; i < EDGE_COUNT; ++i)
                {
                    int a = EDGES[i][0], b = EDGES[i][1];
                    if (a < (int)kpts.size() && b < (int)kpts.size()
                        && kpts[a].conf > 0.2f && kpts[b].conf > 0.2f)
                    {
                        p.drawLine(QPointF(kpts[a].x, kpts[a].y),
                                   QPointF(kpts[b].x, kpts[b].y));
                    }
                }

                for (const auto& k : kpts)
                {
                    if (k.conf > 0.2f)
                        p.drawEllipse(QPointF(k.x, k.y), 3, 3);
                }
            }
        }

        // 7. 결과 이미지 저장
        if (vis.save(posePath))
            qInfo() << "[K] saved pose overlay:" << posePath;
        else
            qWarning() << "[K] save pose failed:" << posePath;

}
