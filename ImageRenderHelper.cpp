#include "ImageRenderHelper.h"



#include <QDebug>


// 행 스텝(바이트) 계산 도우미
// 1) 우선 width * bpp 로 계산
// 2) 만약 sizeFilled가 높이로 딱 나눠떨어지면 그 값을 우선 사용(패딩이 있을 경우 대비)


size_t ImageRenderHelper::calcStepBytes(Arena::IImage* img)
{
    const int w = (int)img->GetWidth();
    const int h = (int)img->GetHeight();
    const int bpp = (int)img->GetBitsPerPixel(); // bits
    size_t step = (size_t)((w * bpp + 7) / 8);   // bytes per row (tight)

    const size_t filled = (size_t)img->GetSizeFilled(); // 전체 채워진 바이트 수
    if (h > 0 && filled >= step * (size_t)h)
    {
        // 딱 나눠떨어지면 패딩 포함 실제 step로 간주
        if (filled % (size_t)h == 0)
            step = filled / (size_t)h;
    }
    return step;
}
bool ImageRenderHelper::extractPointCloudABCY16(
    Arena::IImage* img,
    float scale,
    int   stride,
    quint16 confMin,
    QVector<QVector3D>& outPts,
    QVector<quint16>*   outConf,
    int* outW,
    int* outH
)
{
    if(!img) return false;

    const int w = static_cast<int>(img->GetWidth());
    const int h = static_cast<int>(img->GetHeight());
    const int bpp = static_cast<int>(img->GetBitsPerPixel());

    if(w <= 0 || h <= 0 || bpp != 64)
    {
        return false;
    }

    const auto* base = static_cast<const uchar*>(img->GetData());
    if(!base) return false;

    //stide rPtks
    const size_t stepBytes = calcStepBytes(img);

    outPts.clear();
    if(outConf) outConf->clear();

    const int sx = std::max(1,stride);
    const int sy = std::max(1,stride);

    //다운샘플반영
    const int resW = (w + sx - 1 )/ sx;
    const int resH = (h + sy - 1) / sy;
    outPts.reserve(resW * resH);
    if (outConf) outConf->reserve(resW * resH);

    // ABCY16: (x,y,z,c) = 16-bit signed * 3 + 16-bit (confidence/intensity)
    for (int y = 0; y < h; y += sy) {
        const auto* row = reinterpret_cast<const XYZC_I16*>(base + static_cast<size_t>(y) * stepBytes);
        for (int x = 0; x < w; x += sx) {
            const XYZC_I16 p = row[x];

            // 무효 포인트/신뢰도 필터
            if (isInvalidXYZ(p.x, p.y, p.z)) continue;
            if (p.z <= 0) continue; // 음수/0 깊이 제거(장치 스펙에 맞게 조정 가능)
            if (confMin > 0 && static_cast<quint16>(p.c) < confMin) continue;

            // 스케일 적용 (ex: mm → m이면 scale = 0.001f)
            outPts.push_back(QVector3D(
                static_cast<float>(p.x) * scale,
                static_cast<float>(p.y) * scale,
                static_cast<float>(p.z) * scale
            ));
            if (outConf) outConf->push_back(static_cast<quint16>(p.c));
        }
    }

    if (outW) *outW = w;
    if (outH) *outH = h;

    qDebug() << "[extractPointCloudABCY16] w:" << w
             << " h:" << h
             << " bpp:" << bpp
             << " outPts:" << outPts.size();    // ★ 이 숫자 체크

    return !outPts.isEmpty();

}



bool ImageRenderHelper::makeDepthFalseColor(Arena::IImage *img,
                                            uint16_t zMin,
                                            uint16_t zMax,
                                            QImage &outBGR)
{
    if(!img)
        return false;

    const int w = (int)img->GetWidth();
    const int h = (int)img->GetHeight();
    const uint8_t* base = static_cast<const uint8_t*>(img->GetData());

    if(!base || w <= 0 || h <= 0)
        return false;

    const size_t stepBytes = calcStepBytes(img);
    cv::Mat v16(h,w,CV_16UC1);
    auto bpp = (int)img->GetBitsPerPixel();

    //이름이 애매모호하니 bpp로 식별
    if(bpp == 64) //Coord3D_ABCY16
    {
        for(int y = 0; y< h; ++y)
        {
            const XYZ_I16* row = reinterpret_cast<const XYZ_I16*>(base + y * stepBytes);
            uint16_t* dst = v16.ptr<uint16_t>(y);

            for(int x = 0; x < w ;++x)
            {
                const auto& p = row[x];
                dst[x] = isInvalidXYZ(p.x,p.y,p.z) ? 0u : (uint16_t)std::max(0,(int)p.z);
            }
        }
    }
    else if(bpp ==16)//Mono16 or Z16
    {
        for (int y = 0; y < h; ++y)
        {
            std::memcpy(v16.ptr(y), base + y * stepBytes, (size_t)w * sizeof(uint16_t));
        }
    }
    else if (bpp == 8) // Mono8 → 바로 컬러맵
    {
        cv::Mat v8(h, w, CV_8UC1, const_cast<uint8_t*>(base), (size_t)stepBytes);
        cv::Mat bgr;
        cv::applyColorMap(v8, bgr, cv::COLORMAP_JET);
        outBGR = QImage(bgr.data, bgr.cols, bgr.rows, bgr.step, QImage::Format_BGR888).copy();
        return !outBGR.isNull();
    }
    else
    {
        return false;
    }

    if (zMin == 0 && zMax == 0)
    {
        double mn = 0, mx = 0;
        cv::minMaxLoc(v16, &mn, &mx);
        if (mx <= mn) mx = mn + 1.0;
        zMin = (uint16_t)mn;
        zMax = (uint16_t)mx;
    }
    cv::Mat v8;
    const double scale = 255.0 / std::max(1, (int)zMax - (int)zMin);
    const double shift = -double(zMin) * scale;
    v16.convertTo(v8, CV_8U, scale, shift);

    cv::Mat bgr;
    cv::applyColorMap(255 - v8, bgr, cv::COLORMAP_JET);

    // 범위 밖은 검정
    cv::Mat oob = (v16 < (uint16_t)zMin) | (v16 > (uint16_t)zMax);
    bgr.setTo(cv::Scalar(0, 0, 0), oob);

    outBGR = QImage(bgr.data, bgr.cols, bgr.rows, bgr.step, QImage::Format_BGR888).copy();
    return !outBGR.isNull();

}

bool BuildBackProjLUT(int w, int h, float fx, float fy, float cx, float cy, BackProjLUT &lut)
{
    if (w <= 0 || h <= 0 || fx == 0.0f || fy == 0.0f)
    {
         qInfo() << "[ToF] Build LUT" << w << "x" << h;
        return false;
    }

    lut.w = w;
    lut.h = h;
    lut.ray.resize(w * h);

    for(int y = 0; y < h; ++y)
    {
        for(int x = 0; x< w; ++x)
        {
            const float rx = (float(x) - cx ) /fx;
            const float ry = (float(y) - cy ) / fy;
            lut.ray[y * w + x] = QVector2D(rx,ry);

        }
    }
}

bool ImageRenderHelper::extractPointCloudC16(Arena::IImage* img,
                          const BackProjLUT& lut,
                          float zScale,
                          uint16_t zInvalid,
                          uint16_t zMinValid,
                          uint16_t zMaxValid,
                          QVector<QVector3D>& outPts,
                          QVector<QPoint>* outImgPts,
                          int* outW,
                          int* outH,
                          QVector<float>* outZ0)
{

    static int frameNo = 0;
    size_t nInvalid=0, nLt2m=0, nGe2m=0, nGe25m=0;

    if (!img)
    {
        return false;
    }

    const int w   = (int)img->GetWidth();
    const int h   = (int)img->GetHeight();
    const int bpp = (int)img->GetBitsPerPixel();

    if (w <= 0 || h <= 0 || bpp != 16)
    {
        return false; // C16 전용
    }

    if (lut.w != w || lut.h != h || lut.ray.size() != w * h)
    {
        return false; // LUT 크기 불일치
    }

    const uint8_t* base = static_cast<const uint8_t*>(img->GetData());
    if (!base)
    {
        return false;
    }

    const size_t stepBytes = ImageRenderHelper::calcStepBytes(img); // 이미 존재하는 함수 사용
    // stride 준수: 행 시작 = base + y * stepBytes  :contentReference[oaicite:5]{index=5}

    outPts.clear();
    outPts.reserve(w * h / 2); // 대충 절반 잡기 (필요 시 조정)

    if (outImgPts)
    {
        outImgPts->clear();
        outImgPts->reserve(w * h / 2);
    }

    if (outZ0)
    {
        outZ0->clear();
        outZ0->reserve(w * h / 2);
    }

    static bool kC16IsRange = true;
    static float kZ_A = 1.000f;
    static float kZ_B = 0.000f;
    static float kZ_C = 0.000f;


    for (int y = 0; y < h; ++y)
    {
        const uint16_t* row = reinterpret_cast<const uint16_t*>(base + y * stepBytes);
        for (int x = 0; x < w; ++x)
        {
            const uint16_t Zraw = row[x];

            if (Zraw == zInvalid)
            {
                continue;
            }
            if (zMinValid && Zraw < zMinValid)
            {
                continue;
            }
            if (zMaxValid && Zraw > zMaxValid)
            {
                continue;
            }

            float Zs = Zraw * zScale;
            if (Zs < 2.0f) nLt2m++;
            else           nGe2m++;

            const QVector2D r = lut.ray[y * w + x]; // ((u-cx)/fx, (v-cy)/fy)

            //const float Z = float(Zraw) * zScale;


            float rx = r.x();
            float ry = r.y();

            //float Zs = float(Zraw) * zScale;

            float Zaxis = Zs;

            if(kC16IsRange)
            {
                const float invCos = std::sqrt(1.0f + rx*rx + ry * ry);
                Zaxis = Zs /  invCos;
            }

            //const float Zcorr = (kZ_A * Zaxis) + kZ_B + (kZ_C * Zaxis * Zaxis);
            const float Zcorr = Zaxis;

            const float X = rx * Zcorr;
            const float Y = ry * Zcorr;

            outPts.push_back(QVector3D(X,-Y,Zcorr));

            if(outImgPts)
            {
                outImgPts->push_back((QPoint(x,y)));
            }

            if (outZ0) outZ0->push_back(Zaxis);


            //const float X = r.x() * Z;
            //const float Y = r.y() * Z;

            // (시각계 상하 뒤집기 원하면 Y = -Y)
            //outPts.push_back(QVector3D(X, -Y, Z));
        }
    }

    if (outW) *outW = w;
    if (outH) *outH = h;



    return !outPts.isEmpty();
}
