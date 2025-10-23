#include "ImageRenderHelper.h"



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
    if (zMax <= zMin) zMax = (uint16_t)(zMin + 1);            // //수정


    cv::Mat v8;
    const double scale = 255.0 / std::max(1, (int)zMax - (int)zMin);
    const double shift = -double(zMin) * scale;
    v16.convertTo(v8, CV_8U, scale, shift);

    cv::Mat v8inv;                                            // //변경
    cv::subtract(cv::Scalar::all(255), v8, v8inv);            // //변경

    cv::Mat bgr;
    //cv::applyColorMap(255 - v8, bgr, cv::COLORMAP_JET);
    cv::applyColorMap(v8inv, bgr, cv::COLORMAP_JET);

    // 범위 밖은 검정
    cv::Mat oob = (v16 < (uint16_t)zMin) | (v16 > (uint16_t)zMax);
    bgr.setTo(cv::Scalar(0, 0, 0), oob);

    outBGR = QImage(bgr.data, bgr.cols, bgr.rows, bgr.step, QImage::Format_BGR888).copy();
    return !outBGR.isNull();

}
bool ImageRenderHelper::depthC16ToGray8(
    Arena::IImage* pIn,
    uint16_t zMin,
    uint16_t zMax,
    double gamma,
    bool invert,
    QImage& out_qimg)
{
    // 1. 입력 이미지 포인터 유효성 검사
    if (!pIn) {
        return false;
    }

    // 2. (수정) 픽셀 포맷 이름을 직접 비교하는 대신, 픽셀 당 비트(bpp)가 16인지 확인합니다.
    //    Mono16, Coord3D_C16 등 대부분의 16비트 뎁스 데이터에 대응할 수 있습니다.
    if (pIn->GetBitsPerPixel() != 16) {
        return false;
    }

    const int w = static_cast<int>(pIn->GetWidth());
    const int h = static_cast<int>(pIn->GetHeight());
    if (w <= 0 || h <= 0) {
        return false;
    }

    // 3. 소스 데이터 포인터 및 스트라이드(stride) 가져오기
    const uint8_t* pSrcData = pIn->GetData();
    const size_t srcStride = calcStepBytes(pIn);

    // 4. 8비트 흑백 출력 이미지 생성
    out_qimg = QImage(w, h, QImage::Format_Grayscale8);
    if (out_qimg.isNull()) {
        return false;
    }

    // 5. 정규화를 위한 값 계산 (0으로 나누기 방지)
    const float range = static_cast<float>(zMax - zMin);
    if (range <= 0) {
        out_qimg.fill(0); // 범위를 알 수 없으면 검은색 이미지로 채우고 반환
        return true;
    }
    const float scale = 1.0f / range;
    const float inv_gamma = (gamma > 0.0) ? (1.0f / static_cast<float>(gamma)) : 1.0f;

    // 6. 픽셀 단위로 변환 작업 수행
    for (int y = 0; y < h; ++y) {
        const uint16_t* pSrcLine = reinterpret_cast<const uint16_t*>(pSrcData + y * srcStride);
        uint8_t* pDstLine = out_qimg.scanLine(y);

        for (int x = 0; x < w; ++x) {
            uint16_t z16 = pSrcLine[x];
            z16 = std::max(zMin, std::min(z16, zMax));
            float normalized = static_cast<float>(z16 - zMin) * scale;

            if (invert) {
                normalized = 1.0f - normalized;
            }
            if (inv_gamma != 1.0f) {
                normalized = std::pow(normalized, inv_gamma);
            }
            pDstLine[x] = static_cast<uint8_t>(normalized * 255.0f);
        }
    }

    return true;
}
