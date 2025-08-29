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
