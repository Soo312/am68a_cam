#ifndef IMAGERENDERHELPER_H
#define IMAGERENDERHELPER_H

#pragma once
#include <QImage>
#include <Arena/ArenaApi.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <QVector>
#include <QVector2D>
#include <QVector3D>

struct XYZ_I16
{
    int16_t x;
    int16_t y;
    int16_t z;
    int16_t i;   // intensity
};
struct XYZC_I16
{
    qint16 x;
    qint16 y;
    qint16 z;
    qint16 c;

};

struct BackProjLUT
{
    int w{0};
    int h{0};
    QVector<QVector2D> ray;

};

bool BuildBackProjLUT(int w, int h, float fx, float fy, float cx, float cy, BackProjLUT& lut);



class ImageRenderHelper
{
public:
    static bool makeDepthFalseColor(Arena::IImage* img,
                                    uint16_t zMin,
                                    uint16_t zMax,
                                    QImage& outBGR);

    static bool makeLuminanceFalseColor(Arena::IImage* img,
                                        uint16_t iMin,
                                        uint16_t iMax,
                                        QImage& outBGR);

    static bool makeIntensityGray(Arena::IImage* img,
                                  uint16_t iMin,
                                  uint16_t iMax,
                                  QImage& outGray);

    static bool extractPointCloudABCY16(
            Arena::IImage* img,
            float scale,
            int stride,
            quint16 confMin,
            QVector<QVector3D>& coutPts,
            QVector<quint16>* outConf = nullptr,
            int* outW = nullptr,
            int* outH = nullptr

            );
    static bool extractPointCloudC16(Arena::IImage* img,
                              const BackProjLUT& lut,
                              float zScale,              // ex) 0.001f (mm→m)
                              uint16_t zInvalid,         // ex) 0x8000
                              uint16_t zMinValid,        // ex) 300   (mm)  필요 없으면 0
                              uint16_t zMaxValid,        // ex) 6000  (mm)  필요 없으면 0
                              QVector<QVector3D>& outPts,
                              QVector<QPoint>* outImgPts,
                              int* outW = nullptr,
                              int* outH = nullptr,
                              QVector<float>* outZ0 = nullptr);

    static inline size_t calcStepBytes(Arena::IImage* img);

private:
    static inline bool isInvalidXYZ(int16_t X, int16_t Y, int16_t Z)
    {
        const bool h1 = (X == (int16_t)0x8000 && Y == (int16_t)0x8000 && Z == (int16_t)0x8000 );
        const bool h2 = (X == (int16_t)0xFFFF && Y == (int16_t)0xFFFF && Z == (int16_t)0xFFFF );
        return h1 || h2;
    }



};


#endif // IMAGERENDERHELPER_H
