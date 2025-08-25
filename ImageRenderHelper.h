#ifndef IMAGERENDERHELPER_H
#define IMAGERENDERHELPER_H

#pragma once
#include <QImage>
#include <Arena/ArenaApi.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct XYZ_I16
{
    int16_t x;
    int16_t y;
    int16_t z;
    int16_t i;   // intensity
};

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

private:
    static inline bool isInvalidXYZ(int16_t X, int16_t Y, int16_t Z)
    {
        const bool h1 = (X == (int16_t)0x8000 && Y == (int16_t)0x800 && Z == (int16_t)0x8000 );
        const bool h2 = (X == (int16_t)0xFFFF && Y == (int16_t)0xFFFF && Z == (int16_t)0xFFFF );
        return h1 || h2;
    }
    static inline size_t calcStepBytes(Arena::IImage* img);


};


#endif // IMAGERENDERHELPER_H
