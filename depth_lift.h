#ifndef DEPTH_LIFT_H
#define DEPTH_LIFT_H

#pragma once
#include <vector>
#include "person_detect_onnx.h"

struct KP3 { float x, y, z; };  // 3D point (m 단위)

struct CamIntrin
{
    float fx = 0.0f, fy = 0.0f;
    float cx = 0.0f, cy = 0.0f;
    float depthScale = 0.001f;  // mm → m
};

// bbox 중심을 depthU16에서 샘플링하여 3D 좌표 추정
inline bool liftCenterTo3D(const struct PDBox& box,
                           const uint16_t* depth,
                           int w, int h, int stride,
                           const CamIntrin& K,
                           KP3& out)
{
    int cx = int(box.x + box.w / 2);
    int cy = int(box.y + box.h / 2);
    if (cx < 0 || cy < 0 || cx >= w || cy >= h)
        return false;

    int z = depth[cy * stride + cx];
    if (z == 0) return false;

    float zz = z * K.depthScale;
    float xx = (cx - K.cx) * zz / K.fx;
    float yy = (cy - K.cy) * zz / K.fy;
    out = { xx, yy, zz };
    return true;
}

#endif // DEPTH_LIFT_H
