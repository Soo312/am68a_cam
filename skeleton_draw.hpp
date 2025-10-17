#ifndef SKELETON_DRAW_HPP
#define SKELETON_DRAW_HPP

// skeleton_draw.hpp
#pragma once
#include <QPainter>
#include <QPointF>
#include <vector>
#include <array>

// COCO(17 keypoints) edge list (idx: nose=0, Leye=1, Reye=2, Lear=3, Rear=4,
// Lsh=5, Rsh=6, Lel=7, Rel=8, Lwr=9, Rwr=10, Lhp=11, Rhp=12, Lkn=13, Rkn=14, LAn=15, RAn=16)
static inline const int COCO_EDGES[][2] = {
    {5,6}, {5,7}, {7,9}, {6,8}, {8,10},
    {5,11}, {6,12}, {11,12}, {11,13}, {13,15}, {12,14}, {14,16},
    {0,1}, {0,2}, {1,3}, {2,4}
};
static inline const int NUM_EDGES = sizeof(COCO_EDGES)/sizeof(COCO_EDGES[0]);

inline void drawSkeleton(QPainter& p, const std::vector<std::array<float,3>>& kpts,
                         float sx, float sy, float kptTh=0.35f)
{
    // 점
    p.setPen(QPen(Qt::yellow, 2));
    for (const auto& k : kpts) {
        if (k[2] < kptTh) continue;
        p.drawEllipse(QPointF(k[0]*sx, k[1]*sy), 2.0, 2.0);
    }
    // 선
    p.setPen(QPen(Qt::magenta, 2));
    int K = int(kpts.size());
    for (int i=0; i<NUM_EDGES; ++i) {
        int a = COCO_EDGES[i][0], b = COCO_EDGES[i][1];
        if (a < K && b < K && kpts[a][2] >= kptTh && kpts[b][2] >= kptTh) {
            p.drawLine(QPointF(kpts[a][0]*sx, kpts[a][1]*sy),
                       QPointF(kpts[b][0]*sx, kpts[b][1]*sy));
        }
    }
}


#endif // SKELETON_DRAW_HPP
