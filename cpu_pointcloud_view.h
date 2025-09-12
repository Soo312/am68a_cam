#ifndef CPU_POINTCLOUD_VIEW_H
#define CPU_POINTCLOUD_VIEW_H

#pragma once
#include <QWidget>
#include <QVector>
#include <QVector3D>
#include <QImage>
#include <QTimer>
#include <QMutex>
#include <QPoint>

class CPUPointCloudView : public QWidget
{
    Q_OBJECT

public:
    explicit CPUPointCloudView(QWidget* parent = nullptr);
    ~CPUPointCloudView() override = default;

    void updatePointCloud(const QVector<QVector3D>& pts);
    void updatePointCloudUV(const QVector<QVector3D>& pts,
                                const QVector<QPoint>& uvs); // NEW: XYZ+UV 함께 세팅
    void orbitBy(float dYawDeg, float dPitchDeg);
    void zoomBy(float delta);

    void setZRange(float zmin_m, float zmax_m, bool useFixed)
    {
         zMin_ = zmin_m; zMax_ = zmax_m; useFixedZ_ = useFixed;
    }


public slots:
    void setProjectionIntrinsics(float fx, float fy, float cx, float cy)
    {
        fx_pix_ = fx;
        fy_pix_ = fy;
        cx_pix_ = cx;
        cy_pix_ = cy;
        update();
    }
protected:
    void paintEvent(QPaintEvent*) override;
    void resizeEvent(QResizeEvent*) override;

private:
    QVector<QVector3D> pts_;
    QMutex              ptsMutex_;
    float               yaw_   = 0.0f;
    float               pitch_ = 0.0f;
    float               dist_  = 1.5f;

    QImage              fb_;
    QTimer              repaintTimer_;

    QVector<QPoint>    pts_uv_;  // u v 보관

    bool  useFixedZ_ = true;   // 고정 범위로 색 입히기(추천)
    float zMin_ = 0.3f;        // 0.3 m
    float zMax_ = 6.0f;        // 6.0 m

    void ensureFB();
    void drawPoints();

public:
    bool  usePerspective_ = true;    //*** 수정  원근 투영 on/off (원하면 토글)
    float fx_pix_ = 600.f;           //*** 수정  보기용 fx(px)
    float fy_pix_ = 600.f;           //*** 수정  보기용 fy(px)
    float cx_pix_ = 0.f;             //*** 수정  화면 중심 X(px)
    float cy_pix_ = 0.f;             //*** 수정  화면 중심 Y(px)
    float camZ_   = 1.0f;            //*** 수정  카메라-장면 거리(장면 단위, m 권장)
    float nearZ_  = 0.01f;           //*** 수정  전면 클리핑(투영 안정)

};



#endif // CPU_POINTCLOUD_VIEW_H
