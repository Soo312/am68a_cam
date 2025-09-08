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

    void ensureFB();
    void drawPoints();

};



#endif // CPU_POINTCLOUD_VIEW_H
