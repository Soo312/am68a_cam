#include "cpu_pointcloud_view.h"
#include <QPainter>
#include <QtMath>
#include <algorithm>
#include <QDebug>
#include <QMatrix4x4>
#include <QColor>

#define TEST_BYPASS_ROTATION   0  // 1: 회전 완전 무시 (yaw/pitch 영향 배제)
#define TEST_ORTHO_NO_DIV_Z    0  // 1: 원근 나눗셈(u=x/z,v=y/z) 금지 → u=x, v=y
#define TEST_FAKE_Z_POSITIVE   0  // 1: 모든 점의 Z'를 안전`한 양수로 강제

CPUPointCloudView::CPUPointCloudView(QWidget* parent)
    :QWidget(parent)
{
    // NEW: 보기 초기화
    yaw_   = 0.0f;     // NEW
    pitch_ = 0.0f;     // NEW
    dist_  = 2.0f;     // NEW: 화면에 보이도록 카메라 거리(상황에 맞게)

    setAttribute(Qt::WA_OpaquePaintEvent, true);
    setAutoFillBackground(false);


    setMinimumSize(320,240);
    repaintTimer_.setInterval(16);//~60Hz
    connect(&repaintTimer_, &QTimer::timeout, this, [this]
    {
        update();
    });
    repaintTimer_.start();
}

void CPUPointCloudView::updatePointCloud(const QVector<QVector3D> &pts)
{
    QMutexLocker lk(&ptsMutex_);
    pts_ = pts;

    update();

}

void CPUPointCloudView::updatePointCloudUV(const QVector<QVector3D> &pts, const QVector<QPoint> &uvs)
{
    QMutexLocker lk(&ptsMutex_);
    pts_ = pts;
    pts_uv_ = uvs;

    update();
}

void CPUPointCloudView::orbitBy(float dYawDeg, float dPitchDeg)
{
    yaw_ += dYawDeg;
    pitch_ = std::clamp(pitch_ + dPitchDeg, -89.0f, 89.0f);

    update();

}


void CPUPointCloudView::zoomBy(float delta)
{
    dist_ = std::clamp(dist_ - delta, 0.2f, 10.0f);

    update();
}

void CPUPointCloudView::paintEvent(QPaintEvent* ev)
{
    /*
    ensureFB();                              // MUST: DPR 반영된 FB 준비
    fb_.fill(qRgb(12, 12, 14));              // MUST: 배경은 여기서만 칠함

    drawPoints();                             // MUST: 순수 래스터 경로

    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);
    p.setRenderHint(QPainter::SmoothPixmapTransform, false);
    p.drawImage(0, 0, fb_);
    */

    /*
    // 1) ensureFB(), 배경 fill 없음  // ❌
    drawPoints();                       // ← 여기서도 FB 초기화 안 함

    // 2) 스케일 복사 (rect(), fb_)   // ❌ HiDPI/스케일 왜곡, 누적·찌그러짐 유발
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);
    p.setRenderHint(QPainter::SmoothPixmapTransform, false);
    if (!fb_.isNull())
    {
        p.drawImage(rect(), fb_);       // ❌ 반드시 (0,0,fb_)로 1:1 그려야 함
    }
    */

    ensureFB();
    if (fb_.isNull()) return;
    fb_.fill(qRgb(12,12,14));


    drawPoints();

    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);
    p.setRenderHint(QPainter::SmoothPixmapTransform, false);
    p.drawImage(0, 0, fb_);
}

void CPUPointCloudView::resizeEvent(QResizeEvent*)
{
    update();
}

void CPUPointCloudView::ensureFB()
{
    const qreal dpr = devicePixelRatioF();            // New: HiDPI 대응
    const int w = qMax(1, qRound(width()  * dpr));    // New
    const int h = qMax(1, qRound(height() * dpr));    // New

    if (fb_.isNull() ||
        fb_.width()  != w ||
        fb_.height() != h ||
        fb_.format() != QImage::Format_ARGB32_Premultiplied) // New
    {
        fb_ = QImage(w, h, QImage::Format_ARGB32_Premultiplied); // New
        fb_.setDevicePixelRatio(dpr);                   // New
    }
}


static inline void rotYawPitch(const QVector3D& v, float yawDeg, float pitchDeg, QVector3D& out)
{
    const float cy=qCos(qDegreesToRadians(yawDeg)),  sy=qSin(qDegreesToRadians(yawDeg));
    const float cp=qCos(qDegreesToRadians(pitchDeg)),sp=qSin(qDegreesToRadians(pitchDeg));
    // Yaw(Y) → Pitch(X) 순
    float x =  cy*v.x() + sy*v.z();
    float z = -sy*v.x() + cy*v.z();
    float y =  cp*v.y() + (-sp)*z;
    z       =  sp*v.y() +  cp*z;
    out = QVector3D(x,y,z);
}

void CPUPointCloudView::drawPoints()
{
    if (fb_.isNull())
            return;

        // 배경 클리어(겹그림 방지)
        fb_.fill(qRgb(12, 12, 14));

        const int   W  = fb_.width();
        const int   H  = fb_.height();
        const float cx = W * 0.5f;
        const float cy = H * 0.5f;

        // 점군 로컬 복사 (락 최소화)
        QVector<QVector3D> pts;
        {
            QMutexLocker lk(&ptsMutex_);
            pts = pts_;
        }
        const int n = pts.size();
        if (n <= 0)
            return;
        //회전 행렬 적용

        QMatrix4x4 rot;
        rot.setToIdentity();
        rot.rotate(yaw_,   0, 1, 0);  // 좌우(수평) 회전
        rot.rotate(pitch_, 1, 0, 0);  // 상하(수직) 회전

        // 1-pass: 회전한 좌표를 임시 버퍼에 담으면서 XY 범위 계산
        QVector<QVector3D> rpts;      // rotated pts
        rpts.reserve(n);

        // X,Y 범위 스캔 (Z 완전 무시, 회전도 일단 끔)
        float minX =  std::numeric_limits<float>::infinity();
        float maxX = -std::numeric_limits<float>::infinity();
        float minY =  std::numeric_limits<float>::infinity();
        float maxY = -std::numeric_limits<float>::infinity();

        float minZ = +std::numeric_limits<float>::infinity();
        float maxZ = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < n; ++i)
        {
            const QVector3D& p = rot * pts[i];
            rpts.push_back(p);
            if (!std::isfinite(p.x()) || !std::isfinite(p.y()))
                continue;

            if (p.x() < minX) minX = p.x();
            if (p.x() > maxX) maxX = p.x();
            if (p.y() < minY) minY = p.y();
            if (p.y() > maxY) maxY = p.y();

            if (p.z() < minZ) minZ = p.z();
            if (p.z() > maxZ) maxZ = p.z();
        }

        if (rpts.isEmpty()) return;

        if (!(minX < maxX) || !(minY < maxY))
        {
            //qDebug() << "[FLAT] invalid XY range";
            return;
        }

        float zLo = useFixedZ_ ? zMin_ : minZ;
        float zHi = useFixedZ_ ? zMax_ : maxZ;
        if (!(zLo < zHi))
        { zLo = minZ; zHi = maxZ; }

        auto colorFromZ = [&](float z) -> QRgb
        {
            if (!std::isfinite(z))
                return qRgb(255,255,255);
            float denom = (zHi - zLo);
            if (!(denom > 0.f) || !std::isfinite(denom))
                return qRgb(255,255,255);
            float t = (z - zLo) / (denom);
            if (!std::isfinite(t))
                t = 0.f;

            t = std::clamp(t, 0.0f, 1.0f);
            const qreal hueDeg = 240.0 * (1.0 - t);
            QColor c; c.setHsvF(hueDeg/360.0, 1.0, 1.0);
            return c.rgb();
        };

        const float rangeX = maxX - minX;
        const float rangeY = maxY - minY;
        const float midX   = 0.5f * (minX + maxX);
        const float midY   = 0.5f * (minY + maxY);

        // 화면을 95% 채우도록 동일비율 스케일
        const float scaleX = 0.95f * W / (rangeX > 0 ? rangeX : 1e-6f);
        const float scaleY = 0.95f * H / (rangeY > 0 ? rangeY : 1e-6f);
        const float S      = std::min(scaleX, scaleY);

        uchar* base   = fb_.bits();
        const int stride = fb_.bytesPerLine();
        //auto putPixel = [&](int x, int y)
        //{
        //    if ((unsigned)x < (unsigned)W && (unsigned)y < (unsigned)H)
        //    {
        //        reinterpret_cast<QRgb*>(base + y * stride)[x] = qRgb(255, 220, 40);
        //    }
        //};

        static std::vector<float> zbuf;
        zbuf.assign((size_t)W * (size_t)H, std::numeric_limits<float>::infinity());

        auto putPixelZ = [&](int x, int y, float z, QRgb col)
        {
            if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return;
            size_t idx = (size_t)y * (size_t)W + (size_t)x;
            if (z < zbuf[idx]) {
                zbuf[idx] = z;
                reinterpret_cast<QRgb*>(base + y * stride)[x] = col;
            }
        };

        int plotted = 0;
        for (int i = 0; i < rpts.size(); ++i)
        {
            const QVector3D& p = rpts[i];

            // 정사영: X,Y만 사용 (Y는 화면 상하 반전)
            const float sx = S * (p.x() - midX) + cx;
            const float sy = S * (-(p.y() - midY)) + cy;

            const int ix = (int)std::lround(sx);
            const int iy = (int)std::lround(sy);

            //if ((unsigned)ix < (unsigned)W && (unsigned)iy < (unsigned)H)
            //{
            //    putPixel(ix, iy);
            //    ++plotted;
            //}

            const QRgb col = colorFromZ(p.z());      // ***
            putPixelZ(ix, iy, p.z(), col);           // ***
        }

        //qDebug() << "[FLAT] plotted:" << plotted
        //        << " X:" << minX << "~" << maxX
        //        << " Y:" << minY << "~" << maxY
        //        << " S:" << S;
}
