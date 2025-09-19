#include "cpu_pointcloud_view.h"
#include <QPainter>
#include <QtMath>
#include <algorithm>
#include <QDebug>
#include <QMatrix4x4>
#include <QColor>

#define TEST_BYPASS_ROTATION   0  // 1: 회전 완전 무시 (yaw/pitch 영향 배제)
#define TEST_ORTHO_NO_DIV_Z    0  // 1: 원근 나눗셈(u=x/z,v=y/z) 금지 → u=x, v=y
#define TEST_FAKE_Z_POSITIVE   0  // 1: 모든 점의 Z'를 안전한 양수로 강제

static inline void distortRadTan(float xn, float yn,
                                 float k1, float k2, float p1, float p2,
                                 float& xdp, float& ydp)
{
    const float r2 = xn*xn + yn*yn;
    const float radial = 1.f + k1*r2 + k2*r2*r2;
    const float two_xy = 2.f * xn * yn;
    xdp = xn * radial + 2.f*p1*xn*yn + p2 * (r2 + 2.f*xn*xn);
    ydp = yn * radial + p1 * (r2 + 2.f*yn*yn) + 2.f*p2*xn*yn;
}

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

void CPUPointCloudView::updatePointCloudWithPts(const QVector<QVector3D> &pts,
                                                const QVector<QPoint>& imgPts,
                                                const QVector<float>& z0)
{
    QMutexLocker lock(&ptsMutex_);
    pts_ = pts;
    imgPts_ = imgPts;      //추가
    z0_ = z0;
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

void CPUPointCloudView::keyPressEvent(QKeyEvent* event)
{
    if (true)                           //*** 수정
       {
           //*** 수정  스크린샷 파일명 생성
           QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
           QString filename = QString("screenshot_%1.png").arg(timestamp);  // or QFileDialog::getSaveFileName(...)

           if (!fb_.isNull())
           {
               if (fb_.save(filename))                          //*** 수정
                   qDebug() << "[Screenshot] saved to:" << filename;
               else
                   qWarning() << "[Screenshot] failed to save!";
           }
           else
           {
               qWarning() << "[Screenshot] framebuffer is null!";
           }
       }

       QWidget::keyPressEvent(event); //*** 수정  기본 이벤트 처리도 호출
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
    fb_.fill(qRgb(12,12,14));
    drawPoints();

    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);
    p.setRenderHint(QPainter::SmoothPixmapTransform, false);
    p.drawImage(0, 0, fb_);
}

void CPUPointCloudView::resizeEvent(QResizeEvent* e)
{

    QWidget::resizeEvent(e);                                //*** 수정
    cx_pix_ = width()  * 0.5f;                              //*** 수정
    cy_pix_ = height() * 0.5f;                              //*** 수정
    fx_pix_ = std::max(1, width())  * 0.03f;                 //*** 수정
    fy_pix_ = std::max(1, height()) * 0.03f;                 //*** 수정

    //update();
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
    //bool pureProj_ = true;
    if (fb_.isNull())
        return;

    fb_.fill(qRgb(12, 12, 14));

    const int   W  = fb_.width();
    const int   H  = fb_.height();

    ///*** 추가사항: 비교 가능 여부(pts_ ↔ imgPts_ 1:1)
    bool hasImgPts = false;
    {
        QMutexLocker lk(&ptsMutex_);
        hasImgPts = (!imgPts_.isEmpty() && imgPts_.size() == pts_.size());
    }

    ///*** 추가사항: 순수 투영(정합) + 비교 모드(오토핏/중앙정렬/카메라보정 전부 OFF)
    if (pureProj_ && hasImgPts)
    {
        const float fx = fx_pix_;                 ///*** 중요: 기존 값을 그대로 사용(하한/오토핏 금지)
        const float fy = fy_pix_;
        const float cx = cx_pix_;
        const float cy = cy_pix_;

        std::vector<float> zbuf((size_t)W*(size_t)H, std::numeric_limits<float>::infinity());
        auto putPixelZ = [&](int x, int y, float z, QRgb col)
        {
            if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return;
            size_t idx = (size_t)y * (size_t)W + (size_t)x;
            if (z < zbuf[idx]) {
                zbuf[idx] = z;
                reinterpret_cast<QRgb*>(fb_.bits() + y * fb_.bytesPerLine())[x] = col;
            }
        };

        ///*** 추가사항: Δu/Δv 통계
        std::vector<float> absDu, absDv, err;
        auto median = [](std::vector<float>& v)->float {
            if (v.empty()) return std::numeric_limits<float>::quiet_NaN();
            std::nth_element(v.begin(), v.begin()+v.size()/2, v.end());
            return v[v.size()/2];
        };

        int N = 0;
        QVector<QVector3D> ptsLocal;
        QVector<QPoint>    imgLocal;
        { QMutexLocker lk(&ptsMutex_); ptsLocal = pts_; imgLocal = imgPts_; }

        for (int i = 0; i < ptsLocal.size(); ++i)
        {
            const QVector3D& P = ptsLocal[i];
            float X = P.x(), Y = P.y(), Z = P.z();
            if (!(Z > 0.f) || !std::isfinite(Z)) continue;

            ///*** 수정사항: 순수 원근 투영(여기서만 Y-다운 한 번)
            const float u = fx * (X / Z) + cx;
            const float v = fy * (-Y / Z) + cy;

            const int ix = (int)std::lround(u);
            const int iy = (int)std::lround(v);
            if ((unsigned)ix >= (unsigned)W || (unsigned)iy >= (unsigned)H) continue;

            putPixelZ(ix, iy, /*zbuf key*/ Z, qRgb(255,220,40));

            ///*** 추가사항: (u,v) vs 원본 (x,y) 비교
            const int rx = imgLocal[i].x();
            const int ry = imgLocal[i].y();
            const float du = u - rx;
            const float dv = v - ry;
            absDu.push_back(std::fabs(du));
            absDv.push_back(std::fabs(dv));
            err.push_back(std::sqrt(du*du + dv*dv));
            ++N;
        }

        qDebug() << "[REPROJ] N=" << N
                 << " med|du|=" << median(absDu)
                 << " med|dv|=" << median(absDv)
                 << " med e="  << median(err);

        return; ///*** 수정사항: 비교 모드 종료(아래 보기용 경로 실행 안 함)
    }

    ///*** 수정사항: 이하 = 기존 ‘보기용 렌더’ 경로(오토핏/중앙정렬 유지)
    const float cx = W * 0.5f;
    const float cy = H * 0.5f;

    ///*** 수정사항: 정합 모드가 아닐 때만 안전 하한 적용
    if (!(fx_pix_ > 0.f) || !(fy_pix_ > 0.f))
    {
        fx_pix_ = std::max(1, W) * 0.6f;
        fy_pix_ = std::max(1, H) * 0.6f;
    }

    cx_pix_ = W * 0.5f;
    cy_pix_ = H * 0.5f;

    if (fx_pix_ < W*0.5f)
        fx_pix_ = W*0.5f;
    if (fy_pix_ < H*0.5f)
        fy_pix_ = H*0.5f;

    QVector<QVector3D> pts;
    { QMutexLocker lk(&ptsMutex_); pts = pts_; }
    const int n = pts.size();
    if (n <= 0) return;

    QMatrix4x4 rot;
    rot.setToIdentity();
    rot.rotate(yaw_,   0, 1, 0);
    rot.rotate(pitch_, 1, 0, 0);

    QVector<QVector3D> rpts;
    rpts.reserve(n);

    float minX =  std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity();
    float minY =  std::numeric_limits<float>::infinity();
    float maxY = -std::numeric_limits<float>::infinity();
    float minZ = +std::numeric_limits<float>::infinity();
    float maxZ = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < n; ++i)
    {
        const QVector3D p = rot * pts[i];
        rpts.push_back(p);
        if (!std::isfinite(p.x()) || !std::isfinite(p.y())) continue;
        if (p.x() < minX) minX = p.x();
        if (p.x() > maxX) maxX = p.x();
        if (p.y() < minY) minY = p.y();
        if (p.y() > maxY) maxY = p.y();
        if (p.z() < minZ) minZ = p.z();
        if (p.z() > maxZ) maxZ = p.z();
    }
    if (rpts.isEmpty()) return;
    if (!(minX < maxX) || !(minY < maxY)) return;

    float zLo = useFixedZ_ ? zMin_ : minZ;
    float zHi = useFixedZ_ ? zMax_ : maxZ;
    if (!(zLo < zHi)) { zLo = minZ; zHi = maxZ; }

    auto colorFromZ = [&](float z) -> QRgb
    {
        if (!std::isfinite(z)) return qRgb(255,255,255);
        float denom = (zHi - zLo);
        if (!(denom > 0.f) || !std::isfinite(denom)) return qRgb(255,255,255);
        float t = (z - zLo) / denom;
        t = std::clamp(t, 0.0f, 1.0f);
        const qreal hueDeg = 240.0 * (t);
        QColor c; c.setHsvF(hueDeg/360.0, 1.0, 1.0);
        return c.rgb();
    };

    const float rangeX = maxX - minX;
    const float rangeY = maxY - minY;
    const float sceneRadiusXY = 0.5f * std::max(rangeX, rangeY);

    const float halfFov = std::atan((W * 0.5f) / std::max(1.f, fx_pix_));
    const float camZ_minFit = sceneRadiusXY / std::tan(halfFov) + 1e-3f;

    camZ_ = std::clamp(camZ_, camZ_minFit, camZ_minFit * 1.1f);

    if (fx_pix_ < fb_.width() * 0.5f)  fx_pix_ = fb_.width()  * 0.5f;
    if (fy_pix_ < fb_.height()* 0.5f)  fy_pix_ = fb_.height() * 0.5f;

    nearZ_ = std::max(nearZ_, 0.001f);

    const float midX = 0.5f * (minX + maxX);
    const float midY = 0.5f * (minY + maxY);
    const float midZ = (minZ + maxZ) * 0.5f;

    std::vector<float> zbuf((size_t)W*(size_t)H, std::numeric_limits<float>::infinity());
    auto putPixelZ = [&](int x, int y, float z, QRgb col)
    {
        if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return;
        size_t idx = (size_t)y * (size_t)W + (size_t)x;
        if (z < zbuf[idx]) {
            zbuf[idx] = z;
            reinterpret_cast<QRgb*>(fb_.bits() + y * fb_.bytesPerLine())[x] = col;
        }
    };

    for (int i = 0; i < rpts.size(); ++i)
    {
        const QVector3D& p = rpts[i];

        const float qx = p.x() - midX;
        const float qy = p.y() - midY;
        const float qz = (p.z() - midZ) + camZ_;
        if (!(qz > nearZ_))
            continue;

        const float sx = fx_pix_ * (qx / qz) + cx_pix_;
        const float sy = fy_pix_ * (-(qy / qz)) + cy_pix_;

        const int ix = (int)std::lround(sx);
        const int iy = (int)std::lround(sy);
        if ((unsigned)ix >= (unsigned)W || (unsigned)iy >= (unsigned)H)
            continue;

       //const QRgb col = colorFromZ(p.z());

        float zColorVal = std::numeric_limits<float>::quiet_NaN();
        {
            QMutexLocker lk(&ptsMutex_);
            if (!z0_.isEmpty() && z0_.size() == pts_.size())
                zColorVal = z0_[i];
        }
        if (!std::isfinite(zColorVal)) zColorVal = p.z(); // fallback (또는 qz)

        // 최종 색상
        const QRgb col = colorFromZ(zColorVal);

        putPixelZ(ix, iy, qz, col);
    }
}
