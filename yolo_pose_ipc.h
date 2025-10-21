#pragma once
#include <QObject>
#include <QImage>
#include <QVector>
#include <QElapsedTimer>
#include <vector>

struct Kpt {
    float x{0}, y{0}, score{0};
};

struct DetBox {
    float x1{0}, y1{0}, x2{0}, y2{0}, score{0};
};

class YoloPoseIpc : public QObject {
    Q_OBJECT
public:
    explicit YoloPoseIpc(const QString& sockPath = "/tmp/yolo_pose.sock",
                         int timeoutMs = 100 /* send/recv 타임아웃 */);
    ~YoloPoseIpc();

    // onFrame()에서 그대로 사용
    QVector<DetBox> detect(const QImage& frameRgb, quint32 fid);
    const std::vector<std::vector<Kpt>>& lastKpts() const { return lastKpts_; }

    void setTimeoutMs(int ms);

private:
    int sock_{-1};
    QString sockPath_;
    int timeoutMs_{100};

    std::vector<std::vector<Kpt>> lastKpts_;

    bool connectIfNeeded();
    void closeSock();
    bool sendFrame_RGB888(const QImage& img, quint32 fid);
    bool recvJsonLine(QByteArray& line);
    bool parseJson(const QByteArray& line, QVector<DetBox>& boxes);

    // 헤더(서버와 동일: BE/네트워크 바이트오더로 보냄)
#pragma pack(push,1)
    struct HeaderBE {
        char     magic[4];   // "FRM1"
        quint32  fmt;        // 0=GRAY8, 1=RGB888
        quint32  W;
        quint32  H;
        quint32  stride;
        quint32  fid;
        quint32  payload;
    };
#pragma pack(pop)

    static quint32 hostToBE32(quint32 v);
};
