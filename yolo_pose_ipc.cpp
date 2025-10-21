#include "yolo_pose_ipc.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

static inline quint32 bswap32(quint32 x){
    return ((x & 0x000000FFu) << 24) |
           ((x & 0x0000FF00u) << 8 ) |
           ((x & 0x00FF0000u) >> 8 ) |
           ((x & 0xFF000000u) >> 24);
}

quint32 YoloPoseIpc::hostToBE32(quint32 v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return bswap32(v);
#else
    return v;
#endif
}

YoloPoseIpc::YoloPoseIpc(const QString& sockPath, int timeoutMs)
    : sockPath_(sockPath), timeoutMs_(timeoutMs) {}

YoloPoseIpc::~YoloPoseIpc() {
    closeSock();
}

void YoloPoseIpc::setTimeoutMs(int ms) {
    timeoutMs_ = ms;
    if (sock_ >= 0) {
        timeval tv { timeoutMs_/1000, (timeoutMs_%1000)*1000 };
        ::setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(sock_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }
}

void YoloPoseIpc::closeSock() {
    if (sock_ >= 0) {
        ::close(sock_);
        sock_ = -1;
    }
}

bool YoloPoseIpc::connectIfNeeded() {
    if (sock_ >= 0) return true;
    sock_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_ < 0) {
        qWarning() << "[IPC] socket() failed:" << strerror(errno);
        return false;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    QByteArray path = sockPath_.toLocal8Bit();
    if (path.size() >= (int)sizeof(addr.sun_path)) {
        qWarning() << "[IPC] sock path too long";
        closeSock();
        return false;
    }
    std::strncpy(addr.sun_path, path.constData(), sizeof(addr.sun_path)-1);

    if (::connect(sock_, (sockaddr*)&addr, sizeof(addr)) < 0) {
        qWarning() << "[IPC] connect() failed:" << strerror(errno) << "path=" << sockPath_;
        closeSock();
        return false;
    }

    timeval tv { timeoutMs_/1000, (timeoutMs_%1000)*1000 };
    ::setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(sock_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    qInfo() << "[IPC] connected to" << sockPath_;
    return true;
}

bool YoloPoseIpc::sendFrame_RGB888(const QImage& in, quint32 fid) {
    if (sock_ < 0 && !connectIfNeeded()) return false;

    QImage img = (in.format() == QImage::Format_RGB888)
                 ? in
                 : in.convertToFormat(QImage::Format_RGB888);

    const quint32 W = img.width();
    const quint32 H = img.height();
    const quint32 stride = img.bytesPerLine();
    const quint32 payload = H * stride;

    HeaderBE h{};
    std::memcpy(h.magic, "FRM1", 4);
    h.fmt     = hostToBE32(1);       // RGB888
    h.W       = hostToBE32(W);
    h.H       = hostToBE32(H);
    h.stride  = hostToBE32(stride);
    h.fid     = hostToBE32(fid);
    h.payload = hostToBE32(payload);

    // 헤더
    ssize_t wr = ::send(sock_, &h, sizeof(h), MSG_NOSIGNAL);
    if (wr != (ssize_t)sizeof(h)) {
        qWarning() << "[IPC] send header failed:" << strerror(errno);
        closeSock();
        return false;
    }
    // 페이로드
    const uchar* p = img.bits();
    size_t left = payload;
    while (left > 0) {
        ssize_t n = ::send(sock_, p, left, MSG_NOSIGNAL);
        if (n <= 0) {
            qWarning() << "[IPC] send payload failed:" << strerror(errno);
            closeSock();
            return false;
        }
        p += n; left -= n;
    }
    return true;
}

bool YoloPoseIpc::recvJsonLine(QByteArray& line) {
    line.clear();
    if (sock_ < 0) return false;

    char ch;
    while (true) {
        ssize_t n = ::recv(sock_, &ch, 1, 0);
        if (n == 0) {
            qWarning() << "[IPC] server closed";
            closeSock();
            return false;
        }
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // timeout
                return false;
            }
            qWarning() << "[IPC] recv error:" << strerror(errno);
            closeSock();
            return false;
        }
        if (ch == '\n') break;
        line.push_back(ch);
        // 보호: 과도한 라인 길이 제한(1MB)
        if (line.size() > 1024*1024) {
            qWarning() << "[IPC] json line too long";
            return false;
        }
    }
    return true;
}

bool YoloPoseIpc::parseJson(const QByteArray& line, QVector<DetBox>& boxes) {
    boxes.clear();
    lastKpts_.clear();

    QJsonParseError pe{};
    QJsonDocument doc = QJsonDocument::fromJson(line, &pe);
    if (pe.error != QJsonParseError::NoError || !doc.isObject()) {
        qWarning() << "[IPC] json parse error:" << pe.errorString();
        return false;
    }
    auto obj = doc.object();

    // boxes: [[x1,y1,x2,y2,score], ...]
    auto jboxes = obj.value("boxes").toArray();
    boxes.reserve(jboxes.size());
    for (const auto& v : jboxes) {
        auto arr = v.toArray();
        if (arr.size() < 5) continue;
        DetBox b;
        b.x1 = float(arr.at(0).toDouble());
        b.y1 = float(arr.at(1).toDouble());
        b.x2 = float(arr.at(2).toDouble());
        b.y2 = float(arr.at(3).toDouble());
        b.score = float(arr.at(4).toDouble());
        boxes.push_back(b);
    }

    // kpts: [ [ [x,y,score], ... ], ... ]
    auto jkall = obj.value("kpts").toArray();
    lastKpts_.resize(jkall.size());
    for (int i = 0; i < jkall.size(); ++i) {
        auto jkp = jkall.at(i).toArray();
        auto& dst = lastKpts_[i];
        dst.reserve(jkp.size());
        for (const auto& kv : jkp) {
            auto a = kv.toArray();
            if (a.size() < 3) continue;
            Kpt k;
            k.x = float(a.at(0).toDouble());
            k.y = float(a.at(1).toDouble());
            k.score = float(a.at(2).toDouble());
            dst.push_back(k);
        }
    }
    return true;
}

QVector<DetBox> YoloPoseIpc::detect(const QImage& frameRgb, quint32 fid) {
    QVector<DetBox> out;
    // 1) 연결/전송
    if (!sendFrame_RGB888(frameRgb, fid)) {
        return out; // 실패 시 빈 결과(화면은 last*로 유지)
    }
    // 2) 응답 수신(타임아웃되면 빈 결과 반환)
    QByteArray line;
    if (!recvJsonLine(line)) {
        return out;
    }
    // 3) 파싱 → out + lastKpts_
    parseJson(line, out);
    return out;
}
