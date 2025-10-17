#ifndef YOLO_IPC_HPP
#define YOLO_IPC_HPP
// yolo_ipc.hpp
#pragma once
#include <QtCore>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QImage>


#include <sys/socket.h>
#include <sys/un.h>
#include <arpa/inet.h>   // htonl
#include <unistd.h>      // close

struct DetBox { float x1,y1,x2,y2,score; };
struct Kpt {float x, y, score;};

class YoloIpc : public QObject {
    Q_OBJECT
public:
    explicit YoloIpc(const QString& sockPath="/tmp/yolo.sock", QObject* parent=nullptr)
        : QObject(parent), sockPath_(sockPath) {}

    const QVector<QVector<Kpt>>& lastKpts() const { return lastKpts_; }

    bool connectServer() {
        closeServer();
        fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd_ < 0) return false;
        sockaddr_un addr{}; addr.sun_family = AF_UNIX;
        QByteArray path = sockPath_.toUtf8();
        ::strncpy(addr.sun_path, path.constData(), sizeof(addr.sun_path)-1);
        if (::connect(fd_, (sockaddr*)&addr, sizeof(addr)) != 0) {
            ::close(fd_); fd_=-1; return false;
        }
        return true;
    }
    void closeServer() { if (fd_>=0) { ::close(fd_); fd_=-1; } }
    ~YoloIpc() override { closeServer(); }

    // img를 GRAY8 320x240으로 변환해 전송 → JSON 응답 파싱
    QVector<DetBox> detect(const QImage& img, quint32 frameId=0) {
        if (fd_<0) return {};
        lastKpts_.clear();

        // 1) GRAY8 & 다운스케일(권장 320x240)
        QImage gray = (img.format()==QImage::Format_Grayscale8)
                        ? img
                        : img.convertToFormat(QImage::Format_Grayscale8);
        QImage small = gray.scaled(320, 240, Qt::IgnoreAspectRatio, Qt::FastTransformation);

        const int W = small.width();
        const int H = small.height();
        const int stride = small.bytesPerLine();
        const int payload = stride * H;

        // 2) 헤더(28 bytes): "FRM1" + fmt + W + H + stride + frame_id + payload  (모두 big-endian u32)
        QByteArray hdr; hdr.resize(28);
        char* p = hdr.data();
        memcpy(p, "FRM1", 4); p += 4;
        auto put32=[&](quint32 v){ quint32 n=htonl(v); memcpy(p,&n,4); p+=4; };
        const quint32 FMT_GRAY8 = 0;
        put32(FMT_GRAY8);
        put32(W); put32(H); put32(stride);
        put32(frameId);
        put32(payload);

        // 3) 전송
        if (!writeAll(hdr)) return {};
        if (!writeAll(QByteArray((const char*)small.bits(), payload))) return {};

        // 4) 응답(JSON 한 줄) 수신
        QByteArray line = readLine();
        if (line.isEmpty()) return {};
        QJsonParseError e; auto doc = QJsonDocument::fromJson(line, &e);
        if (e.error != QJsonParseError::NoError || !doc.isObject()) return {};
        auto obj = doc.object();
        auto arr = obj.value("boxes").toArray();

        QVector<DetBox> out; out.reserve(arr.size());
        for (auto v : arr) {
            auto a = v.toArray();
            if (a.size()>=5) {
                out.push_back({ float(a[0].toDouble()), float(a[1].toDouble()),
                                float(a[2].toDouble()), float(a[3].toDouble()),
                                float(a[4].toDouble()) });
            }
        }
        // kpts(optional)
        if (obj.contains("kpts")) {
            QJsonArray all = obj.value("kpts").toArray();
            lastKpts_.reserve(all.size());
            for (auto pv : all) {
                QVector<Kpt> person;
                auto parr = pv.toArray();
                person.reserve(parr.size());
                for (auto kv : parr) {
                    auto ka = kv.toArray();
                    if (ka.size()>=3) {
                        person.push_back({ float(ka[0].toDouble()),
                                           float(ka[1].toDouble()),
                                           float(ka[2].toDouble()) });
                    }
                }
                lastKpts_.push_back(std::move(person));
            }
        }
        return out;
    }

private:
    bool writeAll(const QByteArray& b) {
        const char* p=b.constData(); int left=b.size();
        while (left>0) {
            int n = ::send(fd_, p, left, MSG_NOSIGNAL);
            if (n<=0) return false;
            p+=n; left-=n;
        }
        return true;
    }
    QByteArray readLine() {
        QByteArray out; out.reserve(256);
        char c;
        while (true) {
            int n = ::recv(fd_, &c, 1, 0);
            if (n<=0) return {};
            out.append(c);
            if (c=='\n') break;
            if (out.size()> (1<<20)) return {}; // safety: 1MB
        }
        return out;
    }

    QString sockPath_;
    int fd_ = -1;
    QVector<QVector<Kpt>> lastKpts_;
};

#endif // YOLO_IPC_HPP
