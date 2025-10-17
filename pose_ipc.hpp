#ifndef POSE_IPC_HPP
#define POSE_IPC_HPP

#pragma once
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include <QImage>
#include <QByteArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

struct PoseDet {
    float x1, y1, x2, y2, score;
    std::vector<std::array<float,3>> kpts; // {x,y,score}
};

class PoseIpc {
public:
    explicit PoseIpc(const std::string& sockPath = "/tmp/yolo_pose.sock")
        : path_(sockPath), fd_(-1), fid_(0) {}

    ~PoseIpc() { if (fd_>=0) ::close(fd_); }

    bool connectIfNeeded() {
        if (fd_>=0) return true;
        fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd_ < 0) return false;

        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path_.c_str());
        if (::connect(fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
            ::close(fd_); fd_ = -1;
            return false;
        }
        return true;
    }

    // GRAY8만 보냄 (fmt=0)
    bool inferGray8(const QImage& gray, std::vector<PoseDet>& out) {
        if (gray.isNull()) return false;
        if (!connectIfNeeded()) return false;

        QImage g = (gray.format() == QImage::Format_Grayscale8)
                     ? gray
                     : gray.convertToFormat(QImage::Format_Grayscale8);

        const uint32_t W = g.width();
        const uint32_t H = g.height();
        const uint32_t stride = g.bytesPerLine();
        const uint32_t payload = stride * H;
        const uint32_t fmt = 0; // GRAY8
        const uint32_t fid = ++fid_;

        // 헤더(28B) BE
        uint8_t hdr[28];
        std::memcpy(hdr, "FRM1", 4);
        store_be32(hdr+4, fmt);
        store_be32(hdr+8, W);
        store_be32(hdr+12,H);
        store_be32(hdr+16,stride);
        store_be32(hdr+20,fid);
        store_be32(hdr+24,payload);

        if (!writeAll(hdr, 28)) return false;

        // payload: stride*H (행 단위 전송)
        if (!writeAll(g.bits(), payload)) return false;

        // 응답 한 줄(JSON line)
        QByteArray line;
        if (!readLine(line)) return false;

        // 파싱
        out.clear();
        QJsonParseError pe{};
        QJsonDocument doc = QJsonDocument::fromJson(line, &pe);
        if (pe.error != QJsonParseError::NoError || !doc.isObject()) return false;

        QJsonObject o = doc.object();
        QJsonArray jboxes = o.value("boxes").toArray();
        QJsonArray jkpts  = o.value("kpts").toArray();

        for (int i=0; i<jboxes.size(); ++i) {
            QJsonArray a = jboxes[i].toArray();
            if (a.size() < 5) continue;
            PoseDet d;
            d.x1 = float(a[0].toDouble());
            d.y1 = float(a[1].toDouble());
            d.x2 = float(a[2].toDouble());
            d.y2 = float(a[3].toDouble());
            d.score = float(a[4].toDouble());

            if (i < jkpts.size()) {
                QJsonArray karr = jkpts[i].toArray();
                for (const auto& kv : karr) {
                    QJsonArray kp = kv.toArray();
                    if (kp.size() >= 3) {
                        d.kpts.push_back({ float(kp[0].toDouble()),
                                           float(kp[1].toDouble()),
                                           float(kp[2].toDouble()) });
                    }
                }
            }
            out.push_back(std::move(d));
        }
        return true;
    }

private:
    std::string path_;
    int fd_;
    uint32_t fid_;

    static void store_be32(void* p, uint32_t v) {
        uint32_t be = htonl(v);
        std::memcpy(p, &be, 4);
    }

    bool writeAll(const void* p, size_t n) {
        const uint8_t* cur = static_cast<const uint8_t*>(p);
        size_t left = n;
        while (left > 0) {
            ssize_t w = ::write(fd_, cur, left);
            if (w <= 0) return false;
            cur += w; left -= size_t(w);
        }
        return true;
    }

    bool readLine(QByteArray& out) {
        out.clear();
        char ch;
        while (true) {
            ssize_t r = ::read(fd_, &ch, 1);
            if (r <= 0) return false;
            if (ch == '\n') break;
            out.append(ch);
            if (out.size() > 4*1024*1024) return false; // sanity
        }
        return true;
    }
};


#endif // POSE_IPC_HPP
