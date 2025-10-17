#ifndef POSEWORKER_H
#define POSEWORKER_H

#include <utilHeader.h>
#include <dlfcn.h>

class PoseWorker : public QObject
{
    Q_OBJECT
public :
    explicit PoseWorker(QObject* parent = nullptr);
    ~PoseWorker() override = default;


    PoseEstimatorONNX* pose_ = nullptr;
    PoseParams pPose_;
    std::atomic<bool> busy_{false};

    bool poseReady_ = false;
    QString captureDir_ = "/home/CameraWorker/captures";
    QString saveDir_ ;


signals:
    void poseDone(const QImage& canvas);
    void yoloFrameReady(const int idx, const QImage yimg);
    void poseReady(int camidx, const QImage& painted);

public slots:
    void saveOnly(const QImage& inImg);
    void oneshotsnapPose(const QImage&  inimg);

    void startBatch();
    void runNextInBatch();             //** 추가


private:
    QStringList batchFiles_;           //** 추가
    int batchIdx_ = -1;                //** 추가
    bool batchRunning_ = false;        //** 추가
    QString currentSrcName_;           //** 추가
    bool isloadimg_ = false;

    //tidl 관련
    // ORT 런타임 리소스 (한번 생성 후 계속 사용)
    bool ortReady_ = false;

    // 경로 보관(지연 초기화 시 사용)
    QString modelPath_;
    QString artifactsDir_;

    // 네트 입력 사이즈(네가 쓰던 640x640 유지)
    int netW_ = 640;
    int netH_ = 640;
    static constexpr int SRC_W = 640;
    static constexpr int SRC_H = 480;

    inline int padY() const { return (netH_ - SRC_H) / 2; }

    // 첫 번째 입력 이름(기본값 "images", 필요하면 런타임에 자동탐색)
    std::string inputName_ = "images";
    std::vector<std::string> outputNames_;   // ← 출력 이름 캐시

    bool isUsing_yolo = false;
    std::vector<int64_t> inputShape_{1,3,640,640};

    // 내부 유틸

};

#endif // POSEWORKER_H
