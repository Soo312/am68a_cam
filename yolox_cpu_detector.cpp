#include "yolox_cpu_detector.h"
#include <numeric>  // iota, sort 비교 등
#include <utilHeader.h>
#include <QDebug>
#include <exception>

YoloXNanoLiteCpu::YoloXNanoLiteCpu()
{
    //***FIX: 기본 옵션 설정
    opts_.SetIntraOpNumThreads(4);
    opts_.SetInterOpNumThreads(1);
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    //***FIX: CPU EP 강제 호출 삭제
    // OrtSessionOptionsAppendExecutionProvider_CPU(opts_, 0); // 불필요 + 일부 빌드에 없음
}

bool YoloXNanoLiteCpu::init(const QString& modelPath, int inputSize, int threads)
{
    if (threads > 0)
    {
        opts_.SetIntraOpNumThreads(threads);
    }

    inW_ = inH_ = inputSize;

    session_ = std::make_unique<Ort::Session>(env_, modelPath.toStdString().c_str(), opts_);

    {
        try
        {
            auto tinfo = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();

            // dtype 로그
            auto et = tinfo.GetElementType();
            if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                qWarning() << "[ORT] model input dtype is not float32. type =" << (int)et;
            }

            // ★ 위험한 GetShape() 대신 안전 경로: Count → sanity check → GetDimensions
            size_t rank = 0;
            try
            {
                rank = tinfo.GetDimensionsCount();
            }
            catch (const Ort::Exception& e)
            {
                qWarning() << "[ORT] GetDimensionsCount failed:" << e.what();
                rank = 0;
            }

            if (rank >= 4 && rank <= 8)  // 일반적으로 4
            {
                std::vector<int64_t> dims(rank);
                try
                {
                    tinfo.GetDimensions(dims.data(), dims.size());
                    // dims: [N, C, H, W] 가정. 음수(-1)면 동적.
                    if (dims.size() >= 4 && dims[2] > 0 && dims[3] > 0)
                    {
                        inH_ = static_cast<int>(dims[2]);
                        inW_ = static_cast<int>(dims[3]);
                        qInfo() << "[ORT] fixed input size detected:" << inW_ << "x" << inH_;
                    }
                    else
                    {
                        qInfo() << "[ORT] dynamic input size; using requested:"
                                << inW_ << "x" << inH_;
                    }
                }
                catch (const Ort::Exception& e)
                {
                    qWarning() << "[ORT] GetDimensions failed:" << e.what()
                               << " → use requested:" << inW_ << "x" << inH_;
                }
            }
            else
            {
                // rank가 비정상(0 또는 너무 큼) → 그대로 요청한 값 사용
                qWarning() << "[ORT] unexpected rank =" << (qulonglong)rank
                           << " → use requested:" << inW_ << "x" << inH_;
            }
        }
        catch (const std::exception& e)  // std::length_error 등도 포착
        {
            qWarning() << "[ORT] input shape introspection failed:" << e.what()
                       << " → use requested:" << inW_ << "x" << inH_;
        }
        catch (...)  // 혹시 모를 예외
        {
            qWarning() << "[ORT] input shape introspection failed: unknown error"
                       << " → use requested:" << inW_ << "x" << inH_;
        }
    }

    {
        Ort::AllocatorWithDefaultOptions alloc;
        size_t nin = session_->GetInputCount();
        qInfo() << "[ORT] num inputs =" << (qulonglong)nin;

        inputName_.clear();

        for (size_t i = 0; i < nin; ++i)
        {
            auto nmAlloc = session_->GetInputNameAllocated(i, alloc);
            std::string nm = nmAlloc.get();

            auto ti = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
            auto et = ti.GetElementType();

            size_t r = 0;
            try { r = ti.GetDimensionsCount(); } catch (...) { r = 0; }

            std::vector<int64_t> d;
            if (r > 0 && r < 16)
            {
                d.resize(r);
                try { ti.GetDimensions(d.data(), d.size()); } catch (...) { d.clear(); }
            }

            // 로깅
            QString dimStr;
            if (!d.empty())
            {
                dimStr = "[";
                for (size_t k = 0; k < d.size(); ++k)
                {
                    dimStr += QString::number((long long)d[k]);
                    if (k + 1 < d.size()) dimStr += ",";
                }
                dimStr += "]";
            }
            else
            {
                dimStr = "[?]";
            }

            qInfo() << "[ORT] input[" << (qulonglong)i << "] name =" << nm.c_str()
                    << " type=" << (int)et << " dims=" << dimStr;

            // 이미지 입력 후보: rank>=4 && C==3 (NCHW or NHWC)
            bool isImage = false;
            if (d.size() >= 4)
            {
                if ((d[1] == 3) || (d[3] == 3)) isImage = true;
            }

            if (isImage && et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && inputName_.empty())
            {
                inputName_ = nm; // 첫 번째 이미지 입력 채택
            }
        }

        // 최후의 수단 fallback
        if (inputName_.empty())
        {
            inputName_ = "inputNet_IN";
            qWarning() << "[ORT] image-like input not found; fallback to 'inputNet_IN'";
        }
    }

    //***FIX: 입력/출력 이름은 Allocated API로 받아서 std::string에 복사
    Ort::AllocatorWithDefaultOptions alloc;

    {
        auto inAlloc = session_->GetInputNameAllocated(0, alloc);   // AllocatedStringPtr
        inputName_ = inAlloc.get();                                  // std::string로 복사
    }

    size_t nOut = session_->GetOutputCount();
    outputNames_.clear();
    outputNamePtrs_.clear();

    for (size_t i = 0; i < nOut; ++i)
    {
        auto outAlloc = session_->GetOutputNameAllocated(i, alloc);  //***FIX
        outputNames_.push_back(outAlloc.get());                       // 복사
    }
    for (auto& s : outputNames_)
    {
        outputNamePtrs_.push_back(s.c_str()); // c_str() 포인터는 string이 살아있는 동안 유효
    }
    // ★ 입력 dtype/레이아웃 감지
    try
    {
        auto t2 = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        inputType_ = t2.GetElementType(); // FLOAT or UINT8 등
         size_t r2 = 0; try { r2 = t2.GetDimensionsCount(); } catch (...) { r2 = 0; }
        std::vector<int64_t> d2;
        if (r2 >= 4 && r2 < 16) {
            d2.resize(r2);
            try { t2.GetDimensions(d2.data(), d2.size()); } catch (...) { d2.clear(); }
        }
        // C=3 위치로 NHWC/NCHW 추정
        nhwc_ = (d2.size() >= 4 && d2[3] == 3);
        qInfo() << "[ORT] input dtype=" << (int)inputType_
                << " layout=" << (nhwc_ ? "NHWC" : "NCHW");
    }
    catch (...) {
        inputType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        nhwc_ = false;
        qWarning() << "[ORT] input meta detection failed; assume FLOAT NCHW";
    }
     // ★ 버퍼 할당 (dtype에 맞게)
    if (inputType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        inputBufU8_.resize(3 * inH_ * inW_);
    } else {
        inputBuf_.resize(3 * inH_ * inW_);
    }
    return true;
}

bool YoloXNanoLiteCpu::ready() const
{
    return session_ != nullptr;
}

cv::Mat YoloXNanoLiteCpu::letterbox(const cv::Mat& src, int dstW, int dstH, float& scale, cv::Point& pad)
{
    int w = src.cols, h = src.rows;
    float r = std::min(dstW / (float)w, dstH / (float)h);
    int newW = (int)std::round(w * r);
    int newH = (int)std::round(h * r);
    scale = r;
    pad = cv::Point((dstW - newW) / 2, (dstH - newH) / 2);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(dstH, dstW, src.type(), cv::Scalar(114,114,114));
    resized.copyTo(out(cv::Rect(pad.x, pad.y, newW, newH)));
    return out;
}

float YoloXNanoLiteCpu::iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    float inter = (a & b).area();
    float uni   = a.area() + b.area() - inter;
    return uni <= 0 ? 0.f : inter / uni;
}

//***FIX: QRectF 기반 IoU 구현
static float qiou(const QRectF& a, const QRectF& b)
{
    QRectF inter = a.intersected(b);
    if (inter.isEmpty())
    {
        return 0.0f;
    }
    float inter_area = (float)(inter.width() * inter.height());
    float a_area     = (float)(a.width() * a.height());
    float b_area     = (float)(b.width() * b.height());
    float uni        = a_area + b_area - inter_area;
    return (uni <= 0.0f) ? 0.0f : (inter_area / uni);
}

std::vector<YXBox> YoloXNanoLiteCpu::nms(const std::vector<YXBox>& boxes, float iouTh)
{
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j)
    {
        return boxes[i].score > boxes[j].score;
    });

    std::vector<YXBox> kept;
    std::vector<char> removed(boxes.size(), 0);

    for (size_t a = 0; a < order.size(); ++a)
    {
        int i = order[a];
        if (removed[i]) continue;
        kept.push_back(boxes[i]);

        for (size_t b = a + 1; b < order.size(); ++b)
        {
            int j = order[b];
            if (removed[j]) continue;

            //***FIX: QRectF 기반 IoU 사용
            if (qiou(boxes[i].rect, boxes[j].rect) > iouTh)
            {
                removed[j] = 1;
            }
        }
    }
    return kept;
}

std::vector<YXBox> YoloXNanoLiteCpu::detect(const QImage& qimgIn, float scoreTh, float nmsIouTh)
{
    qInfo() << "[SANITY] qimgIn WxH =" << qimgIn.width() << "x" << qimgIn.height();
    qInfo() << "[SANITY] inW x inH (model input) =" << inW_ << "x" << inH_;
    qInfo() << "[SANITY] inputBuf size =" << (qulonglong)inputBuf_.size()
            << " need =" << (qulonglong)(1ull*3*inW_*inH_);

    std::vector<YXBox> out;

    if (!session_)
        return out;

    // QImage -> cv::Mat BGR
    QImage rgb = qimgIn.convertToFormat(QImage::Format_RGB888);
    cv::Mat src(rgb.height(), rgb.width(), CV_8UC3, const_cast<uchar*>(rgb.bits()), rgb.bytesPerLine());
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);

    // letterbox -> RGB (uint8 유지; float 필요 시에만 변환)
    float scale = 1.f;
    cv::Point pad(0,0);
    cv::Mat lb = letterbox(bgr, inW_, inH_, scale, pad);

    cv::Mat lbRGB8;
    cv::cvtColor(lb, lbRGB8, cv::COLOR_BGR2RGB); // CV_8UC3 (연속일 확률 높음)

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inTensor{nullptr};
     if (inputType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
    {
        // === UINT8 경로 ===
        if (nhwc_)
        {
            // NHWC: OpenCV HWC 메모리를 그대로 사용
            if (!lbRGB8.isContinuous()) lbRGB8 = lbRGB8.clone();
            std::array<int64_t,4> inShapeNHWC{1, inH_, inW_, 3};
            const size_t u8count = (size_t)inH_ * inW_ * 3;
            inTensor = Ort::Value::CreateTensor<uint8_t>(
                memInfo, lbRGB8.data, u8count, inShapeNHWC.data(), inShapeNHWC.size()
            );
        }
        else
        {
            // NCHW: CHW 플래너로 재배열
            std::vector<cv::Mat> chU8(3);
            for (int c = 0; c < 3; ++c) {
                chU8[c] = cv::Mat(inH_, inW_, CV_8U, inputBufU8_.data() + c * inH_ * inW_);
            }
            cv::split(lbRGB8, chU8); // HWC → CHW
             std::array<int64_t,4> inShapeNCHW{1, 3, inH_, inW_};
            inTensor = Ort::Value::CreateTensor<uint8_t>(
                memInfo, inputBufU8_.data(), inputBufU8_.size(), inShapeNCHW.data(), inShapeNCHW.size()
            );
        }
    }
    else
    {
        // === FLOAT 경로 (기존과 동일) ===
        cv::Mat lbRGBf;
        lbRGB8.convertTo(lbRGBf, CV_32F, 1.0f/255.0f);
         std::vector<cv::Mat> chF(3);
        for (int c = 0; c < 3; ++c) {
            chF[c] = cv::Mat(inH_, inW_, CV_32F, inputBuf_.data() + c * inH_ * inW_);
        }
        cv::split(lbRGBf, chF); // HWC → CHW
         std::array<int64_t,4> inShapeF{1, 3, inH_, inW_};
        inTensor = Ort::Value::CreateTensor<float>(
            memInfo, inputBuf_.data(), inputBuf_.size(), inShapeF.data(), inShapeF.size()
        );
    }


    // ★ 입력 이름을 세션에서 "수명 안전"하게 가져와 사용
    Ort::AllocatorWithDefaultOptions alloc;
    auto nmAlloc = session_->GetInputNameAllocated(0, alloc);
    std::string realInputName = nmAlloc.get();      // e.g., "inputNet_IN"
    const char* input_names[] = { realInputName.c_str() };
     // ★ 한 번 만든 텐서를 그대로 사용 (중복 move 금지)
    const Ort::Value* inputs_arr = &inTensor;
    std::vector<Ort::Value> outputs;
    try
    {
        // 6-인자 Run (네 ORT 시그니처)
        outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,                 // const char* const*
            inputs_arr,                  // const Ort::Value*
            1,                           // input_count
            outputNamePtrs_.data(),      // const char* const*
            outputNamePtrs_.size()       // output_count
        );
    }
    catch (const Ort::Exception& e)
    {
        qWarning() << "[ORT] Run failed:" << e.what() <<::endl;
        return {}; // 크래시 대신 빈 결과
    }

    if (outputs.empty())
    {
        qWarning() << "[ORT] no outputs returned";
        return {};
    }

    // 가장 흔한 한 개 텐서: [1,N,85] 또는 [1,1,N,85]
    Ort::Value& o0 = outputs[0];
    auto oinfo = o0.GetTensorTypeAndShapeInfo();
    auto oshp  = oinfo.GetShape();
    const float* pdata = o0.GetTensorData<float>();

    size_t N = 0;
    int stride = 85;

    if (oshp.size() == 3 && oshp[0] == 1 && oshp[2] == 85)
    {
        N = static_cast<size_t>(oshp[1]);
        stride = static_cast<int>(oshp[2]);
    }
    else if (oshp.size() == 4 && oshp[0] == 1 && oshp[3] == 85)
    {
        N = static_cast<size_t>(oshp[2]);
        stride = static_cast<int>(oshp[3]);
    }
    else
    {
        // TODO: 다른 포맷(두 텐서 분리 등) 처리
        return out;
    }

    std::vector<YXBox> raw;
    raw.reserve(N);

    const int srcW = qimgIn.width();
    const int srcH = qimgIn.height();

    for (size_t i = 0; i < N; ++i)
    {
        const float* p = pdata + i * stride;
        float cx = p[0], cy = p[1], w = p[2], h = p[3];
        float obj = p[4];

        // person(cls=0)만
        float cls0 = p[5 + 0];
        float score = obj * cls0;
        if (score < scoreTh) continue;

        float x0 = cx - w * 0.5f;
        float y0 = cy - h * 0.5f;

        float x  = (x0 - pad.x) / scale;
        float y  = (y0 - pad.y) / scale;
        float rw =  w / scale;
        float rh =  h / scale;

        x  = std::max(0.f, std::min(x,  (float)srcW - 1));
        y  = std::max(0.f, std::min(y,  (float)srcH - 1));
        rw = std::max(0.f, std::min(rw, (float)srcW - x));
        rh = std::max(0.f, std::min(rh, (float)srcH - y));

        YXBox b;
        b.rect  = QRectF(x, y, rw, rh);
        b.score = score;
        b.cls   = 0;
        raw.push_back(b);
    }

    out = nms(raw, nmsIouTh);
    return out;
}
