#include "person_detect_onnx.h"
#include "onnxruntime_cxx_api.h"
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace std;

#define PERSON_CLASS_ID 0  // COCO dataset에서 person 클래스 ID

static inline float iou(const PDBox& a, const PDBox& b)
{
    float interW = max(0.f, min(a.x + a.w, b.x + b.w) - max(a.x, b.x));
    float interH = max(0.f, min(a.y + a.h, b.y + b.h) - max(a.y, b.y));
    float inter = interW * interH;
    float unionArea = a.w * a.h + b.w * b.h - inter;
    return inter / unionArea;
}

bool PersonDetectorONNX::init(const QString& modelPath, const PDParams& params, int threads)
{
    P_ = params;

    Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "pd");
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session* sess = new Ort::Session(*env, modelPath.toStdString().c_str(), so);
    Ort::MemoryInfo* mem = new Ort::MemoryInfo(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    env_ = env;
    session_ = sess;
    memoryInfo_ = mem;
    return true;
}

void PersonDetectorONNX::preprocess(const uchar* input, int w, int h, int stride, float* out)
{
    float scaleX = P_.inputW / float(w);
    float scaleY = P_.inputH / float(h);
    float scale = P_.letterbox ? min(scaleX, scaleY) : scaleX;
    int nw = int(w * scale);
    int nh = int(h * scale);
    int dx = (P_.inputW - nw) / 2;
    int dy = (P_.inputH - nh) / 2;

    for (int y = 0; y < P_.inputH; ++y)
    {
        for (int x = 0; x < P_.inputW; ++x)
        {
            int dstIdx = y * P_.inputW + x;
            for (int c = 0; c < 3; ++c)
                out[c * P_.inputW * P_.inputH + dstIdx] = 114.0f / 255.0f;
        }
    }

    for (int y = 0; y < nh; ++y)
    {
        const uchar* py = input + int(y / scale) * stride;
        for (int x = 0; x < nw; ++x)
        {
            int srcX = int(x / scale);
            const uchar* px = py + (P_.rgbInput ? srcX * 3 : srcX);
            int r = P_.rgbInput ? px[0] : px[0];
            int g = P_.rgbInput ? px[1] : px[0];
            int b = P_.rgbInput ? px[2] : px[0];

            int dstX = x + dx;
            int dstY = y + dy;
            int dstIdx = dstY * P_.inputW + dstX;
            out[0 * P_.inputW * P_.inputH + dstIdx] = r / 255.0f;
            out[1 * P_.inputW * P_.inputH + dstIdx] = g / 255.0f;
            out[2 * P_.inputW * P_.inputH + dstIdx] = b / 255.0f;
        }
    }
}

void PersonDetectorONNX::scaleCoords(PDBox& box)
{
    float scale = min(P_.inputW / float(P_.inputW), P_.inputH / float(P_.inputH));
    float dx = (P_.inputW - P_.inputW * scale) / 2;
    float dy = (P_.inputH - P_.inputH * scale) / 2;

    box.x = max(0.f, (box.x - dx) / scale);
    box.y = max(0.f, (box.y - dy) / scale);
    box.w /= scale;
    box.h /= scale;
}

void PersonDetectorONNX::nms(const vector<PDBox>& in, vector<PDBox>& out)
{
    vector<PDBox> sorted = in;
    sort(sorted.begin(), sorted.end(), [](const PDBox& a, const PDBox& b) {
        return a.score > b.score;
    });

    vector<bool> suppressed(sorted.size(), false);
    for (size_t i = 0; i < sorted.size(); ++i)
    {
        if (suppressed[i]) continue;
        out.push_back(sorted[i]);
        for (size_t j = i + 1; j < sorted.size(); ++j)
        {
            if (iou(sorted[i], sorted[j]) > P_.nmsIoU)
                suppressed[j] = true;
        }
    }
}

bool PersonDetectorONNX::detect(const uchar* input, int w, int h, int strideBytes, vector<PDBox>& outBoxes)
{
    outBoxes.clear();
    if (!session_ || !memoryInfo_) return false;

    const int chw = 3 * P_.inputH * P_.inputW;

    std::vector<float> inputTensor(chw);
    preprocess(input, w, h, strideBytes, inputTensor.data());

    std::array<int64_t, 4> inputShape = { 1, 3, P_.inputH, P_.inputW };

    Ort::Value inputOrt = Ort::Value::CreateTensor<float>(
        *(Ort::MemoryInfo*)memoryInfo_, inputTensor.data(), chw,
        inputShape.data(), inputShape.size());

    //변경
    Ort::Session* sess = (Ort::Session*)session_;
    Ort::AllocatorWithDefaultOptions allocator;

    //변경
    size_t n_in = sess->GetInputCount();
    std::vector<const char*> input_names(n_in);
    std::vector<Ort::AllocatedStringPtr> in_name_keeper;
    in_name_keeper.reserve(n_in);
    for (size_t i = 0; i < n_in; ++i) {
        auto n = sess->GetInputNameAllocated(i, allocator);
        input_names[i] = n.get();
        in_name_keeper.emplace_back(std::move(n));
    }

    //변경
    size_t n_out = sess->GetOutputCount(); // 보통 1
    std::vector<const char*> output_names(n_out);
    std::vector<Ort::AllocatedStringPtr> out_name_keeper;
    out_name_keeper.reserve(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        auto n = sess->GetOutputNameAllocated(i, allocator);
        output_names[i] = n.get();
        out_name_keeper.emplace_back(std::move(n));
    }

    //수정
    std::vector<Ort::Value> outputs = sess->Run(
        Ort::RunOptions{nullptr},
        input_names.data(), &inputOrt, 1,
        output_names.data(), n_out);

    float* pred = outputs[0].GetTensorMutableData<float>();

    std::vector<PDBox> cand;
    cand.reserve(128);

    for (int i = 0; i < 8400; ++i)
    {
        float obj = pred[i * 85 + 4];
        float cls = pred[i * 85 + 5 + PERSON_CLASS_ID];
        float score = obj * cls;
        if (score < P_.confThresh) continue;

        float cx = pred[i * 85 + 0];
        float cy = pred[i * 85 + 1];
        float ww = pred[i * 85 + 2];
        float hh = pred[i * 85 + 3];

        PDBox b;
        b.x = cx - ww * 0.5f;
        b.y = cy - hh * 0.5f;
        b.w = ww;
        b.h = hh;
        b.score = score;
        b.label = PERSON_CLASS_ID;

        scaleCoords(b);
        cand.push_back(b);
    }

    nms(cand, outBoxes);
    return true;
}
