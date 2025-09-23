#include "pose_estimate_onnx.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// COCO 17 skeleton edges (idx: nose, eye, ear, shoulder, elbow, wrist, hip, knee, ankle)
static const int EDGES[][2] = {
    {5,7},{7,9}, {6,8},{8,10}, {5,6}, {11,12},
    {5,11},{6,12}, {11,13},{13,15}, {12,14},{14,16},
    {0,5},{0,6},{0,1},{1,3},{0,2},{2,4}
};

static inline float iou(float ax,float ay,float aw,float ah, float bx,float by,float bw,float bh){
    float x1 = std::max(ax, bx), y1 = std::max(ay, by);
    float x2 = std::min(ax+aw, bx+bw), y2 = std::min(ay+ah, by+bh);
    float iw = std::max(0.f, x2-x1), ih = std::max(0.f, y2-y1);
    float inter = iw*ih, uni = aw*ah + bw*bh - inter;
    return (uni>0)? inter/uni : 0.f;
}

bool PoseEstimatorONNX::init(const QString& path, const PoseParams& P, int threads){
    P_ = P;
    auto* env  = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "pose");
    Ort::SessionOptions so; so.SetIntraOpNumThreads(threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    auto* sess = new Ort::Session(*env, path.toStdString().c_str(), so);
    auto* mem  = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    // 이름 캐시
    Ort::AllocatorWithDefaultOptions a;
    size_t ni = sess->GetInputCount(), no = sess->GetOutputCount();
    in_names_.resize(ni);  out_names_.resize(no);
    in_keep_.reserve(ni);  out_keep_.reserve(no);
    for(size_t i=0;i<ni;++i){ auto s=sess->GetInputNameAllocated(i,a); in_names_[i]=s.get(); in_keep_.emplace_back(std::move(s)); }
    for(size_t i=0;i<no;++i){ auto s=sess->GetOutputNameAllocated(i,a); out_names_[i]=s.get(); out_keep_.emplace_back(std::move(s)); }

    env_ = env; sess_ = sess; mem_ = mem;
    return true;
}

void PoseEstimatorONNX::preprocess(const uchar* input, int w, int h, int stride, float* out){
    float sx = P_.inputW/(float)w, sy = P_.inputH/(float)h;
    float s = P_.letterbox? std::min(sx,sy): sx;
    int nw = int(w*s), nh = int(h*s), dx=(P_.inputW-nw)/2, dy=(P_.inputH-nh)/2;

    // fill 114/255
    std::fill(out, out+3*P_.inputW*P_.inputH, 114.f/255.f);

    for(int y=0;y<nh;++y){
        const uchar* py = input + int(y/s)*stride;
        for(int x=0;x<nw;++x){
            int sx0 = int(x/s);
            const uchar* px = py + (P_.rgbInput? sx0*3 : sx0);
            float r = px[0], g = P_.rgbInput? px[1]:px[0], b=P_.rgbInput? px[2]:px[0];
            int dx0 = x+dx, dy0 = y+dy, di = dy0*P_.inputW + dx0;
            out[0*P_.inputW*P_.inputH + di] = r/255.f;
            out[1*P_.inputW*P_.inputH + di] = g/255.f;
            out[2*P_.inputW*P_.inputH + di] = b/255.f;
        }
    }
}

void PoseEstimatorONNX::nms(const std::vector<PosePerson>& in, std::vector<PosePerson>& out){
    auto v = in;
    std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.score>b.score; });
    std::vector<char> sup(v.size(), 0);
    for(size_t i=0;i<v.size();++i){
        if(sup[i]) continue;
        out.push_back(v[i]);
        for(size_t j=i+1;j<v.size();++j)
            if(iou(v[i].x,v[i].y,v[i].w,v[i].h, v[j].x,v[j].y,v[j].w,v[j].h) > P_.nmsIoU) sup[j]=1;
    }
}

void PoseEstimatorONNX::scaleBack(PosePerson& p){
    // 입력->원본 되돌림 (letterbox 기준 간단 복원)
    // 실제론 preprocess의 s,dx,dy를 보관해서 쓰는 게 정석이지만,
    // 여기선 P_.inputW/H == 모델 입력 고정 가정하에 원본 해상도와 동일 비율이면 충분.
    // 필요 시 외부에서 보정해도 OK.
}

bool PoseEstimatorONNX::infer(const uchar* data, int w, int h, int stride, std::vector<PosePerson>& out){
    out.clear();
    if(!sess_||!mem_) return false;

    const int C = 3, IH=P_.inputH, IW=P_.inputW;
    std::vector<float> in(C*IH*IW);
    preprocess(data,w,h,stride,in.data());

    std::array<int64_t, 4> shape = {1, (int64_t)C, (int64_t)IH, (int64_t)IW};
    Ort::Value tin = Ort::Value::CreateTensor<float>(
        *reinterpret_cast<Ort::MemoryInfo*>(mem_),   // 또는 *(Ort::MemoryInfo*)mem_
        in.data(), static_cast<size_t>(in.size()),
        shape.data(), shape.size());

    auto* sess=(Ort::Session*)sess_;
    auto tout = sess->Run(Ort::RunOptions{nullptr}, in_names_.data(), &tin, 1, out_names_.data(), out_names_.size());

    // YOLOv8-pose ONNX: 보통 [1, num, 56] (bbox4 + obj1 + kpts(17*3))
    // 일부 빌드에선 [1,56,num]일 수도 있어 둘 다 처리
    auto& ov = tout[0];
    auto info = ov.GetTensorTypeAndShapeInfo();
    auto shp  = info.GetShape(); // e.g., {1, N, 56} or {1, 56, N}
    float* p  = ov.GetTensorMutableData<float>();

    int N=0, S=0; bool transposed=false;
    if(shp.size()==3 && shp[2]==56){ N = (int)shp[1]; S=56; transposed=false; }
    else if(shp.size()==3 && shp[1]==56){ N = (int)shp[2]; S=56; transposed=true; }
    else { return false; }

    std::vector<PosePerson> cand; cand.reserve(N);
    for(int i=0;i<N;++i){
        const float* row = transposed? (p + i) : (p + i*S);
        float cx,cy,w0,h0,obj;
        if(transposed){
            // [1,56,N] → 각 요소가 56 stride로 N씩 떨어짐
            auto at=[&](int k){ return row[k*N]; };
            cx=at(0); cy=at(1); w0=at(2); h0=at(3); obj=at(4);
        }else{
            cx=row[0]; cy=row[1]; w0=row[2]; h0=row[3]; obj=row[4];
        }
        if(obj < P_.confDet) continue;

        PosePerson per; per.x=cx-w0*0.5f; per.y=cy-h0*0.5f; per.w=w0; per.h=h0; per.score=obj;
        per.kpts.resize(17);
        for(int k=0;k<17;++k){
            int base = 5 + k*3;
            float kx = transposed? row[(base+0)*N] : row[base+0];
            float ky = transposed? row[(base+1)*N] : row[base+1];
            float kc = transposed? row[(base+2)*N] : row[base+2];
            per.kpts[k] = {kx, ky, kc};
        }
        cand.push_back(std::move(per));
    }

    nms(cand, out);
    return true;
}
