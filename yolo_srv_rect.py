# -*- coding: utf-8 -*-
import os, socket, struct, json, time
import numpy as np
from PIL import Image
import onnxruntime as rt

# ---------- 설정 ----------
SOCK_PATH = "/tmp/yolo.sock"
MODEL_DIR = "/opt/model_zoo/ONR-OD-8200-yolox-nano-lite-mmdet-coco-416x416/model"
ART_DIR   = "/opt/model_zoo/ONR-OD-8200-yolox-nano-lite-mmdet-coco-416x416/artifacts"
IN_SIZE   = int(os.getenv("IN_SIZE", "416"))
SCORE_TH  = float(os.getenv("SCORE_TH", "0.20"))
NMS_IOU   = float(os.getenv("NMS_IOU", "0.50"))
PAD_VAL   = 114
VERBOSE   = int(os.getenv("VERBOSE", "1"))   # 1=로그 켜기, 0=끄기
LOG_EVERY = int(os.getenv("LOG_EVERY", "1")) # N프레임마다 로그 (기본 매 프레임)
CLASS_ID  = int(os.getenv("CLASS_ID", "0"))  # COCO person=0, -1이면 전체 허용
MIN_AREA  = float(os.getenv("MIN_AREA", "400")) # (px) 너무 작은 박스 제거(320x240 기준)
AR_MIN    = float(os.getenv("AR_MIN", "0.30"))  # w/h 최소
AR_MAX    = float(os.getenv("AR_MAX", "1.20"))  # w/h 최대
PERSIST   = int(os.getenv("PERSIST", "0"))      # 1이면 직전 프레임과 IoU 매칭 필요
IOU_TRACK = float(os.getenv("IOU_TRACK", "0.30"))


# (가능하면 cv2로 CLAHE; 없으면 건너뜀)
try:
    import cv2
    HAS_CV2=True
except Exception:
    HAS_CV2=False
CLAHE = int(os.getenv("CLAHE", "1")) and HAS_CV2  # 1이면 CLAHE 적용

def find_onnx(d):
    for f in os.listdir(d):
        if f.lower().endswith(".onnx"):
            return os.path.join(d,f)
    raise FileNotFoundError("onnx not found in " + d)

def letterbox(np_rgb, W, H, pad=114):
    h, w, _ = np_rgb.shape
    r = min(W/float(w), H/float(h))
    nw, nh = int(round(w*r)), int(round(h*r))
    canvas = np.full((H,W,3), pad, dtype=np.uint8)
    resized = np.array(Image.fromarray(np_rgb).resize((nw,nh), Image.BILINEAR))
    x0 = (W - nw)//2; y0 = (H - nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas, r, x0, y0

def nms_xyxy(boxes, scores, iou_th):
    if not boxes: return []
    b = np.array(boxes, np.float32)
    s = np.array(scores, np.float32)
    keep=[]
    order = s.argsort()[::-1]
    while order.size>0:
        i = order[0]; keep.append(i)
        if order.size==1: break
        rest=order[1:]
        xx1=np.maximum(b[i,0],b[rest,0]); yy1=np.maximum(b[i,1],b[rest,1])
        xx2=np.minimum(b[i,2],b[rest,2]); yy2=np.minimum(b[i,3],b[rest,3])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        ua=(b[i,2]-b[i,0])*(b[i,3]-b[i,1]) + (b[rest,2]-b[rest,0])*(b[rest,3]-b[rest,1]) - inter
        iou=inter/np.maximum(ua,1e-6)
        order=rest[iou<=iou_th]
    return keep
    
def apply_geo_filters(boxes, scores, W0, H0):
    """면적/종횡비 기반 1차 필터"""
    f_boxes=[]; f_scores=[]
    for (x1,y1,x2,y2), sc in zip(boxes, scores):
        w = max(0.0, x2-x1); h = max(0.0, y2-y1)
        area = w*h
        ar = w / (h + 1e-6)
        if area < MIN_AREA:                  # 너무 작은 박스
            continue
        if not (AR_MIN <= ar <= AR_MAX):     # 사람 비슷한 종횡비만
            continue
        f_boxes.append([x1,y1,x2,y2]); f_scores.append(sc)
    return f_boxes, f_scores

def iou_xyxy(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    xx1=max(ax1,bx1); yy1=max(ay1,by1)
    xx2=min(ax2,bx2); yy2=min(ay2,by2)
    inter = max(0.0, xx2-xx1) * max(0.0, yy2-yy1)
    ua = max(0.0,(ax2-ax1))*max(0.0,(ay2-ay1)) + max(0.0,(bx2-bx1))*max(0.0,(by2-by1)) - inter
    return inter / max(ua, 1e-6)

def preprocess_gray8_to_rgb3(gray, stride, H, W):
    # gray: bytes; stride: bytes/row
    row = np.frombuffer(gray, dtype=np.uint8)
    img = row.reshape(H, stride)[:, :W]           # remove padding
    if CLAHE:
        # 조금 더 또렷하게 (선택)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img = clahe.apply(img)
    rgb = np.stack([img, img, img], axis=-1)
    return rgb

def load_session():
    onnx = find_onnx(MODEL_DIR)
    sess = rt.InferenceSession(
        onnx,
        providers=["TIDLExecutionProvider","CPUExecutionProvider"],
        provider_options=[{"artifacts_folder":ART_DIR,"debug_level":0}, {}]
    )
    inp = sess.get_inputs()[0]
    out_names = [o.name for o in sess.get_outputs()]
    in_type = inp.type
    want_u8 = ("tensor(uint8)" in in_type)
    print("[LOAD]", onnx)
    print(" input:", inp.name, in_type, inp.shape)
    print(" outs :", out_names)
    print(" want_uint8_input:", bool(want_u8))
    return sess, inp.name, out_names, want_u8

sess, in_name, out_names, want_u8 = load_session()

def infer_boxes(np_rgb):
    H0, W0 = np_rgb.shape[:2]
    lb, r, px, py = letterbox(np_rgb, IN_SIZE, IN_SIZE, PAD_VAL)
    x = np.transpose(lb,(2,0,1))[None,...]
    x = x.astype(np.uint8) if want_u8 else (x.astype(np.float32)/255.0)
    y = sess.run(out_names, {in_name: x})

    boxes=[]; scores=[]; parsed=False
    # Case A: (N,5) => x1,y1,x2,y2,score
    try:
        y0 = np.array(y[0])                 # dets
        y1_lab = np.array(y[1]) if len(y)>1 else None  # labels
        if y0.ndim>=2 and y0.shape[-1]==5:
            det = y0.reshape(-1,5)
            labels = y1_lab.reshape(-1) if y1_lab is not None else None
            for i in range(det.shape[0]):
                x1,y1,x2,y2,sc = map(float, det[i])
                if sc < SCORE_TH: continue
                if CLASS_ID >= 0:
                    cls_i = int(labels[i]) if (labels is not None and i < labels.size) else -1
                    if cls_i != CLASS_ID:
                        continue
                x1=(x1-px)/r; y1=(y1-py)/r; x2=(x2-px)/r; y2=(y2-py)/r
                x1=max(0,min(W0-1,x1)); y1=max(0,min(H0-1,y1))
                x2=max(0,min(W0-1,x2)); y2=max(0,min(H0-1,y2))
                boxes.append([x1,y1,x2,y2]); scores.append(sc)
            boxes, scores = apply_geo_filters(boxes, scores, W0, H0)
            parsed=True
    except Exception:
        parsed=False

    # Case B: YOLOX [1,N,85] / [1,1,N,85]
    if not parsed:
        out0=np.array(y[0])
        if out0.ndim==3 and out0.shape[0]==1:
            data=out0.reshape(out0.shape[1], out0.shape[2])
        elif out0.ndim==4 and out0.shape[0]==1:
            data=out0.reshape(out0.shape[2], out0.shape[3])
        else:
            data=np.zeros((0,85), np.float32)
        for i in range(data.shape[0]):
            cx,cy,bw,bh,obj = map(float, data[i,0:5])
            cls0 = float(data[i,5])  # person
            sc = obj*cls0
            if sc < SCORE_TH: continue
            x1=(cx-0.5*bw-px)/r; y1=(cy-0.5*bh-py)/r
            x2=(cx+0.5*bw-px)/r; y2=(cy+0.5*bh-py)/r
            x1=max(0,min(W0-1,x1)); y1=max(0,min(H0-1,y1))
            x2=max(0,min(W0-1,x2)); y2=max(0,min(H0-1,y2))
            boxes.append([x1,y1,x2,y2]); scores.append(sc)

    if not boxes: return []

    keep = nms_xyxy(boxes, scores, NMS_IOU)
    out = [[boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], float(scores[i])] for i in keep]
    return out

def recv_all(sock, n):
    buf=bytearray(n)
    view=memoryview(buf)
    got=0
    while got<n:
        r = sock.recv_into(view[got:], n-got)
        if r<=0: raise ConnectionError("socket closed")
        got += r
    return buf

def run_server():
    try: os.unlink(SOCK_PATH)
    except FileNotFoundError: pass

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(SOCK_PATH)
    os.chmod(SOCK_PATH, 0o777)
    s.listen(1)
    print("[SRV] listening at", SOCK_PATH)

    while True:
        conn, _ = s.accept()
        print("[SRV] client connected")
        try:
            frame_cnt = 0
            prev_boxes = []
            while True:
                # 헤더(고정 28바이트): magic(4s), fmt(u32), w(u32), h(u32), stride(u32), frame_id(u32), payload(u32)
                hdr = recv_all(conn, 28)
                magic, fmt, W, H, stride, fid, payload = struct.unpack("!4sIIIIII", hdr)
                if magic != b"FRM1": raise ValueError("bad magic")
                if payload == 0: continue
                raw = recv_all(conn, payload)
                
                # 페이로드 예상치 검사
                need = stride * H
                if VERBOSE and payload != need:
                    print(f"[SRV][WARN] payload({payload}) != stride*H({need})", flush=True)
                if VERBOSE and (frame_cnt % LOG_EVERY == 0):
                    print(f"[SRV] recv fid={fid} fmt={fmt} {W}x{H} stride={stride} payload={payload}", flush=True)


                if fmt == 0:  # GRAY8
                    rgb = preprocess_gray8_to_rgb3(raw, stride, H, W)
                elif fmt == 1:  # RGB888
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape(H, stride)[:, :W*3]
                    rgb = arr.reshape(H, W, 3)
                else:
                    rgb = np.zeros((H,W,3), np.uint8)

                t0 = time.time()
                boxes = infer_boxes(rgb)  # ✅ 한 번만 호출
                t1 = time.time()
                # 간단한 지속성(이전 프레임과 IoU 매칭되는 박스만 유지)
                if PERSIST:
                    kept = []
                    for b in boxes:
                        if any(iou_xyxy(b[:4], pb[:4]) >= IOU_TRACK for pb in prev_boxes):
                            kept.append(b)
                    boxes = kept
                prev_boxes = boxes[:]  # 다음 프레임 대비
                top = max((b[4] for b in boxes), default=0.0)
                frame_cnt += 1
                if VERBOSE and (frame_cnt % LOG_EVERY == 0):
                    print(f"[SRV] det={len(boxes)} top={top:.2f} infer={int((t1-t0)*1000)}ms", flush=True)
 

                resp = json.dumps({"frame_id": int(fid), "boxes": boxes}, separators=(",",":"))+"\n"
                conn.sendall(resp.encode("utf-8"))
        except Exception as e:
            print("[SRV] client error:", e, flush=True)
        finally:
            conn.close()
            print("[SRV] client closed")

if __name__ == "__main__":
    run_server()