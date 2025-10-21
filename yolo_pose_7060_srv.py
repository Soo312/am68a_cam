# -*- coding: utf-8 -*-
import os, socket, struct, json, time
import numpy as np
from PIL import Image
import onnxruntime as rt

# ================== 환경 변수 ==================
SOCK_PATH   = os.getenv("SOCK_PATH", "/tmp/yolo_pose.sock")

POSE_DIR    = os.getenv("POSE_DIR", "/opt/model_zoo/ONR-KD-7060-human-pose-yolox-s-640x640/model")
POSE_ART    = os.getenv("POSE_ART", "/opt/model_zoo/ONR-KD-7060-human-pose-yolox-s-640x640/artifacts")
POSE_IN     = int(os.getenv("POSE_IN", "640"))    # 모델이 고정 크기면 자동으로 맞춰줌
SCORE_TH    = float(os.getenv("SCORE_TH", "0.50"))
KPT_TH      = float(os.getenv("KPT_TH", "0.45"))
N_TOP       = int(os.getenv("N_TOP", "2"))        # 상위 K개만 반환(0=모두)
MAX_FPS     = float(os.getenv("MAX_FPS", "0"))    # 0=무제한
VERBOSE     = int(os.getenv("VERBOSE", "1"))
NORM_01     = int(os.getenv("NORM_01", "0"))      # 중요: 이번 모델은 raw 0..255가 잘 맞음
PAD_VAL     = int(os.getenv("PAD_VAL", "114"))
ORT_THREADS = int(os.getenv("ORT_THREADS", "2"))
WARMUP      = int(os.getenv("WARMUP", "1"))

# fmt (헤더의 포맷 코드): 0=GRAY8, 1=RGB888
FMT_GRAY8 = 0
FMT_RGB24 = 1

# ================== 공용 유틸 ==================
def find_onnx(d):
    for f in os.listdir(d):
        if f.lower().endswith(".onnx"):
            return os.path.join(d, f)
    raise FileNotFoundError("onnx not found in " + d)

def letterbox_into(np_rgb, dst_hw, pad=114, canvas=None):
    """np_rgb(H,W,3)을 정사각 dst_hw×dst_hw로 letterbox. canvas 재사용."""
    H, W = np_rgb.shape[:2]
    outH = outW = int(dst_hw)
    r = min(outW / float(W), outH / float(H))
    nw, nh = int(round(W * r)), int(round(H * r))
    if canvas is None or canvas.shape[0] != outH or canvas.shape[1] != outW:
        canvas = np.full((outH, outW, 3), pad, dtype=np.uint8)
    else:
        canvas[:] = pad  # 재사용 시 초기화
    if nw > 0 and nh > 0:
        resized = np.array(Image.fromarray(np_rgb).resize((nw, nh), Image.BILINEAR))
        x0 = (outW - nw) // 2
        y0 = (outH - nh) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
    else:
        x0 = y0 = 0
    return canvas, r, x0, y0

def preprocess_gray8_to_rgb3(raw_bytes, stride, H, W):
    row = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = row.reshape(H, stride)[:, :W]
    rgb = np.stack([img, img, img], axis=-1)
    return rgb

def preprocess_rgb888(raw_bytes, stride, H, W):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(H, stride)[:, :W*3]
    return arr.reshape(H, W, 3)

# ================== 포즈 세션 ==================
pose_sess = None
pose_in_name = None
pose_out_names = None
pose_u8 = False
pose_in_size = POSE_IN

# 재사용 버퍼
_canvas = None
_xbuf   = None  # (1,3,H,W) float32/uint8

def load_pose_session():
    global pose_sess, pose_in_name, pose_out_names, pose_u8, pose_in_size, _xbuf, _canvas

    onnx = find_onnx(POSE_DIR)
    so = rt.SessionOptions()
    try:
        so.intra_op_num_threads = ORT_THREADS
    except Exception:
        pass
    try:
        so.set_graph_optimization_level(rt.GraphOptimizationLevel.ORT_ENABLE_BASIC)
    except Exception:
        pass

    pose_sess = rt.InferenceSession(
        onnx, so,
        providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"artifacts_folder": POSE_ART, "debug_level": 0}, {}]
    )
    inp = pose_sess.get_inputs()[0]
    pose_in_name = inp.name
    pose_out_names = [o.name for o in pose_sess.get_outputs()]
    pose_u8 = ("tensor(uint8)" in inp.type)

    print("[POSE-LOAD]", onnx)
    print(" input:", inp.name, inp.type, inp.shape)
    print(" outs :", pose_out_names)
    print(" want_uint8_input:", bool(pose_u8))

    # 모델 입력 크기 동기화(BCHW 가정)
    try:
        h, w = int(inp.shape[2]), int(inp.shape[3])
        if h > 0 and w > 0 and h == w:
            if pose_in_size != h:
                print(f"[POSE] override POSE_IN: {pose_in_size} -> {h}")
            pose_in_size = h
    except Exception:
        pose_in_size = POSE_IN

    # 버퍼 준비
    _canvas = np.full((pose_in_size, pose_in_size, 3), PAD_VAL, dtype=np.uint8)
    if pose_u8:
        _xbuf = np.zeros((1, 3, pose_in_size, pose_in_size), dtype=np.uint8)
    else:
        _xbuf = np.zeros((1, 3, pose_in_size, pose_in_size), dtype=np.float32)

    # 워밍업
    if WARMUP:
        try:
            pose_sess.run(pose_out_names, {pose_in_name: _xbuf})
        except Exception as e:
            print("[POSE][WARMUP-IGNORED]", e)

def _find_kstart(D):
    """
    출력 row 길이 D에서 keypoint 시작 인덱스 추정.
    - 56: [x1,y1,x2,y2,score, 17*(x,y,score)] -> kstart=5
    - 57+: [x1,y1,x2,y2,score,cls, 17*(x,y,score)] -> kstart=6
    """
    if D >= 56:
        if (D - 5) % 3 == 0:  # cls 없음
            return 5, False
        if (D - 6) % 3 == 0:  # cls 포함
            return 6, True
    # 실패 시 보수적으로 5 가정
    return 5, False

def infer_pose_fullframe(np_rgb):
    """
    입력: np_rgb(H,W,3, uint8)
    반환:
      boxes: [[x1,y1,x2,y2,score], ...]
      kpts:  [ [[x,y,conf],...], ... ]
    """
    global _canvas, _xbuf
    H0, W0 = np_rgb.shape[:2]

    # letterbox (캔버스 재사용)
    lb, r, px, py = letterbox_into(np_rgb, pose_in_size, PAD_VAL, _canvas)

    # NCHW 변환(버퍼 재사용)
    if pose_u8:
        # uint8 입력
        _xbuf[0,0,:,:] = lb[:,:,0]
        _xbuf[0,1,:,:] = lb[:,:,1]
        _xbuf[0,2,:,:] = lb[:,:,2]
        x = _xbuf
    else:
        # float32 입력 (정규화 off가 기본)
        _xbuf[0,0,:,:] = lb[:,:,0] / (255.0 if NORM_01 else 1.0)
        _xbuf[0,1,:,:] = lb[:,:,1] / (255.0 if NORM_01 else 1.0)
        _xbuf[0,2,:,:] = lb[:,:,2] / (255.0 if NORM_01 else 1.0)
        x = _xbuf

    # 추론
    try:
        y = pose_sess.run(pose_out_names, {pose_in_name: x})
    except rt.OnnxRuntimeException as e:
        if VERBOSE:
            print("[POSE][ERR]", e, flush=True)
        return [], []

    boxes = []; kpts_all = []

    # 일반적으로 'detections' 하나가 (N,D) 또는 (1,N,D)
    for out in y:
        arr = np.array(out)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] < 56:
            continue

        D = arr.shape[1]
        kstart, has_cls = _find_kstart(D)

        # confidence desc 정렬
        order = np.argsort(arr[:,4])[::-1]
        if N_TOP > 0:
            order = order[:N_TOP]

        for idx in order:
            row = arr[idx]
            conf = float(row[4])
            if conf < SCORE_TH:
                continue

            if has_cls:
                cls_id = int(row[5])
                if cls_id != 0:   # person only
                    continue

            # bbox 복원 (letterbox 역보정)
            x1 = (float(row[0]) - px) / r
            y1 = (float(row[1]) - py) / r
            x2 = (float(row[2]) - px) / r
            y2 = (float(row[3]) - py) / r
            x1 = max(0, min(W0-1, x1)); y1 = max(0, min(H0-1, y1))
            x2 = max(0, min(W0-1, x2)); y2 = max(0, min(H0-1, y2))
            boxes.append([x1,y1,x2,y2, conf])

            # keypoints
            K = (D - kstart) // 3
            kp = []
            base = kstart
            for k in range(K):
                u = float(row[base + 3*k + 0])
                v = float(row[base + 3*k + 1])
                s = float(row[base + 3*k + 2])
                X = (u - px) / r; Y = (v - py) / r
                X = max(0, min(W0-1, X)); Y = max(0, min(H0-1, Y))
                kp.append([X, Y, s if s >= KPT_TH else 0.0])
            kpts_all.append(kp)
        break

    return boxes, kpts_all

# ================== IPC/서버 루프 ==================
def recv_all(conn, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = conn.recv_into(view[got:], n-got)
        if r <= 0:
            raise ConnectionError("socket closed")
        got += r
    return buf

def run_server():
    try:
        os.unlink(SOCK_PATH)
    except FileNotFoundError:
        pass

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(SOCK_PATH)
    os.chmod(SOCK_PATH, 0o777)
    s.listen(2)
    print("[SRV] listening at", SOCK_PATH)

    while True:
        conn, _ = s.accept()
        print("[SRV] client connected")
        try:
            while True:
                # 헤더 28B: magic(4s), fmt(u32), W,H,stride,fid,payload (BE)
                hdr = recv_all(conn, 28)
                magic, fmt, W, H, stride, fid, payload = struct.unpack("!4sIIIIII", hdr)
                if magic != b"FRM1":
                    raise ValueError("bad magic")
                if payload == 0:
                    continue

                raw = recv_all(conn, payload)

                # 입력 파싱
                if fmt == FMT_GRAY8:
                    rgb = preprocess_gray8_to_rgb3(raw, stride, H, W)
                elif fmt == FMT_RGB24:
                    rgb = preprocess_rgb888(raw, stride, H, W)
                else:
                    rgb = np.zeros((H, W, 3), np.uint8)

                t0 = time.time()
                boxes, kpts = infer_pose_fullframe(rgb)
                t1 = time.time()

                if VERBOSE:
                    top = max((b[4] for b in boxes), default=0.0)
                    print(f"[SRV] det={len(boxes)} top={top:.2f} infer={(t1-t0)*1000:.1f}ms",
                          flush=True)

                resp = json.dumps({
                    "frame_id": int(fid),
                    "boxes": boxes,
                    "kpts":  kpts
                }, separators=(",",":")) + "\n"

                try:
                    conn.sendall(resp.encode("utf-8"))
                except BrokenPipeError:
                    break

                # FPS 제한
                if MAX_FPS > 0:
                    budget = 1.0 / MAX_FPS
                    spent  = t1 - t0
                    if spent < budget:
                        time.sleep(budget - spent)

        except Exception as e:
            print("[SRV] client error:", e, flush=True)
        finally:
            conn.close()
            print("[SRV] client closed")

if __name__ == "__main__":
    load_pose_session()
    run_server()