#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combined_infer.py
Run BOTH:
  1) Static pipeline: YOLO(hand).tflite -> hand crop -> classifier.tflite (A..Z, del, space)
  2) Dynamic pipeline: video clip -> dynamic.tflite (sequence model, e.g., LSA64)

Notes:
- Keeps cached HUD stats & tracking tricks from the static pipeline.
- Keeps dynamic time-dimension handling (T may be -1) from the dynamic pipeline.
- Cleaned quantization paths + stable smoothing.
- Robust to missing delegates; uses tflite_runtime if present, else TensorFlow Lite.
"""

import os, json, time, subprocess, gc, csv, argparse
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import psutil

# =========================
# ====== TFLite load ======
# =========================
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite_runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter as TFInterpreter  # type: ignore
        tflite = None
        TFLITE_BACKEND = "tensorflow"
    except Exception as e:
        tflite = None
        TFInterpreter = None
        TFLITE_BACKEND = "none"
        raise RuntimeError("No TFLite backend found. Install tflite_runtime or TensorFlow.") from e

# Silence harmless __del__ error on some tflite_runtime builds
if tflite is not None:
    try:
        if hasattr(tflite, "Delegate") and hasattr(tflite.Delegate, "__del__"):
            _orig_del = tflite.Delegate.__del__
            def _safe_del(self):
                if hasattr(self, "_library"):
                    try:
                        _orig_del(self)
                    except Exception:
                        pass
            tflite.Delegate.__del__ = _safe_del  # type: ignore[attr-defined]
    except Exception:
        pass

# =========================
# ====== ARGUMENTS ========
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Combined static+dynamic ASL inference (webcam)")
    # Detector (static)
    p.add_argument('--det_tflite',   type=str, default='models/v5n_100epoch_img416.tflite', help='Path to YOLO hand detector .tflite')
    p.add_argument('--det_threads',  type=int, default=0, help='Detector threads; 0=auto')
    p.add_argument('--yolo_conf',    type=float, default=0.30, help='YOLO conf threshold')
    p.add_argument('--yolo_iou',     type=float, default=0.45, help='YOLO NMS IoU')

    # Static classifier
    p.add_argument('--cls_tflite',   type=str, default='models/model.tflite', help='Path to static ASL classifier .tflite')
    p.add_argument('--cls_threads',  type=int, default=0, help='Classifier threads; 0=auto')
    p.add_argument('--confidence',   type=float, default=0.80, help='Confidence accept threshold')
    p.add_argument('--smooth_k',     type=int, default=6, help='Smoothing window for static predictions')
    p.add_argument('--crop_pad',     type=float, default=1.0, help='Padding scale for square crop')
    p.add_argument('--min_box_size', type=int, default=10, help='Reject tiny tracks')

    # Dynamic sequence model
    p.add_argument('--dyn_tflite',   type=str, default='models/dynamic.tflite', help='Path to dynamic (sequence) model .tflite')
    p.add_argument('--dyn_meta',     type=str, default='models/dynamic.json', help='Meta JSON for dynamic classes')
    p.add_argument('--dyn_frames',   type=int, default=8, help='Fallback T if model uses dynamic T')
    p.add_argument('--dyn_stride',   type=int, default=2, help='Infer every STRIDE frames when buffer full')
    p.add_argument('--dyn_size',     type=int, default=112, help='Frame size for dynamic model (HxW)')
    p.add_argument('--dyn_ema',      type=float, default=0.6, help='EMA for logits in dynamic branch')
    p.add_argument('--dyn_topk',     type=int, default=5, help='Show top-k for dynamic')

    # Camera / Display
    p.add_argument('--camera_id', type=int, default=0)
    p.add_argument('--width',     type=int, default=640)
    p.add_argument('--height',    type=int, default=480)
    p.add_argument('--view',      action='store_true', help='Show UI window')

    # Pipeline toggles
    p.add_argument('--run_static',  action='store_true', help='Enable static detector+classifier branch')
    p.add_argument('--run_dynamic', action='store_true', help='Enable dynamic sequence branch')

    # Performance
    p.add_argument('--det_every',        type=int, default=4, help='Detect every N frames when tracking')
    p.add_argument('--det_idle_every',   type=int, default=4, help='Detect every N frames when idle')
    p.add_argument('--cls_every',        type=int, default=1, help='Classify every M frames')
    p.add_argument('--no_track',         action='store_true')

    # Saving / logging
    p.add_argument('--save',      action='store_true', help='Save annotated frames')
    p.add_argument('--save_root', type=str, default='captures', help='Root for screenshots/log')
    p.add_argument('--log_csv',   type=str, default='log.csv', help='CSV log filename inside save_root')

    return p.parse_args()

# =========================
# ====== UTILITIES ========
# =========================
ASL_LABELS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','space'
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TOPK_SHOW_STATIC = 3
TOPK_NMS_LIMIT   = 50

def ensure_dirs(save_root: str, log_csv: str):
    os.makedirs(save_root, exist_ok=True)
    log_path = os.path.join(save_root, log_csv)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","filename","label","confidence","x1","y1","x2","y2"])
    return log_path

def save_screenshot(save_root: str, log_path: str, label: str, confidence: float, bbox: Tuple[int,int,int,int], frame_bgr):
    ts = time.strftime('%Y%m%d_%H%M%S_') + f"{int((time.time()%1)*1e6):06d}"
    folder = os.path.join(save_root, label)
    os.makedirs(folder, exist_ok=True)
    filename = f"{label}_{ts}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, frame_bgr)
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([ts, filename, label, f"{confidence:.4f}", *(bbox or (0,0,0,0))])
    print(f"[SAVED] {path}  (conf {confidence:.2f})")

def put_text_bg(img, text, org, color, scale=0.8, thickness=2):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x-4, y-h-base-4), (x+w+4, y+4), (0,0,0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# HUD stats (cached)
_last_stat_t, _last_stats = 0.0, (0.0, 0.0, None)
_cpu_inited, _last_cpu_t, _last_cpu_pct = False, 0.0, 0.0

def _get_cpu_temp_c():
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
        for name, entries in (temps or {}).items():
            for e in entries:
                label = (e.label or name or "").lower()
                if any(k in label for k in ("cpu","soc","package","core","bcm","gpu")) and (e.current and e.current>0):
                    return float(e.current)
        for entries in (temps or {}).values():
            for e in entries:
                if e.current and e.current>0: return float(e.current)
    except Exception:
        pass
    for p in ("/sys/class/thermal/thermal_zone0/temp", "/sys/class/hwmon/hwmon0/temp1_input"):
        try:
            v = float(open(p).read().strip())
            return v/1000.0 if v > 200 else v
        except Exception:
            pass
    try:
        out = subprocess.check_output(["vcgencmd","measure_temp"], text=True).strip()
        if "temp=" in out:
            return float(out.split("temp=")[1].split("'")[0])
    except Exception:
        pass
    return None

def _get_system_stats():
    vm = psutil.virtual_memory()
    used_mb = (vm.total - vm.available) / (1024*1024)
    return used_mb, vm.percent, _get_cpu_temp_c()

def _get_system_stats_cached():
    global _last_stat_t, _last_stats
    now = time.time()
    if now - _last_stat_t > 0.4:
        _last_stats = _get_system_stats()
        _last_stat_t = now
    return _last_stats

def _get_cpu_percent_cached(period=0.5, alpha=0.4):
    global _cpu_inited, _last_cpu_t, _last_cpu_pct
    now = time.time()
    if not _cpu_inited:
        psutil.cpu_percent(interval=None)
        _cpu_inited, _last_cpu_t = True, now
        return 0.0
    if now - _last_cpu_t >= period:
        new = psutil.cpu_percent(interval=None)
        _last_cpu_pct = (1.0 - alpha) * _last_cpu_pct + alpha * new
        _last_cpu_t = now
    return _last_cpu_pct

def draw_performance_hud(frame, fps):
    h, w = frame.shape[:2]
    used_mb, ram_percent, temp_c = _get_system_stats_cached()
    cpu_percent = _get_cpu_percent_cached()
    lines = [f"FPS: {fps:.1f}", f"CPU: {cpu_percent:.0f}%", f"RAM: {used_mb:.0f} MB ({ram_percent:.0f}%)"]
    if temp_c is not None:
        lines.insert(2, f"Temp: {temp_c:.1f}°C")
    y = 30
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = w - tw - 15
        color = (0,255,255)
        if "CPU" in line: color=(0,165,255)
        elif "RAM" in line: color=(255,255,0)
        cv2.rectangle(frame, (x-5, y-th-5), (x+tw+5, y+5), (0,0,0), -1)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += th + 8

# =========================
# ====== Geom helpers =====
# =========================
def expand_square(x1, y1, x2, y2, img_w, img_h, pad=1.25):
    cx = 0.5*(x1 + x2); cy = 0.5*(y1 + y2)
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    s = pad * max(w, h)
    half = s/2
    nx1 = max(0, cx-half); ny1 = max(0, cy-half)
    nx2 = min(img_w, cx+half); ny2 = min(img_h, cy+half)
    if (nx2-nx1) < 20 or (ny2-ny1) < 20:
        nx1 = max(0, x1-10); ny1 = max(0, y1-10)
        nx2 = min(img_w, x2+10); ny2 = min(img_h, y2+10)
    return int(nx1), int(ny1), int(nx2), int(ny2)

def letterbox(im, new_shape, color=(114,114,114)):
    h, w = im.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    unpad_w = int(round(w * r)); unpad_h = int(round(h * r))
    dw = new_w - unpad_w; dh = new_h - unpad_h
    dw2, dh2 = dw // 2, dh // 2
    if (h, w) != (unpad_h, unpad_w):
        im = cv2.resize(im, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, dh2, dh - dh2, dw2, dw - dw2, cv2.BORDER_CONSTANT, value=color)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return im, (r, r), (dw2, dh2)

def scale_boxes_back(xyxy_in, ratio, pad, orig_w, orig_h):
    if xyxy_in.size == 0:
        return xyxy_in
    r_w, r_h = ratio
    dw, dh = pad
    boxes = xyxy_in.copy().astype(np.float32)
    boxes[:, [0, 2]] -= dw; boxes[:, [1, 3]] -= dh
    boxes[:, [0, 2]] /= r_w; boxes[:, [1, 3]] /= r_h
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w-1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h-1)
    return boxes

# =========================
# ====== NMS / softmax ====
# =========================
def nms_xyxy(boxes, scores, iou_thres=0.45, top_k=300):
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1][:top_k]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

def softmax_np(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-9)

# =========================
# ====== Models ===========
# =========================
def _get_tflite_interpreter(model_path, num_threads=0):
    threads = num_threads if (num_threads and num_threads > 0) else max(1, (os.cpu_count() or 4) - 1)
    if tflite is not None:
        intr = tflite.Interpreter(model_path=model_path, num_threads=threads)  # type: ignore[attr-defined]
    else:
        intr = TFInterpreter(model_path=model_path, num_threads=threads)       # type: ignore
    intr.allocate_tensors()
    return intr

class TFLiteDetector:
    def __init__(self, path, threads=0, conf_thres=0.30, iou_thres=0.45):
        self.interpreter = _get_tflite_interpreter(path, threads)
        self.in_details  = self.interpreter.get_input_details()[0]
        self.input_index = self.in_details['index']
        self.in_dtype    = self.in_details['dtype']
        self.in_scale, self.in_zero = (self.in_details.get('quantization') or (0.0, 0))
        self.in_shape = self.in_details['shape']  # [1,H,W,C]
        _, self.in_h, self.in_w, self.in_c = self.in_shape
        self.out_details = self.interpreter.get_output_details()
        self.out_indices = [d['index'] for d in self.out_details]
        self.conf_thres = float(conf_thres)
        self.iou_thres  = float(iou_thres)
        # Warm-up
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=self.in_dtype)
        self.interpreter.set_tensor(self.input_index, dummy)
        self.interpreter.invoke()

    @property
    def input_hw(self): return (self.in_w, self.in_h)

    def _quantize(self, x_float01):
        if np.issubdtype(self.in_dtype, np.integer):
            if self.in_scale and self.in_scale > 0:
                x = (x_float01 / self.in_scale) + self.in_zero
                x = np.round(x)
            else:
                x = x_float01 * 255.0
            if self.in_dtype == np.uint8: x = np.clip(x, 0, 255)
            elif self.in_dtype == np.int8: x = np.clip(x, -128, 127)
            else:
                rng = np.iinfo(self.in_dtype)
                x = np.clip(x, rng.min, rng.max)
            x = x.astype(self.in_dtype)
        else:
            x = np.clip(x_float01, 0.0, 1.0).astype(self.in_dtype)
        return x

    def _gather_outputs(self):
        outs = [self.interpreter.get_tensor(i) for i in self.out_indices]
        deq = []
        for i, out in enumerate(outs):
            scale, zero = self.out_details[i].get('quantization', (0.0, 0))
            if np.issubdtype(out.dtype, np.integer) and scale and scale > 0:
                out = (out.astype(np.float32) - zero) * scale
            deq.append(out)
        deq.sort(key=lambda a: int(np.prod(a.shape)), reverse=True)
        return deq[0]

    def _parse(self, raw, in_size):
        arr = raw
        if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] < 6:
            return (np.zeros((0,4),np.float32), np.zeros((0,),np.float32), np.zeros((0,),np.int32))
        iw, ih = in_size
        cx, cy, w, h = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
        scores = arr[:,4].astype(np.float32)
        classes= arr[:,5].astype(np.int32)
        x1 = (cx - w/2.0) * iw
        y1 = (cy - h/2.0) * ih
        x2 = (cx + w/2.0) * iw
        y2 = (cy + h/2.0) * ih
        boxes = np.stack([x1,y1,x2,y2], axis=1).astype(np.float32)
        return boxes, scores, classes

    def detect(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        iw, ih = self.in_w, self.in_h
        padded_rgb01, ratio, pad = letterbox(frame_bgr, (iw, ih))
        x = self._quantize(padded_rgb01)[None, ...]
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        raw = self._gather_outputs()
        boxes_in, scores, classes = self._parse(raw, (iw, ih))
        if boxes_in.size == 0:
            return (np.zeros((0,4),np.float32), np.zeros((0,),np.float32), np.zeros((0,),np.int32))
        keep = scores >= self.conf_thres
        if not np.any(keep):
            return (np.zeros((0,4),np.float32), np.zeros((0,),np.float32), np.zeros((0,),np.int32))
        boxes_in, scores, classes = boxes_in[keep], scores[keep], classes[keep]
        if scores.size > TOPK_NMS_LIMIT:
            top = np.argpartition(scores, -TOPK_NMS_LIMIT)[-TOPK_NMS_LIMIT:]
            boxes_in, scores, classes = boxes_in[top], scores[top], classes[top]
        keep_idx = nms_xyxy(boxes_in, scores, iou_thres=self.iou_thres, top_k=TOPK_NMS_LIMIT)
        boxes_in, scores, classes = boxes_in[keep_idx], scores[keep_idx], classes[keep_idx]
        boxes_out = scale_boxes_back(boxes_in, ratio, pad, W, H)
        return boxes_out, scores, classes

class TFLiteClassifier:
    def __init__(self, path, threads=0):
        self.interpreter = _get_tflite_interpreter(path, threads)
        self.in_details  = self.interpreter.get_input_details()[0]
        self.out_details = self.interpreter.get_output_details()[0]
        self.input_index = self.in_details['index']
        self.output_index= self.out_details['index']
        self.in_dtype    = self.in_details['dtype']
        self.out_dtype   = self.out_details['dtype']
        self.in_scale, self.in_zero = (self.in_details.get('quantization') or (0.0, 0))
        self.out_scale, self.out_zero = (self.out_details.get('quantization') or (0.0, 0))
        self.in_shape = self.in_details['shape']  # [1,H,W,C]
        _, self.in_h, self.in_w, self.in_c = self.in_shape
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=self.in_dtype)
        self.interpreter.set_tensor(self.input_index, dummy)
        self.interpreter.invoke()

    @property
    def target_hw(self):  # (W,H)
        return (int(self.in_w), int(self.in_h))

    def _quantize_in(self, x_float01):
        if np.issubdtype(self.in_dtype, np.integer):
            if self.in_scale and self.in_scale > 0:
                x = (x_float01 / self.in_scale) + self.in_zero
                x = np.round(x)
            else:
                x = x_float01 * 255.0
            if self.in_dtype == np.uint8: x = np.clip(x, 0, 255)
            elif self.in_dtype == np.int8: x = np.clip(x, -128, 127)
            else:
                rng = np.iinfo(self.in_dtype); x = np.clip(x, rng.min, rng.max)
            x = x.astype(self.in_dtype)
        else:
            x = np.clip(x_float01, 0.0, 1.0).astype(self.in_dtype)
        return x

    def _dequantize_out(self, y):
        if np.issubdtype(y.dtype, np.integer) and self.out_scale and self.out_scale > 0:
            y = (y.astype(np.float32) - self.out_zero) * self.out_scale
        else:
            y = y.astype(np.float32)
        return y

    def predict(self, img_rgb_float01):
        x = self._quantize_in(img_rgb_float01)[None, ...]
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_index)[0]
        return self._dequantize_out(y)

# =========================
# === Simple tracker ======
# =========================
class TemplateTracker:
    def __init__(self):
        self.template = None
        self.bbox = None
        self.ok = False

    def init(self, frame, box_xywh):
        x, y, w, h = box_xywh
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            self.ok = False
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y1:y2, x1:x2].copy()
        self.bbox = (x1, y1, x2, y2)
        self.ok = True
        return True

    def update(self, frame, search_pad=50, min_corr=0.3):
        if not self.ok or self.template is None:
            return False, (0,0,0,0)
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = self.bbox
        sx1 = max(0, x1 - search_pad); sy1 = max(0, y1 - search_pad)
        sx2 = min(W, x2 + search_pad); sy2 = min(H, y2 + search_pad)
        search = frame[sy1:sy2, sx1:sx2]
        if search.size == 0 or self.template.size == 0:
            self.ok = False; return False, (0,0,0,0)
        search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        th, tw = self.template.shape[:2]
        if search_gray.shape[0] < th or search_gray.shape[1] < tw:
            self.ok = False; return False, (0,0,0,0)
        res = cv2.matchTemplate(search_gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < min_corr:
            self.ok = False; return False, (0,0,0,0)
        nx1, ny1 = sx1 + max_loc[0], sy1 + max_loc[1]
        nx2, ny2 = nx1 + tw, ny1 + th
        self.bbox = (nx1, ny1, nx2, ny2)
        return True, (float(nx1), float(ny1), float(tw), float(th))

def create_tracker():
    try:
        return cv2.legacy.TrackerMOSSE_create()
    except Exception:
        return TemplateTracker()

# =========================
# === Dynamic helpers =====
# =========================
def load_dyn_classes(model_path: str, meta_json: Optional[str]) -> List[str]:
    if meta_json and Path(meta_json).exists():
        return (json.load(open(meta_json)) or {}).get("classes", [])
    mp = Path(model_path)
    cands = list(mp.parent.glob("*_meta.json"))
    if cands:
        return (json.load(open(max(cands, key=lambda p: p.stat().st_mtime))) or {}).get("classes", [])
    raise FileNotFoundError("No classes found for dynamic model (provide --dyn_meta or a *_meta.json alongside).")

def get_model_io_shape(interpreter) -> List[int]:
    inp = interpreter.get_input_details()[0]
    sig = inp.get("shape_signature", None)
    shp = inp.get("shape", None)
    arr = np.array(sig if (isinstance(sig, np.ndarray) and sig.size>0) else shp).astype(int)
    return arr.tolist()

def preprocess_for_dynamic(frame_bgr, size: int):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1))  # [3,H,W]

# =========================
# ========= MAIN ==========
# =========================
def main():
    args = parse_args()
    log_path = ensure_dirs(args.save_root, args.log_csv)

    # Open camera
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_id}")

    # Init branches
    det = clf = None
    in_hw = None
    if args.run_static:
        det = TFLiteDetector(args.det_tflite, threads=args.det_threads,
                             conf_thres=args.yolo_conf, iou_thres=args.yolo_iou)
        clf = TFLiteClassifier(args.cls_tflite, threads=args.cls_threads)
        in_hw = clf.target_hw

    dyn_interp = None
    dyn_classes: List[str] = []
    dyn_in = None
    dyn_T = int(args.dyn_frames)
    dyn_size = int(args.dyn_size)
    if args.run_dynamic:
        assert Path(args.dyn_tflite).exists(), f"Dynamic model not found: {args.dyn_tflite}"
        dyn_classes = load_dyn_classes(args.dyn_tflite, args.dyn_meta)
        dyn_interp = _get_tflite_interpreter(args.dyn_tflite, max(1, (os.cpu_count() or 2)-1))
        shape = get_model_io_shape(dyn_interp)  # expect [N,C,H,W,T]
        assert len(shape) == 5, f"Dynamic model expects 5D input, got {shape}"
        N, C, Hm, Wm, Tm = [int(x) for x in shape]
        if Tm > 0: dyn_T = Tm
        if Hm > 0: dyn_size = Hm
        dyn_in = dyn_interp.get_input_details()[0]
        dyn_out= dyn_interp.get_output_details()[0]

    # Runtime state
    fps_times = deque(maxlen=30); fps_times.append(time.perf_counter())
    frame_idx = 0
    disp = None

    # Tracking/static state
    tracker = None
    has_track = False
    track_bbox = None
    box_ema = None
    BOX_BETA = 0.7
    pred_smooth = deque(maxlen=max(1, args.smooth_k))
    stable_label = "uncertain"; stable_conf = 0.0
    cand_label = None; cand_count = 0
    REQUIRED_STREAK = 2
    MIN_CONF_FOR_SWITCH = 0.40
    detection_quality_history = deque(maxlen=10)
    classification_quality_history = deque(maxlen=20)

    # Dynamic state
    frame_buf: deque = deque(maxlen=int(dyn_T))
    ema_logits: Optional[np.ndarray] = None
    last_infer_ms_dyn = 0.0
    last_dyn_label: Optional[str] = None
    last_dyn_prob: float = 0.0
    last_dyn_topk: Optional[np.ndarray] = None
    last_dyn_probs: Optional[np.ndarray] = None

    print("Webcam initialized. Press 'q' to quit, 's' to save screenshot, 'r' to reset tracking.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        disp = frame.copy()

        # ---------- STATIC BRANCH ----------
        if args.run_static and (det is not None) and (clf is not None):
            need_detect = (
                args.no_track
                or (frame_idx % max(1, args.det_every) == 0 and has_track)
                or (frame_idx % max(1, args.det_idle_every) == 0 and not has_track)
            )

            if need_detect:
                boxes, scores, _ = det.detect(frame)
                if boxes.shape[0] > 0:
                    areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                    # Filter tiny only when multiple present
                    if len(areas) > 1:
                        min_area = args.min_box_size * args.min_box_size
                        m = areas >= min_area
                        if np.any(m):
                            boxes, scores, areas = boxes[m], scores[m], areas[m]
                    j = int(np.argmax(areas))
                    detection_quality_history.append(float(scores[j]))
                    x1,y1,x2,y2 = map(int, boxes[j])
                    cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 1)
                    put_text_bg(disp, f"Det: {scores[j]:.2f}", (x1, y1-8), (0,255,0), scale=0.5)

                    if not args.no_track:
                        tracker = create_tracker()
                        w = max(1, x2-x1); h = max(1, y2-y1)
                        has_track = tracker.init(frame, (float(x1), float(y1), float(w), float(h)))
                        track_bbox = (x1,y1,x2,y2) if has_track else None
                        bx = np.array([x1,y1,x2,y2], np.float32)
                        box_ema = bx if box_ema is None else BOX_BETA*box_ema + (1.0-BOX_BETA)*bx
                        sx1,sy1,sx2,sy2 = map(int, np.round(box_ema))
                        track_bbox = (sx1,sy1,sx2,sy2)
                    else:
                        has_track = False
                        track_bbox = (x1,y1,x2,y2)
                else:
                    has_track = False
                    track_bbox = None

            if (not args.no_track) and has_track and tracker is not None:
                ok_tr, box = tracker.update(frame)
                if ok_tr:
                    x,y,w,h = box
                    x1,y1,x2,y2 = int(x), int(y), int(x+w), int(y+h)
                    if w > args.min_box_size and h > args.min_box_size:
                        track_bbox = (x1,y1,x2,y2)
                        cv2.rectangle(disp, (x1,y1), (x2,y2), (255,0,0), 2)
                        put_text_bg(disp, "Tracking", (x1, y1-8), (255,0,0), scale=0.5)
                        bx = np.array([x1,y1,x2,y2], np.float32)
                        box_ema = bx if box_ema is None else BOX_BETA*box_ema + (1.0-BOX_BETA)*bx
                        sx1,sy1,sx2,sy2 = map(int, np.round(box_ema))
                        track_bbox = (sx1,sy1,sx2,sy2)
                    else:
                        has_track = False
                        track_bbox = None
                else:
                    has_track = False
                    track_bbox = None

            # Classify crop
            if track_bbox is not None and (frame_idx % max(1, args.cls_every) == 0):
                try:
                    x1,y1,x2,y2 = track_bbox
                    x1,y1,x2,y2 = expand_square(x1,y1,x2,y2,W,H,pad=args.crop_pad)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0 and roi.shape[0] >= 10 and roi.shape[1] >= 10:
                        roi = cv2.GaussianBlur(roi, (3,3), 0)
                        img = cv2.resize(roi, in_hw, interpolation=cv2.INTER_CUBIC)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                        img = np.ascontiguousarray(np.clip(img*1.1 - 0.05, 0.0, 1.0))
                        pred_raw = clf.predict(img)
                        # Convert logits to probs if necessary
                        if (pred_raw.min() >= -1e-6) and (pred_raw.max() <= 1.0+1e-6) and (0.98 <= float(pred_raw.sum()) <= 1.02):
                            pred = pred_raw.astype(np.float32)
                        else:
                            pred = softmax_np(pred_raw.astype(np.float64)).astype(np.float32)

                        # quality + smoothing
                        top2 = np.argsort(pred)[-2:]
                        gap = float(pred[top2[-1]] - pred[top2[-2]]) if pred.size >= 2 else float(pred.max())
                        classification_quality_history.append(gap)
                        pred_smooth.append(pred)
                        if len(pred_smooth) > 1:
                            n = len(pred_smooth)
                            wts = np.linspace(1.0, 1.0 + 0.1*(n-1), n).astype(np.float32)
                            wts /= wts.sum()
                            p = np.average(list(pred_smooth), axis=0, weights=wts)
                        else:
                            p = pred
                        idx = int(np.argmax(p)); conf = float(p[idx]); lab = ASL_LABELS[idx] if idx < len(ASL_LABELS) else "Unknown"

                        # adaptive accept
                        avg_det_q = np.mean(detection_quality_history) if detection_quality_history else 0.5
                        avg_cls_q = np.mean(classification_quality_history) if classification_quality_history else 0.3
                        dyn_thr = MIN_CONF_FOR_SWITCH
                        if avg_det_q > 0.6 and avg_cls_q > 0.4: dyn_thr *= 0.9
                        elif avg_det_q < 0.4 or avg_cls_q < 0.2: dyn_thr *= 1.2

                        if conf >= dyn_thr and gap > 0.1:
                            if lab == cand_label: cand_count += 1
                            else: cand_label, cand_count = lab, 1
                            if cand_count >= REQUIRED_STREAK:
                                stable_label, stable_conf = lab, conf
                        else:
                            cand_count = max(0, cand_count - 1)

                except Exception as e:
                    # Fail-safe: keep previous stable_label/Conf
                    print(f"[Static classify] error: {e}")

            # Display status
            if track_bbox is None:
                put_text_bg(disp, "No hand detected", (10,30), (0,0,255), scale=1.0)
            elif (stable_label == "uncertain") or (stable_conf < args.confidence):
                put_text_bg(disp, "Hand detected", (10,30), (0,165,255), scale=1.0)
                put_text_bg(disp, f"Analyzing... (conf {stable_conf:.2f})", (10,65), (0,165,255), scale=0.8)
                if cand_label and cand_count > 0:
                    put_text_bg(disp, f"Candidate: {cand_label} ({cand_count}/{REQUIRED_STREAK})",
                                (10,95), (255,165,0), scale=0.7)
            else:
                put_text_bg(disp, f"Static: {stable_label}", (10,30), (0,255,0), scale=1.0)
                put_text_bg(disp, f"Conf: {stable_conf:.2f}", (10,60), (0,255,0), scale=0.8)

        # ---------- DYNAMIC BRANCH ----------
        if args.run_dynamic and (dyn_interp is not None):
            frame_buf.append(preprocess_for_dynamic(frame, args.dyn_size))
            if (len(frame_buf) == dyn_T) and ((frame_idx % max(1, args.dyn_stride)) == 0):
                try:
                    clip = np.stack(list(frame_buf), axis=0)[None, ...].astype(np.float32)  # [1,T,C,H,W]
                    clip = np.transpose(clip, (0, 2, 3, 4, 1))  # -> [1,C,H,W,T] -> but model expects [N,C,H,W,T]? Many TF-Lite conv3d expect NHWDC; here we follow your original dynamic code layout.
                    # Your original dynamic.py used set_tensor on this layout; keep consistent:
                    dyn_interp.set_tensor(dyn_in["index"], clip)
                    t0 = time.time()
                    dyn_interp.invoke()
                    last_infer_ms_dyn = (time.time() - t0) * 1000.0
                    logits = dyn_interp.get_tensor(dyn_interp.get_output_details()[0]["index"])[0]
                    if ema_logits is None: ema_logits = logits
                    else: ema_logits = args.dyn_ema * ema_logits + (1.0 - args.dyn_ema) * logits
                    probs = softmax_np(ema_logits)
                    k = int(np.argmax(probs))
                    last_dyn_label = dyn_classes[k] if k < len(dyn_classes) else f"class_{k}"
                    last_dyn_prob = float(probs[k])
                    last_dyn_topk = np.argsort(-probs)[:max(1, args.dyn_topk)]
                    last_dyn_probs = probs.copy()
                except Exception as e:
                    print(f"[Dynamic] error: {e}")

            # Draw dynamic readout (right below static, offset a bit)
            y0 = 110 if args.run_static else 30
            put_text_bg(disp, "Dynamic:", (10, y0), (0, 255, 255), scale=0.9)
            y = y0 + 28
            if last_infer_ms_dyn > 0:
                put_text_bg(disp, f"Infer: {last_infer_ms_dyn:.1f} ms", (10, y), (0, 255, 255), scale=0.7)
                y += 24
            if (last_dyn_label is not None) and (last_dyn_probs is not None):
                put_text_bg(disp, f"Top-1: {last_dyn_label} ({last_dyn_prob*100:.1f}%)", (10, y), (0, 200, 0), scale=0.8)
                y += 26
                if last_dyn_topk is not None:
                    for i, idx in enumerate(last_dyn_topk):
                        lbl = dyn_classes[int(idx)] if int(idx) < len(dyn_classes) else f"class_{int(idx)}"
                        put_text_bg(disp, f"{i+1}. {lbl}  {last_dyn_probs[int(idx)]*100:5.1f}%", (10, y), (255,255,255), scale=0.65)
                        y += 22

        # ---------- HUD & UI ----------
        # FPS calc (smoothed)
        now = time.perf_counter()
        fps_times.append(now)
        fps = (len(fps_times)-1) / (fps_times[-1]-fps_times[0]+1e-9) if len(fps_times) >= 2 else 0.0
        draw_performance_hud(disp, fps)

        # Controls / display
        if args.view:
            cv2.imshow("Combined Inference (TFLite)", disp)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Prefer static label if confident; else dynamic; else uncertain
            if args.run_static and (stable_conf >= args.confidence):
                lab = stable_label; c = stable_conf
            elif args.run_dynamic and (last_dyn_label is not None):
                lab = f"dyn_{last_dyn_label}"; c = last_dyn_prob
            else:
                lab = "uncertain"; c = 0.0
            bbox = track_bbox if (args.run_static and track_bbox is not None) else (0,0,0,0)
            save_screenshot(args.save_root, log_path, lab, c, bbox, disp)
        elif key == ord('r'):
            # Reset static state
            tracker = None; has_track = False; track_bbox = None; box_ema = None
            pred_smooth.clear(); stable_label = "uncertain"; stable_conf = 0.0
            cand_label = None; cand_count = 0
            # Reset dynamic smoothing (not the buffer)
            ema_logits = None
            print("Reset tracking/predictions")

        if args.save:
            out_dir = os.path.join("annotated_frames")
            os.makedirs(out_dir, exist_ok=True)
            ts_name = time.strftime('%Y%m%d_%H%M%S_') + f"{int((time.time()%1)*1e6):06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, ts_name), disp)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    if dyn_interp is not None: del dyn_interp
    gc.collect()

if __name__ == "__main__":
    main()
