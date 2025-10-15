#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, cv2, numpy as np, argparse, time, psutil, subprocess
from datetime import datetime
from collections import deque

# ----------------------- constants & labels -----------------------
ASL_LABELS = None
# --- HUD stats cache ---
_last_stat_t, _last_stats = 0.0, (0.0, 0.0, None)

def _get_system_stats_cached():
    """
    Returns (used_ram_mb, ram_percent, temp_c) but only refreshes
    the numbers every ~0.4s to avoid psutil overhead on the Pi.
    """
    global _last_stat_t, _last_stats
    now = time.time()
    if now - _last_stat_t > 0.4:              # refresh ~2.5 times/second
        _last_stats = _get_system_stats()     # your existing function
        _last_stat_t = now
    return _last_stats
    
_cpu_inited = False
_last_cpu_t = 0.0
_last_cpu_pct = 0.0

def _get_cpu_percent_cached(period=0.5, alpha=0.4):
    """
    Return a smoothed CPU % updated at most every `period` seconds.
    Uses psutil.cpu_percent(interval=None) (non-blocking) and an EMA.
    """
    global _cpu_inited, _last_cpu_t, _last_cpu_pct
    now = time.time()

    # Prime psutil so first real value isn't 0.0
    if not _cpu_inited:
        psutil.cpu_percent(interval=None)
        _cpu_inited = True
        _last_cpu_t = now
        return _last_cpu_pct  # 0.0 on very first call

    # Only refresh on a slow cadence
    if now - _last_cpu_t >= period:
        new = psutil.cpu_percent(interval=None)
        _last_cpu_pct = (1.0 - alpha) * _last_cpu_pct + alpha * new  # EMA smooth
        _last_cpu_t = now

    return _last_cpu_pct

TOPK = 50
SAVE_ROOT = "captures"
LOG_PATH  = os.path.join(SAVE_ROOT, "log.csv")

# Keep OpenCV from overspinning threads
cv2.setNumThreads(1)

# ====================== TFLITE BASE WRAPPER ======================
def _get_tflite_interpreter(model_path, num_threads):
    Runtime = None
    try:
        import tflite_runtime.interpreter as tflite
        Runtime = tflite
    except Exception:
        try:
            from tensorflow.lite import Interpreter as TFInterpreter
            class _TFWrap:
                Interpreter = TFInterpreter
            Runtime = _TFWrap
        except Exception as e:
            raise RuntimeError("No TFLite interpreter available. Install tflite_runtime or TensorFlow.") from e

    threads = num_threads if (num_threads and num_threads > 0) else max(1, (os.cpu_count() or 4) - 1)
    print(f"Loading TFLite model from {model_path} with {threads} threads …")
    interpreter = Runtime.Interpreter(model_path=model_path, num_threads=threads)
    interpreter.allocate_tensors()
    return interpreter

# ====================== CLASSIFIER (TFLite) ======================
class TFLiteClassifier:
    """
    TFLite classifier: NHWC RGB float/quantized input → logits/probs vector output.
    """
    def __init__(self, tflite_path, num_threads=0):
        self.interpreter = _get_tflite_interpreter(tflite_path, num_threads)
        self.in_details  = self.interpreter.get_input_details()[0]
        self.out_details = self.interpreter.get_output_details()[0]
        self.input_index = self.in_details['index']
        self.output_index= self.out_details['index']
        self.in_dtype    = self.in_details['dtype']
        self.out_dtype   = self.out_details['dtype']
        self.in_scale, self.in_zero = (self.in_details.get('quantization') or (0.0, 0))
        self.out_scale, self.out_zero = (self.out_details.get('quantization') or (0.0, 0))
        self.in_shape = self.in_details['shape']  # [1, H, W, C]
        _, self.in_h, self.in_w, self.in_c = self.in_shape

        print(f"Classifier input shape: {self.in_shape}, dtype: {self.in_dtype}")
        print(f"Classifier quantization - input scale: {self.in_scale}, zero: {self.in_zero}")
        print(f"Classifier quantization - output scale: {self.out_scale}, zero: {self.out_zero}")

        # Warm-up
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=self.in_dtype)
        self.interpreter.set_tensor(self.input_index, dummy)
        self.interpreter.invoke()

    @property
    def target_hw(self):
        return (int(self.in_w), int(self.in_h))  # (W, H) for cv2.resize

    def _quantize_input(self, x_float01):
        """Improved quantization handling"""
        if np.issubdtype(self.in_dtype, np.integer):
            if self.in_scale and self.in_scale > 0:
                # Proper quantization formula
                x = (x_float01 / self.in_scale) + self.in_zero
                x = np.round(x)
            else:
                # Fallback: assume 0-255 range
                x = x_float01 * 255.0
            
            # Clip to proper range based on dtype
            if self.in_dtype == np.uint8:
                x = np.clip(x, 0, 255)
            elif self.in_dtype == np.int8:
                x = np.clip(x, -128, 127)
            else:
                x = np.clip(x, np.iinfo(self.in_dtype).min, np.iinfo(self.in_dtype).max)
            
            x = x.astype(self.in_dtype)
        else:
            # Float input - ensure proper range
            x = np.clip(x_float01, 0.0, 1.0).astype(self.in_dtype)
        
        return x

    def _dequantize_output(self, y):
        """Improved dequantization handling"""
        if np.issubdtype(y.dtype, np.integer) and self.out_scale and self.out_scale > 0:
            y = (y.astype(np.float32) - self.out_zero) * self.out_scale
        elif np.issubdtype(y.dtype, np.integer):
            # If no scale info, assume it's already in reasonable range
            y = y.astype(np.float32)
        return y

    def predict(self, img_rgb_float01):
        x = self._quantize_input(img_rgb_float01)[None, ...]  # [1,H,W,C]
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_index)[0]
        y = self._dequantize_output(y)
        return y

# ====================== DETECTOR (TFLite YOLO) ======================
def _nms_xyxy(boxes, scores, iou_thres=0.45, top_k=300):
    """Classic NMS on xyxy boxes."""
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

class TFLiteDetector:
    """
    TFLite YOLOv5-like detector with improved preprocessing.
    """
    def __init__(self, tflite_path, num_threads=0, conf_thres=0.25, iou_thres=0.45):
        self.interpreter = _get_tflite_interpreter(tflite_path, num_threads)
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

        print(f"Detector input shape: {self.in_shape}, dtype: {self.in_dtype}")
        print(f"Detector quantization - scale: {self.in_scale}, zero: {self.in_zero}")

        # Warm-up
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=self.in_dtype)
        self.interpreter.set_tensor(self.input_index, dummy)
        self.interpreter.invoke()

    @property
    def input_hw(self):
        return (self.in_w, self.in_h)

    def _quantize_input(self, x_float01):
        """Improved quantization for detector"""
        if np.issubdtype(self.in_dtype, np.integer):
            if self.in_scale and self.in_scale > 0:
                x = (x_float01 / self.in_scale) + self.in_zero
                x = np.round(x)
            else:
                x = x_float01 * 255.0
            
            if self.in_dtype == np.uint8:
                x = np.clip(x, 0, 255)
            elif self.in_dtype == np.int8:
                x = np.clip(x, -128, 127)
            else:
                x = np.clip(x, np.iinfo(self.in_dtype).min, np.iinfo(self.in_dtype).max)
            
            x = x.astype(self.in_dtype)
        else:
            x = np.clip(x_float01, 0.0, 1.0).astype(self.in_dtype)
        
        return x

    def _gather_outputs(self):
        outs = [self.interpreter.get_tensor(i) for i in self.out_indices]
        # Dequantize outputs if needed
        dequantized_outs = []
        for i, out in enumerate(outs):
            out_detail = self.out_details[i]
            scale, zero = out_detail.get('quantization', (0.0, 0))
            if np.issubdtype(out.dtype, np.integer) and scale and scale > 0:
                out = (out.astype(np.float32) - zero) * scale
            dequantized_outs.append(out)
        
        # pick the largest tensor as the detection head
        dequantized_outs.sort(key=lambda a: int(np.prod(a.shape)), reverse=True)
        return dequantized_outs[0]

    def _parse_detections(self, raw, in_size):
        arr = raw
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] < 6:
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,),   np.float32),
                    np.zeros((0,),   np.int32))
        iw, ih = in_size

        # ASSUME normalized cx,cy,w,h (Ultralytics TFLite default)
        cx, cy, w, h = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
        scores = arr[:,4].astype(np.float32)
        classes= arr[:,5].astype(np.int32)

        x1 = (cx - w/2.0) * iw
        y1 = (cy - h/2.0) * ih
        x2 = (cx + w/2.0) * iw
        y2 = (cy + h/2.0) * ih

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        return boxes, scores, classes

    def detect(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        iw, ih = self.in_w, self.in_h

        # Improved letterbox preprocessing
        padded_rgb01, ratio, pad = letterbox(frame_bgr, (iw, ih))
        x = self._quantize_input(padded_rgb01)[None, ...]

        # run tflite
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        raw = self._gather_outputs()

        # parse in model/input space
        boxes_in, scores, classes = self._parse_detections(raw, (iw, ih))
        if boxes_in.size == 0:
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,),   np.float32),
                    np.zeros((0,),   np.int32))

        # conf + nms
        keep = scores >= self.conf_thres
        if not np.any(keep):
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,),   np.float32),
                    np.zeros((0,),   np.int32))
        boxes_in, scores, classes = boxes_in[keep], scores[keep], classes[keep]
        if scores.size > TOPK:
            top = np.argpartition(scores, -TOPK)[-TOPK:]
            boxes_in, scores, classes = boxes_in[top], scores[top], classes[top]
        keep_idx = _nms_xyxy(boxes_in, scores, iou_thres=self.iou_thres, top_k=TOPK)
        boxes_in, scores, classes = boxes_in[keep_idx], scores[keep_idx], classes[keep_idx]

        # Scale back to original image
        boxes_out = scale_boxes_back(boxes_in, ratio, pad, W, H)

        return boxes_out, scores, classes


# ====================== UTIL / UI HELPERS ======================
def load_labels_file(path):
    """
    Load class labels from .json (list of strings) or .txt (one label per line).
    Returns a list[str].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labels file not found: {path}")

    # Detect by extension
    ext = os.path.splitext(path)[1].lower()
    labels = []
    try:
        if ext == ".json":
            import json
            with open(path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
                raise ValueError("JSON labels file must be a list of strings.")
        else:
            # Treat as plain text: one label per line
            with open(path, "r", encoding="utf-8") as f:
                labels = [ln.strip() for ln in f.readlines() if ln.strip()]
        return labels
    except Exception as e:
        raise RuntimeError(f"Failed to parse labels from {path}: {e}")

def reconcile_labels_with_output(labels, clf_out_details):
    """
    Ensure len(labels) matches classifier output size.
    If mismatch, pad with 'cls_i' or truncate and print a warning.
    Returns the reconciled label list and the inferred n_classes.
    """
    # Try to infer number of classes from output tensor shape
    out_shape = clf_out_details.get('shape', None)
    # Common shapes: [1, N] or [N]; fallback to tensor size if available
    if out_shape is not None:
        n_out = int(np.prod(out_shape))  # safe for [1,N] or [N]
    else:
        # last resort: try reading one inference to get vector length
        n_out = None

    if n_out is None:
        n_out = len(labels)  # best effort

    if len(labels) < n_out:
        # pad
        orig = len(labels)
        labels = labels + [f"cls_{i}" for i in range(orig, n_out)]
        print(f"[WARN] Label list shorter than model output ({orig} < {n_out}). "
              f"Padded with generic names up to {n_out}.")
    elif len(labels) > n_out:
        # truncate
        print(f"[WARN] Label list longer than model output ({len(labels)} > {n_out}). "
              f"Truncating labels to {n_out}.")
        labels = labels[:n_out]

    return labels, n_out



def parse_arguments():
    p = argparse.ArgumentParser(description='ASL Sign Interpreter (All-TFLite): YOLO(hand).tflite + classifier.tflite + tracking')
    p.add_argument('--det_tflite',   type=str, default='models/v5n_100epoch_img416.tflite', help='Path to YOLO hand detector (.tflite)')
    p.add_argument('--det_threads',  type=int, default=0, help='Detector TFLite threads; 0=auto')
    p.add_argument('--yolo_conf',    type=float, default=0.30, help='Detection confidence threshold')
    p.add_argument('--yolo_iou',     type=float, default=0.45, help='NMS IoU threshold')
    p.add_argument('--yolo_imgsz',   type=int,  default=512,   help='(unused: TFLite infers from model)')

    # Use your new classifier tflite by default
    p.add_argument('--cls_tflite',   type=str, default='models/model_all.tflite', help='Path to ASL classifier (.tflite)')
    p.add_argument('--cls_threads',  type=int, default=0, help='Classifier TFLite threads; 0=auto')
    p.add_argument('--confidence',   type=float, default=0.8, help='ASL classification accept threshold')

    # NEW: labels path (json or txt)
    p.add_argument('--labels',       type=str, default='models/labels_classifier.json',
                   help='Path to labels file (.json list or .txt lines)')

    # Camera / Display
    p.add_argument('--camera_id', type=int, default=0)
    p.add_argument('--width',     type=int, default=640)
    p.add_argument('--height',    type=int, default=480)
    p.add_argument('--view',      action='store_true', help='Show live window')
    p.add_argument('--save',      action='store_true', help='Save annotated frames to disk')

    # Performance knobs
    p.add_argument('--det_every', type=int, default=4, help='Run detector every N frames (track in-between)')
    p.add_argument('--det_idle_every', type=int, default=4, help='Run detector every N frames when not tracking (idle)')
    p.add_argument('--cls_every', type=int, default=1, help='Run classifier every M frames')
    p.add_argument('--smooth_k',  type=int, default=6, help='Smoothing window for prediction averaging')
    p.add_argument('--no_track',  action='store_true', help='Disable tracking (detect every frame)')

    # Preprocessing options
    p.add_argument('--crop_pad',  type=float, default=1, help='Padding factor for hand crop')
    p.add_argument('--min_box_size', type=int, default=10, help='Minimum bounding box size in pixels')

    return p.parse_args()


def ensure_dirs():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","filename","label","confidence","x1","y1","x2","y2"])

def _get_cpu_temp_c():
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
        for name, entries in (temps or {}).items():
            for e in entries:
                label = (e.label or name or "").lower()
                if any(k in label for k in ("cpu","soc","package","core","bcm","gpu")) and (e.current and e.current > 0):
                    return float(e.current)
        for entries in (temps or {}).values():
            for e in entries:
                if e.current and e.current > 0:
                    return float(e.current)
    except Exception:
        pass
    for p in ("/sys/class/thermal/thermal_zone0/temp", "/sys/class/hwmon/hwmon0/temp1_input"):
        try:
            with open(p, "r") as f:
                v = f.read().strip()
                if v:
                    t = float(v)
                    return t/1000.0 if t > 200 else t
        except Exception:
            pass
    try:
        out = subprocess.check_output(["vcgencmd","measure_temp"], text=True).strip()
        if "temp=" in out:
            return float(out.split("temp=")[1].split("'")[0])
    except Exception:
        pass
    return None
    
def draw_performance_hud(frame, fps):
    """
    Draws FPS, CPU %, CPU temperature, and RAM usage in top-right corner.
    Uses cached system stats to keep FPS high.
    """
    h, w = frame.shape[:2]

    # --- Collect stats (cheap) ---
    used_mb, ram_percent, temp_c = _get_system_stats_cached()
    cpu_percent = _get_cpu_percent_cached()

    # --- Text lines ---
    lines = [
        f"FPS: {fps:.1f}",
        f"CPU: {cpu_percent:.0f}%",
        f"RAM: {used_mb:.0f} MB ({ram_percent:.0f}%)"
    ]
    if temp_c is not None:
        lines.insert(2, f"Temp: {temp_c:.1f}°C")

    # --- Draw top-right ---
    y = 30
    for line in lines:
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = w - text_w - 15  # align to right edge with 15px margin

        color = (0, 255, 255)   # default yellow
        if "CPU" in line: color = (0, 165, 255)     # orange
        elif "RAM" in line: color = (255, 255, 0)   # cyan

        cv2.rectangle(frame, (x-5, y-text_h-5), (x+text_w+5, y+5), (0,0,0), -1)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += text_h + 8

def _get_system_stats():
    vm = psutil.virtual_memory()
    used_mb = (vm.total - vm.available) / (1024*1024)
    return used_mb, vm.percent, _get_cpu_temp_c()
    
def put_text_bg(img, text, org, color, scale=1.0, thickness=2):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x-4, y-h-baseline-4), (x+w+4, y+4), (0,0,0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def expand_square(x1, y1, x2, y2, img_w, img_h, pad=1.25):
    """Improved square cropping with better bounds checking"""
    cx = 0.5*(x1 + x2)
    cy = 0.5*(y1 + y2)
    w  = max(1, x2 - x1)
    h  = max(1, y2 - y1)
    s  = pad * max(w, h)
    
    # Calculate new bounds
    half_s = s / 2
    nx1 = max(0, cx - half_s)
    ny1 = max(0, cy - half_s)
    nx2 = min(img_w, cx + half_s)
    ny2 = min(img_h, cy + half_s)
    
    # Ensure we have a reasonable crop size
    if (nx2 - nx1) < 20 or (ny2 - ny1) < 20:
        # Fallback to original bbox with minimal padding
        nx1 = max(0, x1 - 10)
        ny1 = max(0, y1 - 10)
        nx2 = min(img_w, x2 + 10)
        ny2 = min(img_h, y2 + 10)
    
    return int(nx1), int(ny1), int(nx2), int(ny2)
    
def letterbox(im, new_shape, color=(114,114,114)):
    """
    Improved letterbox with better handling
    """
    h, w = im.shape[:2]
    new_w, new_h = new_shape
    
    # Calculate scaling ratio
    r = min(new_w / w, new_h / h)
    
    # Calculate new unpadded dimensions
    unpad_w = int(round(w * r))
    unpad_h = int(round(h * r))
    
    # Calculate padding
    dw = new_w - unpad_w
    dh = new_h - unpad_h
    
    # Distribute padding evenly
    dw2 = dw // 2
    dh2 = dh // 2
    
    # Resize image
    if (h, w) != (unpad_h, unpad_w):
        im = cv2.resize(im, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    
    # Add border
    top, bottom = dh2, dh - dh2
    left, right = dw2, dw - dw2
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # Convert to RGB float32 [0,1]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    return im, (r, r), (dw2, dh2)

def scale_boxes_back(xyxy_in, ratio, pad, orig_w, orig_h):
    """Improved box scaling with better precision"""
    if xyxy_in.size == 0:
        return xyxy_in
    
    r_w, r_h = ratio
    dw, dh = pad
    boxes = xyxy_in.copy().astype(np.float32)
    
    # Remove padding
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    
    # Scale back
    boxes[:, [0, 2]] /= r_w
    boxes[:, [1, 3]] /= r_h
    
    # Clip to image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)
    
    return boxes

# ====================== SIMPLE TEMPLATE TRACKER ======================
class TemplateTracker:
    def __init__(self):
        self.template = None
        self.bbox = None  # (x1,y1,x2,y2)
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

    def update(self, frame, search_pad=50, min_corr=0.3):  # Increased search area and lowered threshold
        if not self.ok or self.template is None:
            return False, (0,0,0,0)
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = self.bbox
        sx1 = max(0, x1 - search_pad)
        sy1 = max(0, y1 - search_pad)
        sx2 = min(W, x2 + search_pad)
        sy2 = min(H, y2 + search_pad)
        search = frame[sy1:sy2, sx1:sx2]
        if search.size == 0 or self.template.size == 0:
            self.ok = False
            return False, (0,0,0,0)
        search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        th, tw = self.template.shape[:2]
        if search_gray.shape[0] < th or search_gray.shape[1] < tw:
            self.ok = False
            return False, (0,0,0,0)
        res = cv2.matchTemplate(search_gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < min_corr:
            self.ok = False
            return False, (0,0,0,0)
        nx1, ny1 = sx1 + max_loc[0], sy1 + max_loc[1]
        nx2, ny2 = nx1 + tw, ny1 + th
        self.bbox = (nx1, ny1, nx2, ny2)
        return True, (float(nx1), float(ny1), float(tw), float(th))

def create_tracker():
    try:
        return cv2.legacy.TrackerMOSSE_create()
    except Exception:
        return TemplateTracker()

def _softmax_if_needed(vec):
    """Improved softmax with better numerical stability"""
    v = vec.astype(np.float64)  # Use higher precision
    
    # Check if already probabilities
    if np.all(v >= -1e-6) and np.all(v <= 1.0 + 1e-6):
        s = np.sum(v)
        if 0.98 <= s <= 1.02:  # Already probabilities
            return v.astype(np.float32)
    
    # Apply softmax with numerical stability
    v_max = np.max(v)
    v = v - v_max  # Subtract max for numerical stability
    exp_v = np.exp(v)
    softmax = exp_v / (np.sum(exp_v) + 1e-9)
    
    return softmax.astype(np.float32)

# ====================== MAIN LOOP ======================
def main():
    args = parse_arguments()
    ensure_dirs()

    # Load TFLite models
    det = TFLiteDetector(args.det_tflite, num_threads=args.det_threads,
                         conf_thres=args.yolo_conf, iou_thres=args.yolo_iou)
    clf = TFLiteClassifier(args.cls_tflite, num_threads=args.cls_threads)
    in_hw = clf.target_hw
    print(f"Detector input WxH: {det.input_hw}")
    print(f"Classifier input WxH: {in_hw}")

    # === Load labels dynamically ===
    try:
        labels_file = args.labels
        loaded_labels = load_labels_file(labels_file)
        # Make sure label count matches the classifier output size
        reconciled, n_classes = reconcile_labels_with_output(loaded_labels, clf.out_details)
        # Set global used by the rest of the pipeline
        global ASL_LABELS
        ASL_LABELS = reconciled
        print(f"Loaded {len(ASL_LABELS)} labels from {labels_file} (model outputs {n_classes}).")
    except Exception as e:
        raise SystemExit(f"ERROR loading labels: {e}\n"
                         f"Tip: Provide --labels pointing to your labels .json/.txt file.")


    # Camera
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam initialized. Press 'q' to quit, 's' to save screenshot.")
    fps_times = deque(maxlen=30)
    start_t = time.perf_counter()
    fps_times.append(start_t)

    tracker = None
    has_track = False
    track_bbox = None
    frame_idx = 0
    pred_smooth = deque(maxlen=max(1, args.smooth_k))
    stable_label = "uncertain"
    stable_conf  = 0.0
    cand_label   = None
    cand_count   = 0
    REQUIRED_STREAK = 2  # Reduced from 1 to 2 for better stability
    MIN_CONF_FOR_SWITCH = 0.40  # Lowered threshold
    box_ema = None
    BOX_BETA = 0.7  # Slightly more smoothing for box positions

    # Quality metrics tracking
    detection_quality_history = deque(maxlen=10)
    classification_quality_history = deque(maxlen=20)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: Failed to read frame.")
            break
        frame = cv2.flip(frame, 1)

        # FPS calc
        now = time.perf_counter()
        fps = (len(fps_times)-1) / (fps_times[-1]-fps_times[0]+1e-9) if len(fps_times) >= 2 else 0.0

        # 1) DETECT (if schedule or no track)
        need_detect = (
            args.no_track
            or (frame_idx % max(1, args.det_every) == 0 and has_track)
            or (frame_idx % max(1, args.det_idle_every) == 0 and not has_track)
        )
        detection_score = 0.0
        
        if need_detect:
            boxes, scores, classes = det.detect(frame)
            if boxes.shape[0] > 0:
                # Use original logic - choose largest area, not highest score
                areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                
                # Optional: Filter very small boxes only if there are multiple detections
                if len(areas) > 1:
                    min_area = args.min_box_size * args.min_box_size
                    size_filter = areas >= min_area
                    if np.any(size_filter):
                        boxes = boxes[size_filter]
                        scores = scores[size_filter] 
                        classes = classes[size_filter]
                        areas = areas[size_filter]
                
                    # Choose largest detection
                    j = int(np.argmax(areas))
                    detection_score = float(scores[j])
                    detection_quality_history.append(detection_score)
                    
                    x1,y1,x2,y2 = map(int, boxes[j])
                    disp = frame
                    
                    # Always use green for detected boxes to avoid confusion
                    cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 1)
                    put_text_bg(disp, f"Det: {detection_score:.2f}", (x1, y1-10), (0,255,0), scale=0.5)
                    
                    if not args.no_track:
                        tracker = create_tracker()
                        w = max(1, x2 - x1)
                        h = max(1, y2 - y1)
                        has_track = tracker.init(frame, (float(x1), float(y1), float(w), float(h)))
                        track_bbox = (x1, y1, x2, y2) if has_track else None
                        
                        # Smooth box positions
                        bx = np.array([x1, y1, x2, y2], dtype=np.float32)
                        box_ema = bx if box_ema is None else BOX_BETA*box_ema + (1.0-BOX_BETA)*bx
                        sx1, sy1, sx2, sy2 = map(int, np.round(box_ema))
                        track_bbox = (sx1, sy1, sx2, sy2)
                    else:
                        has_track = False
                        track_bbox = (x1, y1, x2, y2)
                else:
                    if boxes.shape[0] > 0:
                        j = int(np.argmax(areas))
                        detection_score = float(scores[j])
                        detection_quality_history.append(detection_score)
                        x1, y1, x2, y2 = map(int, boxes[j])

                        disp = frame  # draw in place (cheaper)
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        put_text_bg(disp, f"Det: {detection_score:.2f}", (x1, y1-10), (255,0,0), scale=0.5)

                        if not args.no_track:
                            tracker = create_tracker()
                            w = max(1, x2 - x1); h = max(1, y2 - y1)
                            has_track = tracker.init(frame, (float(x1), float(y1), float(w), float(h)))
                            track_bbox = (x1, y1, x2, y2) if has_track else None

                            # optional smoothing same as other branch
                            bx = np.array([x1, y1, x2, y2], dtype=np.float32)
                            box_ema = bx if box_ema is None else 0.7*box_ema + 0.3*bx
                            sx1, sy1, sx2, sy2 = map(int, np.round(box_ema))
                            track_bbox = (sx1, sy1, sx2, sy2)
                        else:
                            has_track = False
                            track_bbox = (x1, y1, x2, y2)
            else:
                disp = frame.copy()
                has_track = False
                track_bbox = None
        else:
            disp = frame.copy()

        # 2) TRACK
        if (not args.no_track) and has_track and tracker is not None:
            ok_tr, box = tracker.update(frame)
            if ok_tr:
                x,y,w,h = box
                x1,y1,x2,y2 = int(x), int(y), int(x+w), int(y+h)
                
                # Validate tracking result
                if w > args.min_box_size and h > args.min_box_size:
                    track_bbox = (x1, y1, x2, y2)
                    cv2.rectangle(disp, (x1,y1), (x2,y2), (255,0,0), 2)
                    put_text_bg(disp, "Tracking", (x1, y1-10), (255,0,0), scale=0.5)
                    
                    # Smooth box positions
                    bx = np.array([x1, y1, x2, y2], dtype=np.float32)
                    box_ema = bx if box_ema is None else BOX_BETA*box_ema + (1.0-BOX_BETA)*bx
                    sx1, sy1, sx2, sy2 = map(int, np.round(box_ema))
                    track_bbox = (sx1, sy1, sx2, sy2)
                else:
                    has_track = False
                    track_bbox = None
            else:
                has_track = False
                track_bbox = None

        # 3) CLASSIFY (per schedule)
        label, conf = "uncertain", 0.0
        do_classify = (frame_idx % max(1, args.cls_every) == 0)
        active_bbox = track_bbox if track_bbox is not None else None

        if active_bbox is not None and do_classify:
            H, W = frame.shape[:2]
            x1, y1, x2, y2 = active_bbox

            # Improved cropping with better validation
            try:
                # Make square crop with padding
                x1, y1, x2, y2 = expand_square(x1, y1, x2, y2, W, H, pad=args.crop_pad)
                roi = frame[y1:y2, x1:x2]

                if roi.size > 0 and roi.shape[0] >= 10 and roi.shape[1] >= 10:
                    # Enhanced preprocessing for classifier
                    # Apply slight blur to reduce noise
                    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
                    
                    # Resize with high-quality interpolation
                    img = cv2.resize(roi_blur, in_hw, interpolation=cv2.INTER_CUBIC)
                    
                    # Convert to RGB and normalize
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    
                    # Ensure proper memory layout
                    img = np.ascontiguousarray(img)
                    
                    # Add subtle contrast enhancement
                    img = np.clip(img * 1.1 - 0.05, 0.0, 1.0)

                    # Predict and process
                    pred_raw = clf.predict(img)
                    pred = _softmax_if_needed(pred_raw)
                    
                    # Quality check on raw predictions
                    pred_quality = np.max(pred) - np.partition(pred, -2)[-2]  # Gap between top 2
                    classification_quality_history.append(pred_quality)

                    # Smoothing with quality weighting
                    pred_smooth.append(pred)
                    if len(pred_smooth) > 1:
                        n = len(pred_smooth)
                        weights = np.linspace(1.0, 1.0 + 0.1*(n-1), n, dtype=np.float32)  # older→smaller
                        weights /= weights.sum()
                        pred_weighted = np.average(list(pred_smooth), axis=0, weights=weights)
                    else:
                        pred_weighted = pred

                    # Top-1 prediction
                    idx  = int(np.argmax(pred_weighted))
                    conf = float(pred_weighted[idx])
                    lab  = ASL_LABELS[idx] if idx < len(ASL_LABELS) else "Unknown"

                    top2_idx = np.argsort(pred_weighted)[-2:][::-1]
                    top1_conf, top2_conf = pred_weighted[top2_idx[0]], pred_weighted[top2_idx[1]]
                    confidence_gap = top1_conf - top2_conf

                    # Adaptive thresholding based on recent performance
                    avg_det_quality = np.mean(detection_quality_history) if detection_quality_history else 0.5
                    avg_cls_quality = np.mean(classification_quality_history) if classification_quality_history else 0.3
                    
                    # Adjust thresholds based on quality
                    dynamic_threshold = MIN_CONF_FOR_SWITCH
                    if avg_det_quality > 0.6 and avg_cls_quality > 0.4:
                        dynamic_threshold *= 0.9  # Lower threshold for high-quality detections
                    elif avg_det_quality < 0.4 or avg_cls_quality < 0.2:
                        dynamic_threshold *= 1.2  # Higher threshold for low-quality detections

                    # Update stable label with improved logic
                    if conf >= dynamic_threshold and confidence_gap > 0.1:  # Require clear winner
                        if lab == cand_label:
                            cand_count += 1
                        else:
                            cand_label = lab
                            cand_count = 1
                        
                        if cand_count >= REQUIRED_STREAK:
                            stable_label = lab
                            stable_conf  = conf
                    else:
                        cand_count = max(0, cand_count - 1)
                        
            except Exception as e:
                print(f"Classification error: {e}")
                # Continue with previous stable prediction

        # 4) Enhanced status display
        if active_bbox is None:
            put_text_bg(disp, "No hand detected", (10,30), (0,0,255), scale=1.0, thickness=2)
        elif (stable_label == "uncertain") or (stable_conf < args.confidence):
            put_text_bg(disp, "Hand detected", (10,30), (0,165,255), scale=1.0, thickness=2)
            put_text_bg(disp, f"Analyzing... (conf {stable_conf:.2f})", (10,70), (0,165,255), scale=0.8, thickness=2)
            
            # Show current candidate if available
            if cand_label and cand_count > 0:
                put_text_bg(disp, f"Candidate: {cand_label} ({cand_count}/{REQUIRED_STREAK})", 
                           (10,110), (255,165,0), scale=0.7, thickness=2)
        else:
            put_text_bg(disp, f"Sign: {stable_label}", (10,30), (0,255,0), scale=1.2, thickness=2)
            put_text_bg(disp, f"Confidence: {stable_conf:.2f}", (10,70), (0,255,0), scale=0.8, thickness=2)

        # 5) Enhanced system stats with quality metrics
        draw_performance_hud(disp, fps)

        # UI
        if args.view:
            cv2.imshow("ASL Sign Interpreter (TFLite)", disp)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            lbl_to_use = stable_label if stable_conf >= args.confidence else "uncertain"
            save_screenshot(lbl_to_use, stable_conf, active_bbox, disp)
        elif key == ord('r'):  # Reset tracking and stable predictions
            tracker = None
            has_track = False
            track_bbox = None
            box_ema = None
            pred_smooth.clear()
            stable_label = "uncertain"
            stable_conf = 0.0
            cand_label = None
            cand_count = 0
            print("Reset tracking and predictions")

        if args.save:
            out_dir = os.path.join("annotated_frames")
            os.makedirs(out_dir, exist_ok=True)
            ts_name = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + ".jpg"
            cv2.imwrite(os.path.join(out_dir, ts_name), disp)


        now = time.perf_counter()
        fps_times.append(now)
        if len(fps_times) >= 2:
            fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0] + 1e-9)
        else:
            fps = 0.0
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Interpreter closed.")

def save_screenshot(label, confidence, bbox, frame_bgr):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder = os.path.join(SAVE_ROOT, label)
    os.makedirs(folder, exist_ok=True)
    filename = f"{label}_{timestamp}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, frame_bgr)
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([timestamp, filename, label, f"{confidence:.4f}", *(bbox or (0,0,0,0))])
    print(f"[SAVED] screenshot → {path}  (conf {confidence:.2f})")

if __name__ == "__main__":
    main()
