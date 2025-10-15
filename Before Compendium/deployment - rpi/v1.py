#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, cv2, numpy as np, argparse, time
import tensorflow as tf
import psutil, subprocess
try:
    from tf_keras.models import load_model   # works even without full tensorflow
except Exception:
    from tensorflow.keras.models import load_model
from ultralytics import YOLO
from datetime import datetime
from collections import deque
import json

# ----------------------- constants & labels -----------------------
ASL_LABELS = None  # <— loaded dynamically
SAVE_ROOT = "captures"                # where screenshots go
LOG_PATH  = os.path.join(SAVE_ROOT, "log.csv")

# Reduce OpenCV thread contention with TF/TFLite
cv2.setNumThreads(1)

# ----------------------- label helpers ----------------------------
def load_labels_file(path):
    """
    Load class labels from .json (list[str]) or .txt (one label per line).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labels file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise ValueError("JSON labels file must be a list of strings.")
        return labels
    else:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

def reconcile_labels_with_n(labels, n_out):
    """
    Force len(labels) == n_out by padding (cls_i) or truncating.
    """
    if n_out is None:
        return labels
    if len(labels) < n_out:
        base = len(labels)
        labels = labels + [f"cls_{i}" for i in range(base, n_out)]
        print(f"[WARN] Labels shorter ({base}) than model outputs ({n_out}); padded.")
    elif len(labels) > n_out:
        print(f"[WARN] Labels longer ({len(labels)}) than model outputs ({n_out}); truncating.")
        labels = labels[:n_out]
    return labels

def _softmax_if_needed(vec):
    """
    If vec doesn't sum ~1 or has values outside [0,1], apply numerically-stable softmax.
    """
    v = np.asarray(vec, dtype=np.float64)
    if np.all(v >= -1e-6) and np.all(v <= 1.0 + 1e-6):
        s = v.sum()
        if 0.98 <= s <= 1.02:
            return v.astype(np.float32)
    v -= v.max()
    e = np.exp(v)
    sm = e / (e.sum() + 1e-9)
    return sm.astype(np.float32)

# ----------------------- TFLite wrapper --------------------------
class TFLiteClassifier:
    """
    Lightweight TFLite classifier wrapper.
    Prefers tflite_runtime (fast on edge); falls back to tensorflow.lite.
    Expects NHWC input, returns float logits/probs.
    """
    def __init__(self, tflite_path, num_threads=None):
        Runtime = None
        try:
            import tflite_runtime.interpreter as tflite  # small, fast
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
        print(f"Loading TFLite model from {tflite_path} with {threads} threads …")
        interpreter = Runtime.Interpreter(model_path=tflite_path, num_threads=threads)
        interpreter.allocate_tensors()

        self.interpreter = interpreter
        self.in_details  = interpreter.get_input_details()[0]
        self.out_details = interpreter.get_output_details()[0]
        self.input_index = self.in_details['index']
        self.output_index= self.out_details['index']
        self.in_dtype    = self.in_details['dtype']
        self.out_dtype   = self.out_details['dtype']
        self.in_scale, self.in_zero = (self.in_details.get('quantization') or (0.0, 0))
        self.out_scale, self.out_zero = (self.out_details.get('quantization') or (0.0, 0))

        self.in_shape = self.in_details['shape']  # [1, H, W, C]
        _, self.in_h, self.in_w, self.in_c = self.in_shape

        # Warm-up
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=self.in_dtype)
        interpreter.set_tensor(self.input_index, dummy)
        interpreter.invoke()

    @property
    def target_hw(self):
        # Return (W, H) for cv2.resize
        return (int(self.in_w), int(self.in_h))

    def num_outputs(self):
        # Works for shapes like [1, N] or [N]
        shape = self.out_details.get("shape", None)
        return int(np.prod(shape)) if shape is not None else None

    def _quantize_input(self, x_float01):
        if np.issubdtype(self.in_dtype, np.integer):
            if self.in_scale and self.in_scale > 0:
                x = x_float01 / self.in_scale + self.in_zero
            else:
                x = x_float01 * 255.0
            x = np.clip(x, 0, 255).astype(self.in_dtype)
        else:
            x = x_float01.astype(self.in_dtype)
        return x

    def _dequantize_output(self, y):
        if np.issubdtype(y.dtype, np.integer) and self.out_scale:
            y = (y.astype(np.float32) - self.out_zero) * self.out_scale
        return y

    def predict(self, img_rgb_float01):
        # img_rgb_float01: HxWxC, float32 [0..1]
        x = self._quantize_input(img_rgb_float01)[None, ...]  # add batch
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_index)[0]
        y = self._dequantize_output(y)
        return y

# ----------------------- helper functions -------------------------
def parse_arguments():
    p = argparse.ArgumentParser(description='ASL Sign Interpreter: YOLO (.pt) + Keras/TFLite classifier with tracking')
    # Classifier options
    p.add_argument('--model_path',      type=str, default='models/model_all.h5', help='Keras classifier (.h5); ignored if --tflite is set')
    p.add_argument('--tflite',          type=str, default='', help='Path to TFLite classifier (.tflite). If set, uses TFLite.')
    p.add_argument('--tflite_threads',  type=int, default=0, help='TFLite num_threads; 0 = auto')
    p.add_argument('--confidence',      type=float, default=0.7, help='ASL classification accept threshold')
    p.add_argument('--labels',          type=str, default='models/labels_classifier.json', help='Path to labels file (.json list or .txt lines)')

    # YOLO options
    p.add_argument('--yolo_weights', type=str, default='models/best_hand.pt', help='Ultralytics YOLO .pt weights')
    p.add_argument('--yolo_conf',    type=float, default=0.30, help='YOLO hand detection conf threshold')
    p.add_argument('--yolo_imgsz',   type=int, default=192, help='YOLO inference size (smaller = faster)')
    p.add_argument('--device',       type=str, default='', help='YOLO device (e.g. "cpu", "cuda:0")')

    # Camera / Display
    p.add_argument('--camera_id', type=int, default=0)
    p.add_argument('--width',     type=int, default=640)
    p.add_argument('--height',    type=int, default=480)
    p.add_argument('--view',      action='store_true', help='Show live window')
    p.add_argument('--save',      action='store_true', help='Save annotated frames to disk')

    # Performance knobs
    p.add_argument('--det_every', type=int, default=5, help='Run YOLO every N frames (tracking in-between)')
    p.add_argument('--cls_every', type=int, default=2, help='Run classifier every M frames')
    p.add_argument('--smooth_k',  type=int, default=5, help='Smoothing window for prediction averaging')
    p.add_argument('--no_track',  action='store_true', help='Disable tracking (detect every frame)')

    return p.parse_args()

def load_and_prep_model(path):
    print(f"Loading Keras (.h5) classification model from {path} …")
    model = load_model(path, compile=False)
    in_shape = model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    h, w = in_shape[1], in_shape[2]
    chans = in_shape[3] if len(in_shape) >= 4 else 3
    if chans != 3:
        print(f"[WARN] Classifier expects {chans} channels; script assumes RGB.")
    print(f"Classifier expects input HxW: {(h, w)}")
    return model, (int(w), int(h))  # (W, H) for cv2.resize

def ensure_dirs():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp","filename","label","confidence","x1","y1","x2","y2"]
            )

def detect_hand_and_preprocess(frame_bgr, yolo_model, yolo_conf, yolo_imgsz, target_size):
    """
    Run YOLO .pt on the BGR frame, pick the best box, return:
      disp_frame (with box), preprocessed_roi (RGB/float [0..1] HxW), bbox tuple, raw_roi (BGR)
    """
    results = yolo_model.predict(
        source=frame_bgr,
        imgsz=yolo_imgsz,
        conf=yolo_conf,
        device=None,
        verbose=False
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return frame_bgr, None, None, None

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
    idx = int(np.argmax(areas))
    x1,y1,x2,y2 = map(int, xyxy[idx])

    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    disp = frame_bgr.copy()
    cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return disp, None, (x1,y1,x2,y2), None

    img = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.ascontiguousarray(img)
    return disp, img, (x1,y1,x2,y2), roi

def draw_prediction(frame, pred_vec, labels, thresh):
    pred_vec = _softmax_if_needed(pred_vec)
    idx  = int(np.argmax(pred_vec))
    conf = float(pred_vec[idx])
    lab  = labels[idx] if idx < len(labels) else "Unknown"
    if conf < thresh:
        lab_show, color = "Uncertain", (0,0,255)
    else:
        lab_show, color = f"Predicted: {lab}", (0,255,0)
    cv2.putText(frame, lab_show, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Confidence: {conf:.2f}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return lab, conf

def save_screenshot(label, confidence, bbox, frame_bgr):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder    = os.path.join(SAVE_ROOT, label)
    os.makedirs(folder, exist_ok=True)
    filename  = f"{label}_{timestamp}.jpg"
    path      = os.path.join(folder, filename)
    cv2.imwrite(path, frame_bgr)

    with open(LOG_PATH,"a",newline="") as f:
        csv.writer(f).writerow([timestamp, filename, label, f"{confidence:.4f}", *(bbox or (0,0,0,0))])

    print(f"[SAVED] screenshot → {path}  (conf {confidence:.2f})")
    
def put_text_bg(img, text, org, color, scale=1.0, thickness=2):
    # draw text with a black background box for readability
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x-4, y-h-baseline-4), (x+w+4, y+4), (0,0,0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    
def get_cpu_temp_c():
    """Best-effort CPU/SoC temperature (°C) on RPi/Linux."""
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
        for name, entries in (temps or {}).items():
            for e in entries:
                label = (e.label or name or "").lower()
                if any(k in label for k in ("cpu", "soc", "package", "core", "bcm", "gpu")):
                    if e.current and e.current > 0:
                        return float(e.current)
        for entries in (temps or {}).values():
            for e in entries:
                if e.current and e.current > 0:
                    return float(e.current)
    except Exception:
        pass
    for path in ("/sys/class/thermal/thermal_zone0/temp",
                 "/sys/class/hwmon/hwmon0/temp1_input"):
        try:
            with open(path, "r") as f:
                v = f.read().strip()
                if v:
                    t = float(v)
                    if t > 200:  # millidegrees
                        t = t / 1000.0
                    return t
        except Exception:
            pass
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True).strip()
        if "temp=" in out:
            s = out.split("temp=")[1].split("'")[0]
            return float(s)
    except Exception:
        pass
    return None

def get_system_stats():
    """Return (ram_used_MB, ram_percent, cpu_temp_C)"""
    vm = psutil.virtual_memory()
    used_mb = (vm.total - vm.available) / (1024 * 1024)
    percent = vm.percent
    temp_c = get_cpu_temp_c()
    return used_mb, percent, temp_c

# ----------------------- Template tracker (no contrib needed) -----
class TemplateTracker:
    def __init__(self):
        self.template = None
        self.bbox = None  # (x1,y1,x2,y2)
        self.ok = False
    def init(self, frame, box_xywh):
        x, y, w, h = box_xywh
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
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
    def update(self, frame, search_pad=40, min_corr=0.4):
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

# ----------------------- tracker factory --------------------------
def create_tracker():
    try:
        return cv2.legacy.TrackerMOSSE_create()
    except Exception:
        pass
    try:
        return cv2.legacy.TrackerKCF_create()
    except Exception:
        pass
    try:
        return cv2.TrackerKCF_create()
    except Exception:
        pass
    try:
        return cv2.legacy.TrackerCSRT_create()
    except Exception:
        pass
    try:
        return cv2.TrackerCSRT_create()
    except Exception:
        pass
    return TemplateTracker()

# --------------------------- main loop ----------------------------
def main():
    global ASL_LABELS
    args = parse_arguments()
    ensure_dirs()

    # ---- load classifier (prefer TFLite) ----
    is_tflite = False
    n_out = None
    if args.tflite:
        threads = args.tflite_threads if args.tflite_threads > 0 else max(1, (os.cpu_count() or 4)-1)
        clf = TFLiteClassifier(args.tflite, num_threads=threads)
        in_hw = clf.target_hw  # (W, H)
        is_tflite = True
        n_out = clf.num_outputs()
        print(f"TFLite classifier input WxH: {in_hw}, outputs: {n_out}")
    else:
        clf, in_hw = load_and_prep_model(args.model_path)  # (W, H)
        # infer outputs from model output shape (e.g., (None, N))
        out_shape = clf.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        n_out = int(out_shape[-1]) if out_shape is not None else None
        print(f"Keras classifier input WxH: {in_hw}, outputs: {n_out}")

    # ---- load and reconcile labels with model outputs ----
    try:
        loaded = load_labels_file(args.labels)
    except Exception as e:
        raise SystemExit(f"ERROR loading labels: {e}\n"
                         f"Tip: Provide --labels pointing to your .json/.txt labels file.")
    ASL_LABELS = reconcile_labels_with_n(loaded, n_out)
    print(f"Loaded {len(ASL_LABELS)} labels from {args.labels}")

    # ---- YOLO (.pt) ----
    print(f"Loading YOLO weights: {args.yolo_weights}")
    yolo = YOLO(args.yolo_weights)
    if args.device:
        try:
            yolo.to(args.device)
            print(f"YOLO device → {args.device}")
        except Exception as e:
            print(f"[WARN] Could not set YOLO device '{args.device}': {e}")

    # ---- Camera init ----
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

    # Tracking + smoothing state
    tracker = None
    has_track = False
    track_bbox = None
    frame_idx = 0
    pred_smooth = deque(maxlen=max(1, args.smooth_k))
    stable_label = "uncertain"
    stable_conf  = 0.0
    cand_label   = None
    cand_count   = 0
    REQUIRED_STREAK = 3          # how many consecutive wins to switch labels
    MIN_CONF_FOR_SWITCH = 0.55
    box_ema = None
    BOX_BETA = 0.6

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)

        now = time.perf_counter()
        fps_times.append(now)
        fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0] + 1e-9) if len(fps_times) >= 2 else 0.0

        # 1) DETECT
        need_detect = args.no_track or (frame_idx % max(1, args.det_every) == 0) or (not has_track)
        if need_detect:
            disp = frame.copy()

            # Get up to 2 biggest hand boxes
            boxes = detect_hands(yolo, disp, conf=args.yolo_conf, iou_thresh=0.45, max_det=4)

            # Choose a single working box: merge if two close, else largest
            chosen_box = None
            if len(boxes) == 2 and should_merge(boxes[0], boxes[1]):
                chosen_box = union_box(boxes[0], boxes[1])
            elif len(boxes) >= 1:
                chosen_box = boxes[0]

            # Draw chosen and hand to tracker
            yolo_bbox = None
            if chosen_box is not None:
                x1,y1,x2,y2 = chosen_box
                cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
                yolo_bbox = chosen_box

            if (not args.no_track) and (yolo_bbox is not None):
                tracker = create_tracker()
                x1,y1,x2,y2 = yolo_bbox
                w = max(1, x2 - x1); h = max(1, y2 - y1)
                has_track = tracker.init(frame, (float(x1), float(y1), float(w), float(h)))
                track_bbox = (x1, y1, x2, y2) if has_track else None

                # smooth the box a bit (EMA)
                bx = np.array([x1, y1, x2, y2], dtype=np.float32)
                box_ema = bx if box_ema is None else 0.6 * box_ema + 0.4 * bx
                sx1, sy1, sx2, sy2 = map(int, box_ema.round())
                track_bbox = (sx1, sy1, sx2, sy2)
            else:
                has_track = False if args.no_track else False
                track_bbox = yolo_bbox
        else:
            disp = frame.copy()

        # 2) TRACK
        if (not args.no_track) and has_track and tracker is not None:
            ok_tr, box = tracker.update(frame)
            if ok_tr:
                x, y, w, h = box
                x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
                track_bbox = (x1, y1, x2, y2)
                cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
                bx = np.array([x1, y1, x2, y2], dtype=np.float32)
                box_ema = bx if box_ema is None else BOX_BETA * box_ema + (1.0 - BOX_BETA) * bx
                sx1, sy1, sx2, sy2 = map(int, box_ema.round())
                track_bbox = (sx1, sy1, sx2, sy2)
            else:
                has_track = False
                track_bbox = None

        # 3) CLASSIFY
        label, conf = "uncertain", 0.0
        do_classify = (frame_idx % max(1, args.cls_every) == 0)
        active_bbox = track_bbox if track_bbox is not None else None

        if active_bbox is not None and do_classify:
            x1,y1,x2,y2 = active_bbox
            H, W = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                img = cv2.resize(roi, in_hw, interpolation=cv2.INTER_LINEAR)  # in_hw = (W,H)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = np.ascontiguousarray(img)

                raw = clf.predict(img) if is_tflite else load_and_run_keras(clf, img)
                pred = _softmax_if_needed(raw)
                pred_smooth.append(pred)
                pred_avg = np.mean(pred_smooth, axis=0) if len(pred_smooth) > 1 else pred

                idx  = int(np.argmax(pred_avg))
                conf = float(pred_avg[idx])
                lab  = ASL_LABELS[idx] if idx < len(ASL_LABELS) else "Unknown"

                # ---- Debounce: require same top-1 several times before switching ----
                if conf >= MIN_CONF_FOR_SWITCH:
                    if cand_label == lab:
                        cand_count += 1
                    else:
                        cand_label = lab
                        cand_count = 1
                    if cand_count >= REQUIRED_STREAK:
                        stable_label = lab
                        stable_conf  = conf
                else:
                    cand_count = max(0, cand_count - 1)

        # 4) HUD with STABLE label
        if active_bbox is None:
            put_text_bg(disp, "No hand detected", (10,30), (0,0,255), scale=1.0, thickness=2)
        elif (stable_label == "uncertain") or (stable_conf < args.confidence):
            put_text_bg(disp, "Hand detected", (10,30), (0,165,255), scale=1.0, thickness=2)
            put_text_bg(disp, f"Uncertain (conf {stable_conf:.2f})", (10,70), (0,165,255), scale=0.8, thickness=2)
        else:
            put_text_bg(disp, f"Predicted: {stable_label}", (10,30), (0,255,0), scale=1.0, thickness=2)
            put_text_bg(disp, f"Confidence: {stable_conf:.2f}", (10,70), (0,255,0), scale=0.8, thickness=2)

        # 5) System stats
        ram_mb, ram_percent, temp_c = get_system_stats()
        stat_lines = [f"RAM: {ram_mb:.0f} MB ({ram_percent:.1f}%)", f"FPS: {fps:.1f}"]
        if temp_c is not None:
            stat_lines.insert(1, f"CPU Temp: {temp_c:.1f} C")
        y0 = disp.shape[0] - 60
        for i, line in enumerate(stat_lines):
            put_text_bg(disp, line, (10, y0 + i * 25), (255, 255, 255), scale=0.6, thickness=1)

        # UI
        key = cv2.waitKey(1) & 0xFF
        if args.view:
            cv2.imshow("ASL Sign Interpreter", disp)
        if   key == ord('q'):
            break
        elif key == ord('s'):
            label_to_use = (stable_label if stable_conf >= args.confidence else "uncertain")
            save_screenshot(label_to_use, stable_conf, active_bbox, disp)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Interpreter closed.")

def load_and_run_keras(keras_model, img_rgb_float01):
    pred = keras_model.predict(np.expand_dims(img_rgb_float01, 0), verbose=0)[0]
    return pred
    
def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
    inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0,inter_x2-inter_x1), max(0,inter_y2-inter_y1)
    inter = iw*ih
    if inter == 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

def should_merge(b1,b2):
    # Sort left-to-right by center x to compute gap/overlap consistently
    (lx1,ly1,lx2,ly2),(rx1,ry1,rx2,ry2) = sorted([b1,b2], key=lambda b:(b[0]+b[2])/2)
    # vertical overlap ratio measured on smaller height
    ov_y = max(0, min(ly2, ry2) - max(ly1, ry1))
    h1, h2 = ly2-ly1, ry2-ry1
    v_overlap_ratio = ov_y / max(1, min(h1,h2))
    # horizontal gap normalized by smaller width
    gap = max(0, rx1 - lx2)
    w1, w2 = lx2-lx1, rx2-rx1
    gap_ratio = gap / max(1, min(w1,w2))
    # merge if vertically aligned and not far apart horizontally OR good IoU
    return (v_overlap_ratio >= 0.5 and gap_ratio <= 0.35) or (iou(b1,b2) >= 0.15)

def union_box(b1,b2):
    return (min(b1[0],b2[0]), min(b1[1],b2[1]),
            max(b1[2],b2[2]), max(b1[3],b2[3]))

def detect_hands(yolo, frame, conf, iou_thresh=0.45, max_det=4):
    """
    Run YOLO on BGR frame and return top 2 boxes (xyxy int) by area.
    """
    res = yolo.predict(source=frame, conf=conf, iou=iou_thresh,
                       max_det=max_det, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    arr = res.boxes.xyxy.cpu().numpy()
    boxes = [tuple(map(int, b)) for b in arr]
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes[:2]


if __name__ == "__main__":
    main()
