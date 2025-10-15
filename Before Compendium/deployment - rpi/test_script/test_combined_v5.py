#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate ASL classifier with TFLite YOLOv5-style detector.
Merged logic from asl_interpreter.py for robust TFLite handling, preprocessing,
and label reconciliation; plus two-hand merge similar to v1-style logic.
"""

import os, sys, cv2, argparse, json, time, csv
from time import perf_counter
from pathlib import Path
from collections import Counter
from datetime import datetime
import numpy as np
from tqdm import tqdm

# deps for metrics & excel
try:
    import psutil
except ImportError as e:
    raise SystemExit("psutil is required. Install: pip install psutil") from e

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install: pip install pandas") from e

# matplotlib (optional for CM PNG)
_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

TOPK = 300  # NMS cap

# ====================== TFLITE BASE ======================
def _get_tflite_interpreter(model_path, num_threads):
    Runtime = None
    try:
        import tflite_runtime.interpreter as tflite
        Runtime = tflite
    except Exception:
        try:
            from tensorflow.lite import Interpreter as TFInterpreter
            class _TFWrap: Interpreter = TFInterpreter
            Runtime = _TFWrap
        except Exception as e:
            raise RuntimeError("No TFLite interpreter available. Install tflite_runtime or TensorFlow.") from e
    threads = num_threads if (num_threads and num_threads > 0) else max(1, (os.cpu_count() or 4) - 1)
    print(f"[INFO] Loading TFLite model: {model_path}  (threads={threads})")
    itp = Runtime.Interpreter(model_path=str(model_path), num_threads=threads)
    itp.allocate_tensors()
    return itp

# ====================== CLASSIFIER (TFLite) ======================
class TFLiteClassifier:
    """ NHWC RGB float/quantized input → logits/probs vector output. """
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
        self.in_shape = self.in_details['shape']  # [1,H,W,C]
        if len(self.in_shape) != 4 or self.in_shape[-1] != 3:
            raise ValueError(f"Expected NHWC input, got {self.in_shape}")
        _, self.in_h, self.in_w, self.in_c = self.in_shape

        # Warm-up
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=self.in_dtype)
        self.interpreter.set_tensor(self.input_index, dummy)
        self.interpreter.invoke()

    @property
    def target_hw(self):
        return (int(self.in_w), int(self.in_h))  # (W,H)

    def output_size(self):
        shp = self.out_details.get('shape', None)
        return int(np.prod(shp)) if shp is not None else None

    def _quantize_input(self, x_float01):
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
            return x.astype(self.in_dtype)
        return np.clip(x_float01, 0.0, 1.0).astype(self.in_dtype)

    def _dequantize_output(self, y):
        if np.issubdtype(y.dtype, np.integer) and self.out_scale and self.out_scale > 0:
            return (y.astype(np.float32) - self.out_zero) * self.out_scale
        return y.astype(np.float32)

    def predict_rgb01(self, img_rgb01):
        x = self._quantize_input(img_rgb01)[None, ...]
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_index)[0]
        return self._dequantize_output(y)

# ====================== DETECTOR (TFLite YOLOv5-like) ======================
def _nms_xyxy(boxes, scores, iou_thres=0.45, top_k=TOPK):
    if boxes.size == 0: return []
    x1,y1,x2,y2 = boxes.T
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1][:top_k]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds+1]
    return keep

class TFLiteDetector:
    """ YOLOv5-like TFLite detector. Assumes normalized [cx,cy,w,h,score,cls] rows. """
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

        # Warm-up
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=self.in_dtype)
        self.interpreter.set_tensor(self.input_index, dummy)
        self.interpreter.invoke()

    @property
    def input_hw(self):  # (W,H)
        return (self.in_w, self.in_h)

    def _quantize_input(self, x_float01):
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
            return x.astype(self.in_dtype)
        return np.clip(x_float01, 0.0, 1.0).astype(self.in_dtype)

    def _gather_outputs(self):
        outs = [self.interpreter.get_tensor(i) for i in self.out_indices]
        dq = []
        for i, out in enumerate(outs):
            det = self.out_details[i]
            scale, zero = det.get('quantization', (0.0, 0))
            if np.issubdtype(out.dtype, np.integer) and scale and scale > 0:
                out = (out.astype(np.float32) - zero) * scale
            dq.append(out)
        dq.sort(key=lambda a: int(np.prod(a.shape)), reverse=True)
        return dq[0]

    def _parse(self, raw, in_size):
        arr = raw
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] < 6:
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,),   np.float32),
                    np.zeros((0,),   np.int32))
        iw, ih = in_size
        cx, cy, w, h = arr[:,0], arr[:,1], arr[:,2], arr[:,3]
        scores = arr[:,4].astype(np.float32)
        classes= arr[:,5].astype(np.int32)
        x1 = (cx - w/2.0) * iw; y1 = (cy - h/2.0) * ih
        x2 = (cx + w/2.0) * iw; y2 = (cy + h/2.0) * ih
        boxes = np.stack([x1,y1,x2,y2], axis=1).astype(np.float32)
        return boxes, scores, classes

    def detect(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        iw, ih = self.in_w, self.in_h
        padded_rgb01, ratio, pad = letterbox(frame_bgr, (iw, ih))
        x = self._quantize_input(padded_rgb01)[None, ...]
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        raw = self._gather_outputs()
        boxes_in, scores, classes = self._parse(raw, (iw, ih))
        if boxes_in.size == 0:
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,),   np.float32),
                    np.zeros((0,),   np.int32))
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
        boxes_out = scale_boxes_back(boxes_in, ratio, pad, W, H)
        return boxes_out, scores, classes

# ====================== UTILITIES ======================
def letterbox(im, new_shape, color=(114,114,114)):
    h, w = im.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    unpad_w = int(round(w * r)); unpad_h = int(round(h * r))
    dw = new_w - unpad_w; dh = new_h - unpad_h
    dw2 = dw // 2; dh2 = dh // 2
    if (h, w) != (unpad_h, unpad_w):
        im = cv2.resize(im, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = dh2, dh - dh2
    left, right = dw2, dw - dw2
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return im, (r, r), (dw2, dh2)

def scale_boxes_back(xyxy_in, ratio, pad, orig_w, orig_h):
    if xyxy_in.size == 0: return xyxy_in
    r_w, r_h = ratio; dw, dh = pad
    boxes = xyxy_in.copy().astype(np.float32)
    boxes[:, [0,2]] -= dw; boxes[:, [1,3]] -= dh
    boxes[:, [0,2]] /= r_w; boxes[:, [1,3]] /= r_h
    boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, orig_w-1)
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, orig_h-1)
    return boxes

def expand_square(x1, y1, x2, y2, img_w, img_h, pad=1.25):
    cx = 0.5*(x1 + x2); cy = 0.5*(y1 + y2)
    w  = max(1, x2 - x1); h  = max(1, y2 - y1)
    s  = pad * max(w, h); half = s/2
    nx1 = max(0, cx - half); ny1 = max(0, cy - half)
    nx2 = min(img_w, cx + half); ny2 = min(img_h, cy + half)
    if (nx2 - nx1) < 20 or (ny2 - ny1) < 20:
        nx1 = max(0, x1 - 10); ny1 = max(0, y1 - 10)
        nx2 = min(img_w, x2 + 10); ny2 = min(img_h, y2 + 10)
    return int(nx1), int(ny1), int(nx2), int(ny2)

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x -= np.max(x)
    e = np.exp(x)
    return (e / (np.sum(e) + 1e-12)).astype(np.float32)

def _softmax_if_needed(vec):
    v = np.asarray(vec, dtype=np.float64)
    if np.all(v >= -1e-6) and np.all(v <= 1.0 + 1e-6):
        s = v.sum()
        if 0.98 <= s <= 1.02:
            return v.astype(np.float32)
    v -= v.max()
    e = np.exp(v)
    return (e / (e.sum() + 1e-9)).astype(np.float32)

def load_labels_file(path):
    """
    Load class labels from .json (list[str] or {'classes'|'labels'|'names':[]})
    or from .txt (one per line).
    """
    if not path:
        return []
    ext = Path(path).suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            obj = json.load(f)
            if isinstance(obj, list):
                return [str(x) for x in obj]
            if isinstance(obj, dict):
                for k in ("classes","labels","names"):
                    if k in obj and isinstance(obj[k], list):
                        return [str(x) for x in obj[k]]
            raise ValueError("Unsupported JSON schema for labels; use list[str] or dict with classes/labels/names.")
        else:
            return [ln.strip() for ln in f if ln.strip()]

def reconcile_labels_with_output(labels, clf_out_details):
    """Pad or truncate labels to match classifier output vector length."""
    out_shape = clf_out_details.get('shape', None)
    n_out = int(np.prod(out_shape)) if out_shape is not None else len(labels)
    if len(labels) < n_out:
        base = len(labels)
        labels = labels + [f"cls_{i}" for i in range(base, n_out)]
        print(f"[WARN] Labels shorter ({base}) than model outputs ({n_out}); padded.")
    elif len(labels) > n_out:
        print(f"[WARN] Labels longer ({len(labels)}) than model outputs ({n_out}); truncating.")
        labels = labels[:n_out]
    return labels, n_out

def infer_labels_from_folders(root):
    root = Path(root)
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def iter_images_with_labels(root, exts={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}):
    root = Path(root)
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        lab = class_dir.name
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in exts:
                yield img_path, lab

def confusion_matrix(nc): return np.zeros((nc, nc), dtype=np.int32)

# ---------- Two-hand merge helpers ----------
def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
    inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0,inter_x2-inter_x1), max(0,inter_y2-inter_y1)
    inter = iw*ih
    if inter == 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

def should_merge(b1,b2):
    (lx1,ly1,lx2,ly2),(rx1,ry1,rx2,ry2) = sorted([b1,b2], key=lambda b:(b[0]+b[2])/2)
    ov_y = max(0, min(ly2, ry2) - max(ly1, ry1))
    h1, h2 = ly2-ly1, ry2-ry1
    v_overlap_ratio = ov_y / max(1, min(h1,h2))
    gap = max(0, rx1 - lx2)
    w1, w2 = lx2-lx1, rx2-rx1
    gap_ratio = gap / max(1, min(w1,w2))
    return (v_overlap_ratio >= 0.5 and gap_ratio <= 0.35) or (iou(b1,b2) >= 0.15)

def union_box(b1,b2):
    return (min(b1[0],b2[0]), min(b1[1],b2[1]), max(b1[2],b2[2]), max(b1[3],b2[3]))

# ====================== CLI ======================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate ASL classifier with TFLite YOLOv5 detector (two-hand merge + labels reconciliation)"
    )
    ap.add_argument("--det_tflite",   type=str, default="../models/v5n_100epoch_img416.tflite", help="YOLO hand detector (.tflite)")
    ap.add_argument("--det_threads",  type=int, default=0, help="Detector TFLite threads (0=auto)")
    ap.add_argument("--yolo_conf",    type=float, default=0.30, help="Detection confidence threshold")
    ap.add_argument("--yolo_iou",     type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--crop_pad",     type=float, default=0.77, help="Padding for square crop")
    ap.add_argument("--min_box_size", type=int,   default=10,   help="Minimum bbox size (pixels) to accept a detection")

    ap.add_argument("--cls_tflite",   type=str, default="../models/model_all.tflite", help="ASL classifier (.tflite)")
    ap.add_argument("--cls_threads",  type=int, default=0, help="Classifier TFLite threads (0=auto)")

    ap.add_argument("--data_dir",     type=str, default="../test_img", help="Dataset: test_img/<CLASS>/*.jpg …")
    ap.add_argument("--labels_file",  type=str, default="../models/labels_classifier.json",
                    help="Labels (.json list/dict or .txt lines). Will be reconciled to model output size.")

    ap.add_argument("--limit",        type=int, default=0, help="Cap #images for a quick test")
    ap.add_argument("--batch_size",   type=int, default=32, help="Metrics aggregation batch size")
    ap.add_argument("--out_dir",      type=str, default="eval_results_combined", help="Output root folder")
    return ap.parse_args()

# ====================== MAIN ======================
def main():
    args = parse_args()

    # Models
    det = TFLiteDetector(args.det_tflite, num_threads=args.det_threads,
                         conf_thres=args.yolo_conf, iou_thres=args.yolo_iou)
    clf = TFLiteClassifier(args.cls_tflite, num_threads=args.cls_threads)
    in_w, in_h = clf.target_hw
    print(f"[INFO] Detector input WxH: {det.input_hw} | Classifier input WxH: {(in_w, in_h)}")

    # Labels
    if args.labels_file and os.path.exists(args.labels_file):
        labels = load_labels_file(args.labels_file)
        labels, n_out = reconcile_labels_with_output(labels, clf.out_details)
        print(f"[INFO] Using labels from {args.labels_file} (aligned to {n_out} outputs).")
    else:
        labels = infer_labels_from_folders(args.data_dir)
        # Best effort alignment without out_details
        print(f"[INFO] Inferred {len(labels)} labels from folder names.")
    if not labels:
        print("[ERROR] No labels available.")
        sys.exit(1)

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    num_classes = len(labels)

    # Outputs
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"eval_yolo_v5_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    cm_csv_path = out_dir / "confusion_matrix.csv"
    cm_png_path = out_dir / "confusion_matrix.png"
    summary_json = out_dir / "summary.json"
    xlsx_path = out_dir / "metrics.xlsx"

    # Confusion + counters
    cm = confusion_matrix(num_classes)
    total = 0; correct = 0
    per_class_counts = Counter(); per_class_correct = Counter()

    # Detection stats
    num_images = 0
    num_detected = 0
    num_too_small = 0
    num_fallback_fullimg = 0

    # CSV header
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("path,true_label,true_idx,pred_label,pred_idx,correct,prob_top1,det_conf,det_box\n")

    # Metrics collectors
    metrics_rows = []
    start_wall = perf_counter()
    last_ms = last_cpu = last_ram = last_acc = 0.0

    batch_imgs = []
    batch_gts  = []
    batch_paths= []

    def get_cpu_ram():
        return float(psutil.cpu_percent(interval=0.0)), float(psutil.virtual_memory().percent)

    def classify_roi(roi_bgr):
        roi_blur = cv2.GaussianBlur(roi_bgr, (3, 3), 0)
        img = cv2.resize(roi_blur, (in_w, in_h), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.clip(img * 1.1 - 0.05, 0.0, 1.0)
        scores = clf.predict_rgb01(img)
        return _softmax_if_needed(scores)

    def flush_batch():
        nonlocal total, correct, cm, metrics_rows
        nonlocal num_detected, num_too_small, num_fallback_fullimg
        if not batch_imgs:
            return None

        t0 = perf_counter()
        preds, confs, det_confs, det_boxes = [], [], [], []

        for img in batch_imgs:
            # 1) detect hands (allow up to two, then optionally merge)
            boxes, scores, _ = det.detect(img)
            det_box = None; det_conf = 0.0

            if boxes.shape[0] > 0:
                # sort by area desc
                areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
                idxs = np.argsort(-areas)
                boxes = boxes[idxs]; scores = scores[idxs]
                # keep top2
                boxes = boxes[:2]; scores = scores[:2]

                # filter tiny if multiple
                if boxes.shape[0] > 1 and args.min_box_size > 0:
                    min_area = float(args.min_box_size * args.min_box_size)
                    mask = ((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])) >= min_area
                    boxes = boxes[mask]; scores = scores[mask]
                    if boxes.shape[0] == 0:
                        num_too_small += 1

                if boxes.shape[0] == 2 and should_merge(tuple(map(int, boxes[0])), tuple(map(int, boxes[1]))):
                    det_box = union_box(tuple(map(int, boxes[0])), tuple(map(int, boxes[1])))
                    det_conf = float(np.max(scores))
                elif boxes.shape[0] >= 1:
                    det_box = tuple(map(int, boxes[0]))
                    det_conf = float(scores[0])

            # classify
            if det_box is not None:
                num_detected += 1
                x1,y1,x2,y2 = det_box
                H, W = img.shape[:2]
                x1,y1,x2,y2 = expand_square(x1, y1, x2, y2, W, H, pad=args.crop_pad)
                roi = img[y1:y2, x1:x2]
                if roi.size <= 0:
                    roi = img
                    num_fallback_fullimg += 1
            else:
                roi = img
                num_fallback_fullimg += 1

            probs = classify_roi(roi)
            p = int(np.argmax(probs)); c = float(np.max(probs))
            preds.append(p); confs.append(c)
            det_confs.append(det_conf)
            det_boxes.append(det_box if det_box is not None else (0,0,0,0))

        t1 = perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        # 2) update stats + write CSV
        with csv_path.open("a", encoding="utf-8") as f:
            for i, p in enumerate(preds):
                gt_lab = batch_gts[i]; path = batch_paths[i]
                if gt_lab not in label_to_idx:
                    continue
                gt_idx = label_to_idx[gt_lab]
                cm[gt_idx, p] += 1
                per_class_counts[gt_lab] += 1
                ok = int(p == gt_idx)
                total += 1; correct += ok
                if ok: per_class_correct[gt_lab] += 1
                f.write(f"{path},{gt_lab},{gt_idx},{labels[p]},{p},{ok},{confs[i]:.6f},{det_confs[i]:.4f},{det_boxes[i]}\n")

        running_acc = (correct / max(1, total))
        cpu_pct, ram_pct = get_cpu_ram()
        now = datetime.now().isoformat(timespec="seconds")
        metrics_rows.append({
            "timestamp": now,
            "batch_size": len(batch_imgs),
            "images_seen_total": total,
            "running_accuracy": running_acc,
            "inference_ms": infer_ms,
            "inference_ms_per_image": infer_ms / max(1, len(batch_imgs)),
            "cpu_percent": cpu_pct,
            "ram_percent": ram_pct,
        })

        batch_imgs.clear(); batch_gts.clear(); batch_paths.clear()
        return infer_ms, cpu_pct, ram_pct, running_acc

    # Build dataset list
    all_items = list(iter_images_with_labels(args.data_dir))
    if args.limit: all_items = all_items[:args.limit]
    if not all_items:
        print("[ERROR] No images found. Check --data_dir.")
        sys.exit(1)

    # Eval loop
    pbar = tqdm(all_items, desc="Eval [yolo(tflite)+cls]", total=len(all_items))
    for img_path, gt_label in pbar:
        img = cv2.imread(str(img_path))
        if img is None: continue
        num_images += 1
        batch_imgs.append(img); batch_gts.append(gt_label); batch_paths.append(str(img_path))

        if len(batch_imgs) >= args.batch_size:
            out = flush_batch()
            if out is not None:
                last_ms, last_cpu, last_ram, last_acc = out
            pbar.set_postfix({
                "acc": f"{last_acc*100:.2f}%",
                "ms/batch": f"{last_ms:.1f}",
                "cpu%": f"{last_cpu:.0f}",
                "ram%": f"{last_ram:.0f}",
            })

    out = flush_batch()
    if out is not None:
        last_ms, last_cpu, last_ram, last_acc = out

    # Final metrics + saves
    overall_acc = correct / max(1, total)
    per_class_acc = {lab: (per_class_correct[lab] / per_class_counts[lab]) if per_class_counts[lab] else 0.0
                     for lab in labels}
    elapsed_s = perf_counter() - start_wall

    np.savetxt(cm_csv_path, cm, fmt="%d", delimiter=",")

    det_rate = (num_detected / max(1, num_images))
    fallback_rate = (num_fallback_fullimg / max(1, num_images))

    summary = {
        "pipeline": "yolo_tflite_detector + tflite_classifier",
        "detector_tflite": args.det_tflite,
        "classifier_tflite": args.cls_tflite,
        "data_dir": str(args.data_dir),
        "num_classes": num_classes,
        "num_images": int(total),
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": [float(per_class_acc[lab]) for lab in labels],
        "classes": labels,
        "cm_csv": str(cm_csv_path),
        "predictions_csv": str(csv_path),
        "wall_time_sec": float(elapsed_s),
        "avg_inference_ms_per_batch": float(np.mean([r["inference_ms"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_inference_ms_per_image": float(np.mean([r["inference_ms_per_image"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_cpu_percent": float(np.mean([r["cpu_percent"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_ram_percent": float(np.mean([r["ram_percent"] for r in metrics_rows])) if metrics_rows else 0.0,
        # detection stats
        "detected_images": int(num_detected),
        "no_usable_box_small": int(num_too_small),
        "fallback_full_image": int(num_fallback_fullimg),
        "detection_rate": float(det_rate),
        "fallback_rate": float(fallback_rate),
        "det_conf_thres": float(args.yolo_conf),
        "det_iou_thres": float(args.yolo_iou),
        "crop_pad": float(args.crop_pad),
        "min_box_size": int(args.min_box_size),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if _HAS_MPL:
        try:
            fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Confusion Matrix — Acc {overall_acc*100:.2f}%")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            tick_step = max(1, len(labels)//16) if len(labels) > 0 else 1
            ticks = list(range(0, len(labels), tick_step))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=7)
            ax.set_yticklabels([labels[i] for i in ticks], fontsize=7)
            plt.tight_layout()
            fig.savefig(cm_png_path)
            plt.close(fig)
        except Exception as e:
            print(f"[warn] matplotlib plot failed: {e}")

    # Excel workbook
    try:
        df_batches = pd.DataFrame(metrics_rows)
        df_summary = pd.DataFrame([summary])
        try:
            import xlsxwriter  # noqa: F401
            engine = "xlsxwriter"
        except Exception:
            engine = "openpyxl"
        with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
            df_batches.to_excel(writer, index=False, sheet_name="batches")
            df_summary.to_excel(writer, index=False, sheet_name="summary")
    except Exception as e:
        print(f"[warn] writing Excel failed: {e}")

    # Console summary
    print("\n===== RESULTS (YOLOv5-tflite + classifier tflite, merged logic) =====")
    print(f"Overall accuracy: {overall_acc*100:.2f}%  ({correct}/{total})")
    print(f"Detections merged/used: {num_detected}/{num_images}  (fallback {fallback_rate*100:.1f}%)\n")

    print("Per-class accuracy:")
    for lab in labels:
        n = per_class_counts[lab]
        k = per_class_correct[lab]
        acc = (k / n) if n > 0 else 0.0
        print(f"  {lab:>12s}: {acc*100:.2f}%  ({k}/{n})")

    print("\nConfusion Matrix (rows = GT, cols = Pred):")
    header = "pred→  " + " ".join([f"{lab:>6s}" for lab in labels])
    print(header)
    for r, lab in enumerate(labels):
        row = " ".join([f"{cm[r,c]:6d}" for c in range(num_classes)])
        print(f"{lab:>6s} {row}")

    print(f"\nPredictions CSV: {csv_path}")
    print(f"Confusion matrix CSV: {cm_csv_path}")
    print(f"Summary JSON: {summary_json}")
    if Path(cm_png_path).exists():
        print(f"Confusion matrix PNG: {cm_png_path}")
    if Path(xlsx_path).exists():
        print(f"Metrics Excel: {xlsx_path}")

if __name__ == "__main__":
    main()
