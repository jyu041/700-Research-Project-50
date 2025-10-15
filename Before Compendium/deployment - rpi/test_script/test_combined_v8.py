#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined evaluator: YOLOv8 (.pt) hand detector + classifier (Keras .h5 or TFLite .tflite)
- Evaluates a folder dataset structured as data_dir/CLASS/*.jpg
- Adopts robust label handling and two-hand merge logic from v1.py
"""

import os, sys, cv2, argparse, json, time, csv
from time import perf_counter
from pathlib import Path
from collections import Counter, deque
from datetime import datetime
import numpy as np
from tqdm import tqdm

# metrics / excel deps
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

# ---- detection deps ----
try:
    from ultralytics import YOLO   # YOLOv8 (PyTorch)
except Exception as e:
    raise SystemExit("Ultralytics YOLO is required. Install: pip install ultralytics") from e

# ---- keras loader (optional if using .h5) ----
try:
    from tf_keras.models import load_model as _load_model
except Exception:
    try:
        from tensorflow.keras.models import load_model as _load_model
    except Exception:
        _load_model = None  # fine if using --tflite

# ====================== UTILITIES ======================
def softmax_np(vec: np.ndarray) -> np.ndarray:
    v = vec.astype(np.float64)
    v -= np.max(v)
    e = np.exp(v)
    s = e.sum() + 1e-12
    return (e / s).astype(np.float32)

def _softmax_if_needed(vec):
    v = np.asarray(vec, dtype=np.float64)
    if np.all(v >= -1e-6) and np.all(v <= 1.0 + 1e-6):
        s = v.sum()
        if 0.98 <= s <= 1.02:
            return v.astype(np.float32)
    v -= v.max()
    e = np.exp(v)
    sm = e / (e.sum() + 1e-9)
    return sm.astype(np.float32)

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

def load_labels_file(path):
    """
    Load class labels from .json (list[str] or {'classes'|'labels'|'names':[]})
    or from .txt (one label per line).
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
                for k in ("classes", "labels", "names"):
                    if k in obj and isinstance(obj[k], list):
                        return [str(x) for x in obj[k]]
            raise ValueError("Unsupported JSON schema for labels. Use list[str] or {'classes':[...]}.")
        else:
            return [ln.strip() for ln in f if ln.strip()]

def reconcile_labels_with_n(labels, n_out):
    """
    Ensure len(labels) matches classifier outputs by padding with 'cls_i' or truncating.
    """
    if n_out is None or n_out <= 0:
        return labels
    if len(labels) < n_out:
        base = len(labels)
        labels = labels + [f"cls_{i}" for i in range(base, n_out)]
        print(f"[WARN] Labels shorter ({base}) than model outputs ({n_out}); padded.")
    elif len(labels) > n_out:
        print(f"[WARN] Labels longer ({len(labels)}) than model outputs ({n_out}); truncating.")
        labels = labels[:n_out]
    return labels

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

def get_cpu_ram():
    return float(psutil.cpu_percent(interval=0.0)), float(psutil.virtual_memory().percent)

# ---------- IoU + two-hand merge helpers (from v1.py logic) ----------
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

def detect_hands(yolo, frame, conf, iou_thresh=0.45, max_det=4):
    res = yolo.predict(source=frame, conf=conf, iou=iou_thresh,
                       max_det=max_det, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    arr = res.boxes.xyxy.cpu().numpy()
    boxes = [tuple(map(int, b)) for b in arr]
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes[:2]

# ----------------------- TFLite wrapper (from v1.py) ----------------
class TFLiteClassifier:
    """
    Lightweight TFLite classifier wrapper (prefers tflite_runtime; falls back to TF Lite).
    Expects NHWC input, returns float logits/probs.
    """
    def __init__(self, tflite_path, num_threads=None):
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
        print(f"[INFO] Loading TFLite model: {tflite_path} (threads={threads})")
        interpreter = Runtime.Interpreter(model_path=tflite_path, num_threads=threads)
        interpreter.allocate_tensors()

        self.interpreter = interpreter
        self.in_details  = interpreter.get_input_details()[0]
        self.out_details = interpreter.get_output_details()[0]
        self.input_index = self.in_details['index']
        self.output_index= self.out_details['index']
        self.in_dtype    = self.in_details['dtype']
        self.out_dtype   = self.out_details['dtype']
        self.in_scale, self.in_zero   = (self.in_details.get('quantization') or (0.0, 0))
        self.out_scale, self.out_zero = (self.out_details.get('quantization') or (0.0, 0))

        self.in_shape = self.in_details['shape']  # [1,H,W,C]
        _, self.in_h, self.in_w, self.in_c = self.in_shape

        # warmup
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=self.in_dtype)
        interpreter.set_tensor(self.input_index, dummy)
        interpreter.invoke()

    @property
    def target_hw(self):
        return (int(self.in_w), int(self.in_h))  # (W,H) for cv2.resize

    def num_outputs(self):
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
        x = self._quantize_input(img_rgb_float01)[None, ...]
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_index)[0]
        y = self._dequantize_output(y)
        return y

# ====================== CLI ======================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate with YOLOv8 detector + classifier (Keras .h5 or TFLite .tflite) using v1.py logic"
    )
    # YOLOv8 detector
    ap.add_argument("--yolo_weights", type=str, default="../models/best_hand.pt", help="Ultralytics YOLOv8 .pt weights")
    ap.add_argument("--yolo_conf",    type=float, default=0.30, help="YOLO detection confidence threshold")
    ap.add_argument("--yolo_imgsz",   type=int,   default=416,  help="YOLO input size")
    ap.add_argument("--device",       type=str,   default="",   help='YOLO device, e.g. "cpu", "cuda:0"')

    ap.add_argument("--crop_pad",     type=float, default=1.25, help="Padding factor for square crop around detection")
    ap.add_argument("--min_box_size", type=int,   default=10,   help="Minimum bbox size (pixels) to accept a detection")

    # Classifier options
    ap.add_argument("--tflite",       type=str, default="", help="Path to classifier .tflite; if empty uses --cls_h5")
    ap.add_argument("--tflite_threads", type=int, default=0, help="TFLite num_threads; 0=auto")
    ap.add_argument("--cls_h5",       type=str, default="../models/model_all.h5", help="Classifier model (.h5), used if --tflite is empty")

    # Data
    ap.add_argument("--data_dir",     type=str, default="../test_img", help="Dataset: test_img/<CLASS>/*.jpg …")
    ap.add_argument("--labels_file",  type=str, default="/mnt/data/labels_classifier.json",
                    help="Labels file (.json list or dict with 'classes'/… or .txt lines)")

    # Run/IO
    ap.add_argument("--limit",        type=int, default=0,  help="Cap #images for a quick test")
    ap.add_argument("--batch_size",   type=int, default=32, help="Metrics aggregation batch size")
    ap.add_argument("--out_dir",      type=str, default="eval_results_combined", help="Output root folder")
    return ap.parse_args()

# ====================== MAIN EVAL ======================
def main():
    args = parse_args()

    # --- Load YOLOv8 detector (.pt) ---
    print(f"[INFO] Loading YOLOv8 weights: {args.yolo_weights}")
    yolo = YOLO(args.yolo_weights)
    if args.device:
        try:
            yolo.to(args.device)
            print(f"[INFO] YOLO device → {args.device}")
        except Exception as e:
            print(f"[WARN] Could not set YOLO device '{args.device}': {e}")

    # --- Load classifier (prefer TFLite if provided) ---
    is_tflite = False
    n_out = None
    if args.tflite:
        threads = args.tflite_threads if args.tflite_threads > 0 else max(1, (os.cpu_count() or 4)-1)
        clf = TFLiteClassifier(args.tflite, num_threads=threads)
        in_w, in_h = clf.target_hw
        is_tflite = True
        n_out = clf.num_outputs()
        print(f"[INFO] TFLite classifier input WxH: {(in_w, in_h)}, outputs: {n_out}")
    else:
        if _load_model is None:
            raise SystemExit("No Keras loader available and --tflite not set. Install tensorflow or tf-keras.")
        print(f"[INFO] Loading Keras (.h5) classifier: {args.cls_h5}")
        clf = _load_model(args.cls_h5, compile=False)
        in_shape = clf.input_shape
        if isinstance(in_shape, list): in_shape = in_shape[0]
        in_h, in_w = int(in_shape[1]), int(in_shape[2])
        if len(in_shape) < 4 or in_shape[3] != 3:
            print(f"[WARN] Classifier expects shape {in_shape}; proceeding assuming 3-channel RGB input.")
        out_shape = clf.output_shape
        if isinstance(out_shape, list): out_shape = out_shape[0]
        n_out = int(out_shape[-1]) if out_shape is not None else None

    # --- Labels ---
    if args.labels_file and os.path.exists(args.labels_file):
        labels = load_labels_file(args.labels_file)
        print(f"[INFO] Loaded {len(labels)} labels from {args.labels_file}")
    else:
        labels = infer_labels_from_folders(args.data_dir)
        print(f"[INFO] Inferred {len(labels)} labels from folder structure")
    labels = reconcile_labels_with_n(labels, n_out)
    if not labels:
        print("[ERROR] No labels available.")
        sys.exit(1)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    num_classes = len(labels)

    # --- Outputs ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"eval_yolo_v8_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    cm_csv_path = out_dir / "confusion_matrix.csv"
    cm_png_path = out_dir / "confusion_matrix.png"
    summary_json = out_dir / "summary.json"
    xlsx_path = out_dir / "metrics.xlsx"

    # --- Counters / CM ---
    cm = confusion_matrix(num_classes)
    total = correct = 0
    per_class_counts = Counter()
    per_class_correct = Counter()

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

    batch_imgs, batch_gts, batch_paths = [], [], []

    def classify_roi(roi_bgr):
        # Preprocess similar to v1.py
        img = cv2.resize(roi_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img)
        if is_tflite:
            raw = clf.predict(img)
        else:
            raw = clf.predict(np.expand_dims(img, 0), verbose=0)[0]
        return _softmax_if_needed(raw)

    def flush_batch():
        nonlocal total, correct, cm, metrics_rows
        nonlocal num_detected, num_too_small, num_fallback_fullimg
        if not batch_imgs:
            return None

        t0 = perf_counter()
        preds, confs, det_confs, det_boxes = [], [], [], []

        for img in batch_imgs:
            # 1) detect hands via YOLOv8 (per-image), get up to 2 boxes
            boxes = detect_hands(yolo, img, conf=args.yolo_conf, iou_thresh=0.45, max_det=4)

            # Filter too small and keep area-sorted
            if len(boxes) > 1 and args.min_box_size > 0:
                min_area = float(args.min_box_size * args.min_box_size)
                boxes = [b for b in boxes if (b[2]-b[0])*(b[3]-b[1]) >= min_area]
                if not boxes:
                    num_too_small += 1

            # Choose working box: merge if two close, else largest
            yolo_bbox = None
            this_det_conf = 0.0
            if len(boxes) == 2 and should_merge(boxes[0], boxes[1]):
                yolo_bbox = union_box(boxes[0], boxes[1])
                # best-effort conf (no class conf merging here)
                this_det_conf = 1.0
            elif len(boxes) >= 1:
                yolo_bbox = boxes[0]
                this_det_conf = 1.0
            else:
                yolo_bbox = None

            if yolo_bbox is not None:
                num_detected += 1
                x1,y1,x2,y2 = yolo_bbox
                H, W = img.shape[:2]
                x1, y1, x2, y2 = expand_square(x1, y1, x2, y2, W, H, pad=args.crop_pad)
                roi = img[y1:y2, x1:x2]
                if roi.size <= 0:
                    roi = img
                    num_fallback_fullimg += 1
            else:
                roi = img
                num_fallback_fullimg += 1

            # 2) classify ROI
            probs = classify_roi(roi)
            preds.append(int(np.argmax(probs)))
            confs.append(float(np.max(probs)))
            det_confs.append(float(this_det_conf))
            det_boxes.append(tuple(yolo_bbox) if yolo_bbox is not None else (0,0,0,0))

        t1 = perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        # 3) update stats + write rows
        with csv_path.open("a", encoding="utf-8") as f:
            for i, p in enumerate(preds):
                gt_lab = batch_gts[i]; path = batch_paths[i]
                if gt_lab not in label_to_idx:
                    continue
                gt_idx = label_to_idx[gt_lab]
                cm[gt_idx, p] += 1
                per_class_counts[gt_lab] += 1
                ok = int(p == gt_idx)
                nonlocal correct, total
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

    # --- Build dataset list ---
    all_items = list(iter_images_with_labels(args.data_dir))
    if args.limit: all_items = all_items[:args.limit]
    if not all_items:
        print("[ERROR] No images found. Check --data_dir.")
        sys.exit(1)

    # --- Eval loop ---
    num_images = len(all_items)
    last_ms = last_cpu = last_ram = last_acc = 0.0
    pbar = tqdm(all_items, desc="Eval [yolo(pt)+cls]", total=len(all_items))
    for img_path, gt_label in pbar:
        img = cv2.imread(str(img_path))
        if img is None:  # skip unreadables
            continue
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

    # --- Final metrics + saves ---
    overall_acc = correct / max(1, total)
    per_class_acc = {lab: (per_class_correct[lab] / per_class_counts[lab]) if per_class_counts[lab] else 0.0
                     for lab in labels}
    elapsed_s = perf_counter() - start_wall

    np.savetxt(cm_csv_path, cm, fmt="%d", delimiter=",")

    det_rate = (num_detected / max(1, num_images))
    fallback_rate = (num_fallback_fullimg / max(1, num_images))

    summary = {
        "pipeline": "yolo_v8_pt_detector + classifier(h5/tflite)",
        "yolo_weights": str(args.yolo_weights),
        "classifier_tflite": str(args.tflite) if args.tflite else "",
        "classifier_h5": str(args.cls_h5) if not args.tflite else "",
        "data_dir": str(args.data_dir),
        "num_classes": int(num_classes),
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
        "yolo_imgsz": int(args.yolo_imgsz),
        "crop_pad": float(args.crop_pad),
        "min_box_size": int(args.min_box_size),
        "device": args.device or "auto",
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if _HAS_MPL:
        try:
            fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Confusion Matrix — Acc {overall_acc*100:.2f}%")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
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
    print("\n===== RESULTS (YOLOv8 + classifier h5/tflite, v1.py logic) =====")
    print(f"Overall accuracy: {overall_acc*100:.2f}%  ({correct}/{total})")
    print(f"Detections used or merged: {num_detected}/{num_images}  "
          f"(rate {det_rate*100:.1f}%, fallback {fallback_rate*100:.1f}%)\n")

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
