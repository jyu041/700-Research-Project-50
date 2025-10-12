#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# headless-safe
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys, cv2, argparse, json, time
from time import perf_counter
from pathlib import Path
from collections import Counter
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

# -------------------- Backends --------------------
class BackendBase:
    """Common interface for backends: must implement target_hw() and predict(bgr_image)->scores"""
    def target_hw(self): ...
    def predict(self, bgr_image): ...

# --------- TensorFlow (SavedModel / Keras) backend ---------
class TFClassifier(BackendBase):
    """
    Loads a TF SavedModel directory OR a Keras model file (.keras/.h5).
    Assumes NHWC input (H,W,3). If the model’s first layer has a different size,
    we read it from model.inputs[0].shape. We do not attempt channels-first here.
    """
    def __init__(self, model_path, threads=0):
        try:
            import tensorflow as tf
        except Exception as e:
            raise RuntimeError("TensorFlow is required for --backend tf. Install 'tensorflow'.") from e
        self.tf = tf

        # Optional thread control (CPU); safe defaults
        if threads and threads > 0:
            try:
                tf.config.threading.set_intra_op_parallelism_threads(threads)
                tf.config.threading.set_inter_op_parallelism_threads(max(1, threads // 2))
            except Exception:
                pass

        print(f"[INFO] Loading TensorFlow model: {model_path}")
        mp = str(model_path)
        self.model = None
        self.fn = None  # callable for inference

        # Try Keras load first (works for .keras/.h5 and many SavedModel exports)
        try:
            self.model = self.tf.keras.models.load_model(mp, compile=False)
            self._init_from_keras()
        except Exception as e_keras:
            # Fallback: SavedModel signatures
            try:
                sm = self.tf.saved_model.load(mp)
                # pick serving_default if present
                sig = sm.signatures.get("serving_default", None)
                if sig is None:
                    # if only one signature, take that
                    sigs = list(sm.signatures.values())
                    if not sigs:
                        raise RuntimeError("No callable signature in SavedModel.")
                    sig = sigs[0]
                self.fn = sig
                # infer input shape from signature
                in_tensor = list(sig.structured_input_signature[1].values())[0]
                shape = in_tensor.shape  # (None, H, W, 3) expected
                if len(shape) != 4 or shape[-1] != 3:
                    raise ValueError(f"Expected NHWC [B,H,W,3] input, got {tuple(shape)}")
                self.in_h, self.in_w, self.in_c = int(shape[1]), int(shape[2]), int(shape[3])
            except Exception as e_sig:
                raise RuntimeError(
                    f"Failed to load model as Keras ({e_keras}) and as SavedModel ({e_sig})."
                )

        # Warmup
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=np.float32)
        _ = self._predict_rgb01(dummy)

    def _init_from_keras(self):
        # Keras Functional/Sequential models expose .inputs
        if not hasattr(self.model, "inputs") or not self.model.inputs:
            raise RuntimeError("Keras model has no .inputs; can’t infer input shape.")
        ishape = self.model.inputs[0].shape  # (None,H,W,3)
        if len(ishape) != 4 or ishape[-1] != 3:
            raise ValueError(f"Expected NHWC [B,H,W,3] input, got {tuple(ishape)}")
        self.in_h, self.in_w, self.in_c = int(ishape[1]), int(ishape[2]), int(ishape[3])
        self.fn = self.model.__call__  # use Keras call

    def target_hw(self):
        return (int(self.in_w), int(self.in_h))

    def _predict_rgb01(self, x01_batched):
        # x01_batched: float32 [N,H,W,3] in 0..1
        y = self.fn(x01_batched, training=False)
        # Handle EagerTensors
        try:
            y = y.numpy()
        except Exception:
            y = np.array(y)
        # If batched output, squeeze to [C]
        if y.ndim > 1 and y.shape[0] == 1:
            y = y[0]
        return y.astype(np.float32)

    def predict(self, bgr_image):
        x = cv2.resize(bgr_image, self.target_hw(), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = x[None, ...]  # [1,H,W,3]
        return self._predict_rgb01(x)

# --------- TFLite backend (kept for convenience/compat) ---------
class TFLiteClassifier(BackendBase):
    def __init__(self, model_path, threads=0):
        try:
            import tflite_runtime.interpreter as tflite
            Runtime = tflite
        except Exception:
            try:
                from tensorflow.lite import Interpreter as TFInterpreter
            except Exception as e:
                raise RuntimeError("Install 'tflite_runtime' or 'tensorflow' to run TFLite.") from e
            class _Wrap: Interpreter = TFInterpreter
            Runtime = _Wrap

        threads = threads if (threads and threads > 0) else max(1, (os.cpu_count() or 4) - 1)
        print(f"[INFO] Loading TFLite model: {model_path}  (threads={threads})")
        self.itp = Runtime.Interpreter(model_path=str(model_path), num_threads=threads)
        self.itp.allocate_tensors()

        self.in_info  = self.itp.get_input_details()[0]
        self.out_info = self.itp.get_output_details()[0]
        self.in_idx   = self.in_info['index']
        self.out_idx  = self.out_info['index']
        self.in_dtype = self.in_info['dtype']
        self.out_dtype= self.out_info['dtype']

        # quantization (tuple or dict depending on runtime)
        self.in_q  = self._parse_q(self.in_info)
        self.out_q = self._parse_q(self.out_info)

        self.in_shape = self.in_info['shape']  # assume [1,H,W,3]
        if len(self.in_shape) != 4 or self.in_shape[-1] != 3:
            raise ValueError(f"Expected NHWC input, got {self.in_shape}")
        _, self.in_h, self.in_w, self.in_c = self.in_shape

        # Warmup
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=self.in_dtype)
        self.itp.set_tensor(self.in_idx, dummy); self.itp.invoke()

    def _parse_q(self, info):
        qp = info.get('quantization_parameters', None)
        if qp and 'scales' in qp and 'zero_points' in qp and len(qp['scales']) > 0:
            return (float(qp['scales'][0]), int(qp['zero_points'][0]))
        q = info.get('quantization', None)
        if isinstance(q, (tuple, list)) and len(q) == 2:
            return (float(q[0]), int(q[1]))
        return (0.0, 0)

    def target_hw(self):
        return (int(self.in_w), int(self.in_h))

    def _quantize_in(self, rgb01):
        scale, zero = self.in_q
        if np.issubdtype(self.in_dtype, np.integer):
            if scale and scale > 0:
                x = rgb01 / scale + zero
            else:
                x = rgb01 * 255.0
            return np.clip(x, 0, 255).astype(self.in_dtype)
        return rgb01.astype(self.in_dtype)

    def _dequantize_out(self, y):
        if np.issubdtype(y.dtype, np.integer):
            scale, zero = self.out_q
            if scale and scale > 0:
                y = (y.astype(np.float32) - zero) * scale
        return y

    def predict(self, bgr_image):
        x = cv2.resize(bgr_image, self.target_hw(), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = self._quantize_in(x)[None, ...]
        self.itp.set_tensor(self.in_idx, x)
        self.itp.invoke()
        y = self.itp.get_tensor(self.out_idx)[0]
        y = self._dequantize_out(y).astype(np.float32)
        return y  # logits or probs

# -------------------- Utilities --------------------
def _load_labels_lines(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        labs = [ln.strip() for ln in f.readlines() if ln.strip()]
    return labs

def _load_labels_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        for key in ("classes", "labels", "names"):
            if key in obj and isinstance(obj[key], list):
                return [str(x) for x in obj[key]]
    raise ValueError(f"Unsupported JSON structure in {path}; expected a list or a dict with 'classes'/'labels'/'names'.")

def load_labels(path: Path):
    if not path or not str(path):
        return []
    ext = path.suffix.lower()
    if ext == ".json":
        return _load_labels_json(path)
    return _load_labels_lines(path)

def infer_labels_from_folders(data_dir):
    classes = sorted([p.name for p in Path(data_dir).iterdir() if p.is_dir()])
    return classes

def iter_images_with_labels(data_dir, exts={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}):
    data_dir = Path(data_dir)
    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in exts:
                yield img_path, label

def confusion_matrix(num_classes):
    return np.zeros((num_classes, num_classes), dtype=np.int32)

def top1_index(scores):
    return int(np.argmax(scores))

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

# -------------------- Main eval with dynamic-style metrics --------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate a TF/TFLite classifier on an image folder with live metrics/logging")
    ap.add_argument("--backend", type=str, default="auto", choices=["auto","tf","tflite"],
                    help="Which backend to use. 'auto' picks 'tflite' for .tflite files, else 'tf'.")
    ap.add_argument("--model", default='../models/model_all.tflite', type=str,
                    help="Path to model: TF SavedModel dir / .keras / .h5 / .tflite")
    ap.add_argument("--data_dir", default='../test_img', type=str,
                    help="Dataset root: data/CLASS_A/*.jpg, data/CLASS_B/*.jpg, ...")
    ap.add_argument("--labels_file", type=str, default="..models/labels_classifier.json",
                    help="Labels file. Supports .json (array or {'classes':[...]}) or plain text (one label per line).")
    ap.add_argument("--threads", type=int, default=0, help="Num threads (TF CPU or TFLite). 0=auto")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on #images for quick test")
    ap.add_argument("--batch_size", type=int, default=32, help="Python-level batch size for metrics aggregation")
    ap.add_argument("--out_dir", type=str, default="eval_results_static", help="Output root folder")
    args = ap.parse_args()

    # Backend selection
    model_path = Path(args.model)
    backend = args.backend
    if backend == "auto":
        backend = "tflite" if model_path.suffix.lower() == ".tflite" else "tf"

    if backend == "tf":
        clf = TFClassifier(model_path, threads=args.threads)
    else:
        clf = TFLiteClassifier(model_path, threads=args.threads)

    W, H = clf.target_hw()

    # Labels
    labels = []
    if args.labels_file:
        try:
            labels = load_labels(Path(args.labels_file))
            print(f"[INFO] Loaded {len(labels)} labels from {args.labels_file}")
        except Exception as e:
            print(f"[WARN] Failed to load labels from {args.labels_file}: {e}")
    if not labels:
        labels = infer_labels_from_folders(args.data_dir)
        print(f"[INFO] Inferred {len(labels)} labels from folder names")
    if not labels:
        print("[ERROR] No labels found. Check --data_dir or --labels_file.")
        sys.exit(1)

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    num_classes = len(labels)

    # Outputs
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"{backend}_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    cm_csv_path = out_dir / "confusion_matrix.csv"
    cm_png_path = out_dir / "confusion_matrix.png"
    summary_json = out_dir / "summary.json"
    xlsx_path = out_dir / "metrics.xlsx"

    # Confusion matrix + counters
    cm = confusion_matrix(num_classes)
    total = 0
    correct = 0
    per_class_counts = Counter()
    per_class_correct = Counter()

    # Predictions CSV header
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("path,true_label,true_idx,pred_label,pred_idx,correct,prob_top1\n")

    # Metrics collectors (per batch)
    proc = psutil.Process(os.getpid())
    metrics_rows = []
    start_wall = perf_counter()

    batch_imgs = []
    batch_gts  = []
    batch_paths= []

    def get_cpu_ram():
        return float(psutil.cpu_percent(interval=0.0)), float(psutil.virtual_memory().percent)

    last_ms = last_cpu = last_ram = last_acc = 0.0

    def flush_batch():
        nonlocal total, correct, cm, metrics_rows
        if not batch_imgs:
            return None

        t0 = perf_counter()
        preds = []
        confs = []
        # run image-by-image (works for both TF and TFLite backends)
        for img in batch_imgs:
            scores = clf.predict(img)
            # If outputs are logits, turn into probs (safer for "prob_top1")
            probs = softmax_np(scores)
            preds.append(int(np.argmax(probs)))
            confs.append(float(np.max(probs)))
        t1 = perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        # update stats + write csv rows
        with csv_path.open("a", encoding="utf-8") as f:
            for i, p in enumerate(preds):
                gt_lab = batch_gts[i]
                path = batch_paths[i]
                if gt_lab not in label_to_idx:
                    continue
                gt_idx = label_to_idx[gt_lab]
                cm[gt_idx, p] += 1
                per_class_counts[gt_lab] += 1
                total += 1
                corr = int(p == gt_idx)
                correct += corr
                if corr: per_class_correct[gt_lab] += 1
                f.write(f"{path},{gt_lab},{gt_idx},{labels[p]},{p},{corr},{confs[i]:.6f}\n")

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

    # Iterate dataset
    all_items = list(iter_images_with_labels(args.data_dir))
    if args.limit:
        all_items = all_items[:args.limit]
    if not all_items:
        print("[ERROR] No images found. Check --data_dir.")
        sys.exit(1)

    pbar = tqdm(all_items, desc=f"Eval [{backend}]", total=len(all_items))
    for img_path, gt_label in pbar:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        batch_imgs.append(img)
        batch_gts.append(gt_label)
        batch_paths.append(str(img_path))

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

    # flush last partial batch
    out = flush_batch()
    if out is not None:
        last_ms, last_cpu, last_ram, last_acc = out

    # Final metrics
    overall_acc = correct / max(1, total)
    per_class_acc = {lab: (per_class_correct[lab] / per_class_counts[lab]) if per_class_counts[lab] else 0.0
                     for lab in labels}
    elapsed_s = perf_counter() - start_wall

    # Save confusion matrix CSV
    np.savetxt(cm_csv_path, cm, fmt="%d", delimiter=",")

    # Save summary JSON
    summary = {
        "backend": backend,
        "model_path": str(model_path),
        "data_dir": args.data_dir,
        "num_classes": num_classes,
        "num_images": int(total),
        "overall_accuracy": overall_acc,
        "per_class_accuracy": [per_class_acc[lab] for lab in labels],
        "classes": labels,
        "cm_csv": str(cm_csv_path),
        "predictions_csv": str(csv_path),
        "wall_time_sec": float(elapsed_s),
        "avg_inference_ms_per_batch": float(np.mean([r["inference_ms"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_inference_ms_per_image": float(np.mean([r["inference_ms_per_image"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_cpu_percent": float(np.mean([r["cpu_percent"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_ram_percent": float(np.mean([r["ram_percent"] for r in metrics_rows])) if metrics_rows else 0.0,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save confusion matrix PNG
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
    print("\n===== RESULTS =====")
    print(f"Overall accuracy: {overall_acc*100:.2f}%  ({correct}/{total})\n")
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
