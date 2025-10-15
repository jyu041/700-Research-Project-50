# --- snip: same imports as your last script ---
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import re, json, math, time
from time import perf_counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
from tqdm import tqdm

# metrics / excel
import psutil, pandas as pd

# optional readers
HAS_DECORD = False
try:
    import decord
    from decord import VideoReader, cpu
    HAS_DECORD = True
except Exception:
    pass

HAS_CV2 = False
try:
    import cv2
    HAS_CV2 = True
except Exception:
    pass

HAS_TVVR = False
try:
    from torchvision.io import VideoReader as TVVideoReader
    HAS_TVVR = True
except Exception:
    pass

# matplotlib (optional)
HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    pass

# =========================
# ====== CONFIG AREA ======
# =========================

TFLITE_MODEL_PATH = r"../models/dynamic.tflite"
META_JSON_PATH: Optional[str] = r"../models/dynamic.json"

DATA_DIR = r"../test_vid"
EVAL_SPLIT = "all"
MAX_VIDEOS: Optional[int] = 1280

# Default inference clip spec (will be overridden by model if fixed)
NUM_FRAMES_DEFAULT = 16
FRAME_SIZE_DEFAULT = 160

# Python-level batch size (interpreter runs 1-by-1 for portability)
INFER_BATCH_SIZE = 4

# Preprocess options
USE_TORCH_NORM = False
TORCH_MEAN = [0.485, 0.456, 0.406]
TORCH_STD  = [0.229, 0.224, 0.225]

OUTPUT_DIR = "eval_results_dynamic"

# =========================
# ===== END OF CONFIG =====
# =========================

FNAME_RE  = re.compile(r"^(?P<cls>\d{3})_(?P<actor>\d{3})_(?P<rep>\d{3})\.(?:mp4|avi|mov|mkv)$", re.IGNORECASE)
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# ---------- meta ----------

def load_meta(meta_path: Optional[str], model_path: str) -> dict:
    if meta_path and Path(meta_path).exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    cand = sorted(Path(model_path).parent.glob("*_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        raise FileNotFoundError("No *_meta.json found; set META_JSON_PATH.")
    with open(cand[0], "r") as f:
        return json.load(f)

def build_class_index(meta: dict) -> Tuple[List[str], Dict[str, int]]:
    classes = meta.get("classes", [])
    if not classes:
        raise RuntimeError("Could not recover class list from meta.")
    return classes, {c:i for i,c in enumerate(classes)}

# ---------- scan ----------

def scan_cut_folder(data_dir: str) -> List[Dict]:
    root = Path(data_dir)
    items: List[Dict] = []
    if not root.exists():
        return items
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            m = FNAME_RE.match(f.name)
            if m:
                items.append({
                    "path": str(f.resolve()),
                    "cls": m.group("cls"),
                    "actor": m.group("actor"),
                    "rep": m.group("rep"),
                })
    return items

def choose_items_by_split(all_items: List[Dict], meta: dict, split: str) -> List[Dict]:
    if split == "all":
        return list(all_items)
    actors = set(meta.get("actors", {}).get(split, []))
    return [x for x in all_items if x["actor"] in actors]

# ---------- readers ----------

def uniform_indices(n_total: int, n_samples: int):
    if n_total <= 0:
        return [0]*n_samples
    if n_samples <= 1:
        return [min(n_total-1, 0)]
    return [min(n_total-1, int(round(x))) for x in np.linspace(0, n_total-1, num=n_samples).tolist()]

def _resize_np(img: np.ndarray, size_hw: Tuple[int,int]) -> np.ndarray:
    Ht, Wt = size_hw
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s)//2; x0 = (w - s)//2
    crop = img[y0:y0+s, x0:x0+s]
    ys = np.linspace(0, s-1, Ht).astype(np.int32)
    xs = np.linspace(0, s-1, Wt).astype(np.int32)
    return crop[np.ix_(ys, xs)]

def sample_frames_cv2(path: str, num_frames: int, size_hw: Tuple[int,int]) -> np.ndarray:
    Ht, Wt = size_hw
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return np.zeros((num_frames, Ht, Wt, 3), dtype=np.uint8)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n_total <= 0:
        cap.release()
        return np.zeros((num_frames, Ht, Wt, 3), dtype=np.uint8)
    idxs = set(uniform_indices(n_total, num_frames))
    frames = []; i = 0; ok = True
    while ok and len(frames) < num_frames:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        if i in idxs:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
        i += 1
    cap.release()
    if len(frames) == 0:
        return np.zeros((num_frames, Ht, Wt, 3), dtype=np.uint8)
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames, axis=0)

def sample_frames_decord(path: str, num_frames: int, size_hw: Tuple[int,int]) -> np.ndarray:
    Ht, Wt = size_hw
    vr = VideoReader(path, ctx=cpu(0))
    n_total = len(vr)
    idxs = uniform_indices(n_total, num_frames)
    frames = vr.get_batch(idxs).asnumpy()  # [T,H,W,3]
    if HAS_CV2:
        frames = np.stack([cv2.resize(fr, (Wt, Ht), interpolation=cv2.INTER_AREA) for fr in frames], axis=0)
    else:
        frames = np.stack([_resize_np(fr, size_hw) for fr in frames], axis=0)
    return frames

def sample_frames_tvreader(path: str, num_frames: int, size_hw: Tuple[int,int]) -> np.ndarray:
    Ht, Wt = size_hw
    try:
        vr = TVVideoReader(path, "video")
    except Exception:
        return np.zeros((num_frames, Ht, Wt, 3), dtype=np.uint8)
    seq = []
    for frame in vr:
        img = frame["data"].permute(1,2,0).contiguous().numpy()
        if HAS_CV2:
            img = cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_AREA)
        else:
            img = _resize_np(img, size_hw)
        seq.append(img)
    if len(seq) == 0:
        return np.zeros((num_frames, Ht, Wt, 3), dtype=np.uint8)
    take = uniform_indices(len(seq), num_frames)
    return np.stack([seq[k] for k in take], axis=0)

def sample_frames(path: str, num_frames: int, size_hw: Tuple[int,int]) -> np.ndarray:
    if HAS_DECORD:
        return sample_frames_decord(path, num_frames, size_hw)
    if HAS_CV2:
        return sample_frames_cv2(path, num_frames, size_hw)
    if HAS_TVVR:
        return sample_frames_tvreader(path, num_frames, size_hw)
    Ht, Wt = size_hw
    return np.zeros((num_frames, Ht, Wt, 3), dtype=np.uint8)

# ---------- TFLite ----------

def _import_tflite():
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except Exception:
        from tensorflow.lite import Interpreter  # type: ignore
        return Interpreter

Interpreter = _import_tflite()

def create_interpreter(model_path: str):
    interpreter = Interpreter(model_path=model_path, num_threads=max(1, (os.cpu_count() or 2)//2))
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    if not in_details:
        raise RuntimeError("No input tensor in TFLite model.")
    return interpreter, in_details[0], out_details[0]

def parse_model_io(input_detail: dict):
    """
    Infer layout and required sizes from model input shape.
    Returns: dict with keys:
      layout: one of {"BTHWC","BHWC","BCHWT","BTCHW"}
      need_T: bool (True if fixed T is required)
      T,H,W,C: ints (None for dynamic)
      dtype: np.dtype
    """
    shape = input_detail["shape"]
    dtype = input_detail["dtype"]
    dims = [int(d) for d in shape]
    dims = [None if d < 0 else d for d in dims]

    layout = None
    T = H = W = C = None

    if len(dims) == 5:
        b, d1, d2, d3, d4 = dims
        # Try standard BTHWC
        if d4 == 3:
            # [B, T, H, W, 3]
            layout = "BTHWC"; T, H, W, C = d1, d2, d3, d4
        # Try channels-first, time-last BCHWT (your case)
        elif d1 == 3:
            # [B, 3, H, W, T]
            layout = "BCHWT"; C, H, W, T = d1, d2, d3, d4
        # Try time-first channels-first BTCHW
        elif d2 == 3:
            # [B, T, 3, H, W]
            layout = "BTCHW"; T, C, H, W = d1, d2, d3, d4
        else:
            raise ValueError(f"Unsupported 5D input layout: {dims}")

    elif len(dims) == 4:
        b, h, w, c = dims
        if c == 3:
            layout = "BHWC"; H, W, C = h, w, c
        else:
            raise ValueError(f"Unsupported 4D input layout: {dims}")
    else:
        raise ValueError(f"Unsupported input rank: {len(dims)} (shape={dims})")

    need_T = (layout in ("BTHWC","BCHWT","BTCHW") and T is not None and T > 0)
    return {"layout": layout, "need_T": need_T, "T": T, "H": H, "W": W, "C": C, "dtype": dtype}


def preprocess_frames(frames_uint8: np.ndarray,
                      input_detail: dict,
                      io_spec: dict,
                      use_torch_norm: bool) -> np.ndarray:
    """
    frames_uint8: [T,H,W,3] uint8 RGB already resized to target H,W (if known).
    Returns array matching interpreter dtype AND layout.
    """
    dtype   = io_spec["dtype"]
    layout  = io_spec["layout"]
    T_fixed = io_spec["T"]

    # Ensure time length matches fixed T (pad/trim)
    T_in = frames_uint8.shape[0]
    if T_fixed and T_in != T_fixed:
        if T_in < T_fixed:
            reps = (T_fixed + T_in - 1) // T_in
            frames_uint8 = np.tile(frames_uint8, (reps, 1, 1, 1))[:T_fixed]
        else:
            frames_uint8 = frames_uint8[:T_fixed]

    # Convert to float & normalize if needed
    def to_float(x):
        x = x.astype(np.float32) / 255.0
        if use_torch_norm:
            mean = np.array(TORCH_MEAN, dtype=np.float32).reshape(1,1,1,3)
            std  = np.array(TORCH_STD,  dtype=np.float32).reshape(1,1,1,3)
            x = (x - mean) / std
        return x

    # Arrange layout then quantize (if required)
    if np.issubdtype(dtype, np.floating):
        x = to_float(frames_uint8)
        if layout == "BTHWC":
            x = x[np.newaxis, ...]                      # [1,T,H,W,3]
        elif layout == "BHWC":
            x = x[0][np.newaxis, ...]                   # [1,H,W,3]
        elif layout == "BCHWT":
            x = np.transpose(x, (3,1,2,0))              # [3,H,W,T]
            x = x[np.newaxis, ...]                      # [1,3,H,W,T]
        elif layout == "BTCHW":
            x = np.transpose(x, (0,3,1,2))              # [T,3,H,W]
            x = x[np.newaxis, ...]                      # [1,T,3,H,W]
        else:
            raise ValueError(f"Unsupported layout {layout}")
        return x.astype(dtype)

    # Quantized path
    qparams = input_detail.get("quantization_parameters", {})
    scales = qparams.get("scales")
    zeros  = qparams.get("zero_points")
    s  = float(scales[0]) if (scales is not None and len(scales)>0) else 1.0
    zp = int(zeros[0])    if (zeros  is not None and len(zeros) >0) else 0

    x = to_float(frames_uint8)
    if layout == "BTHWC":
        x = x[np.newaxis, ...]                          # [1,T,H,W,3]
    elif layout == "BHWC":
        x = x[0][np.newaxis, ...]                       # [1,H,W,3]
    elif layout == "BCHWT":
        x = np.transpose(x, (3,1,2,0))                  # [3,H,W,T]
        x = x[np.newaxis, ...]                          # [1,3,H,W,T]
    elif layout == "BTCHW":
        x = np.transpose(x, (0,3,1,2))                  # [T,3,H,W]
        x = x[np.newaxis, ...]                          # [1,T,3,H,W]
    else:
        raise ValueError(f"Unsupported layout {layout}")

    q = np.round(x / s + zp)
    if dtype == np.int8:
        q = np.clip(q, -128, 127).astype(np.int8)
    else:
        q = np.clip(q, 0, 255).astype(np.uint8)
    return q


def infer_clip_tflite(interpreter, in_detail: dict, out_detail: dict, clip_np: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(in_detail["index"], clip_np)
    interpreter.invoke()
    logits = interpreter.get_tensor(out_detail["index"])
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    return logits.astype(np.float32)

# ---------- utils ----------

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

# ---------- main ----------

def evaluate():
    # Load meta / classes
    meta = load_meta(META_JSON_PATH, TFLITE_MODEL_PATH)
    classes, class_to_idx = build_class_index(meta)
    num_classes = len(classes)

    # Interpreter + IO spec
    interpreter, in_detail, out_detail = create_interpreter(TFLITE_MODEL_PATH)
    io_spec = parse_model_io(in_detail)

    # Derive target sizes
    layout = io_spec["layout"]
    T_target = io_spec["T"] if (io_spec["T"] and io_spec["T"] > 0) else NUM_FRAMES_DEFAULT
    H_target = io_spec["H"] if (io_spec["H"] and io_spec["H"] > 0) else FRAME_SIZE_DEFAULT
    W_target = io_spec["W"] if (io_spec["W"] and io_spec["W"] > 0) else FRAME_SIZE_DEFAULT

    print(f"[io] layout={layout}, dtype={io_spec['dtype']}, "
          f"T={io_spec['T']}, H={io_spec['H']}, W={io_spec['W']}. "
          f"Using T={T_target}, H={H_target}, W={W_target}")

    # Scan / split
    items_all = scan_cut_folder(DATA_DIR)
    items = choose_items_by_split(items_all, meta, EVAL_SPLIT)
    if MAX_VIDEOS is not None:
        items = items[:MAX_VIDEOS]
    assert items, f"No videos to evaluate in split={EVAL_SPLIT}."

    # Outputs
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(OUTPUT_DIR) / f"eval_{EVAL_SPLIT}_tflite_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    cm_csv_path = out_dir / "confusion_matrix.csv"
    cm_png_path = out_dir / "confusion_matrix.png"
    summary_json = out_dir / "summary.json"
    xlsx_path = out_dir / "metrics.xlsx"

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_class_counts = np.zeros(num_classes, dtype=np.int64)
    correct_total = 0

    batch_frames: List[np.ndarray] = []
    batch_labels: List[int] = []
    batch_paths:  List[str] = []

    proc = psutil.Process(os.getpid())
    metrics_rows = []
    start_wall = perf_counter()

    def get_cpu_ram():
        return float(psutil.cpu_percent(interval=0.0)), float(psutil.virtual_memory().percent)

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("path,true_cls,true_idx,pred_cls,pred_idx,correct,prob_top1\n")

    last_ms = last_cpu = last_ram = last_acc = 0.0

    def flush_batch():
        nonlocal correct_total, cm, per_class_counts, metrics_rows
        if not batch_frames:
            return None
        t0 = perf_counter()
        preds = []; confs = []
        for fr in batch_frames:
            # ensure correct sample count and size for the model
            if fr.shape[0] != T_target:
                idxs = uniform_indices(fr.shape[0], T_target)
                fr = fr[idxs]
            if fr.shape[1] != H_target or fr.shape[2] != W_target:
                if HAS_CV2:
                    fr = np.stack([cv2.resize(x, (W_target, H_target), interpolation=cv2.INTER_AREA) for x in fr], axis=0)
                else:
                    fr = np.stack([_resize_np(x, (H_target, W_target)) for x in fr], axis=0)

            clip_np = preprocess_frames(fr, in_detail, io_spec, USE_TORCH_NORM)
            logits = infer_clip_tflite(interpreter, in_detail, out_detail, clip_np)
            probs = softmax_np(logits)
            p = int(np.argmax(probs))
            preds.append(p); confs.append(float(np.max(probs)))
        t1 = perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        with csv_path.open("a", encoding="utf-8") as f:
            for i, p in enumerate(preds):
                t = int(batch_labels[i])
                cm[t, p] += 1
                per_class_counts[t] += 1
                correct_total += int(t == p)
                f.write(f"{batch_paths[i]},{classes[t]},{t},{classes[p]},{p},{1 if t==p else 0},{confs[i]:.6f}\n")

        total_seen = int(cm.sum())
        running_acc = (correct_total / max(1, total_seen))
        cpu_pct, ram_pct = get_cpu_ram()
        now = datetime.now().isoformat(timespec="seconds")
        metrics_rows.append({
            "timestamp": now,
            "batch_size": len(batch_frames),
            "clips_seen_total": total_seen,
            "running_accuracy": running_acc,
            "inference_ms": infer_ms,
            "inference_ms_per_clip": infer_ms / max(1, len(batch_frames)),
            "cpu_percent": cpu_pct,
            "ram_percent": ram_pct,
        })
        batch_frames.clear(); batch_labels.clear(); batch_paths.clear()
        return infer_ms, cpu_pct, ram_pct, running_acc

    # iterate
    pbar = tqdm(items, desc=f"Eval [{EVAL_SPLIT}]", total=len(items))
    for it in pbar:
        path = it["path"]
        cls_id = it["cls"]
        if cls_id not in class_to_idx:
            continue
        y = class_to_idx[cls_id]

        frames = sample_frames(path, NUM_FRAMES_DEFAULT, (H_target, W_target))  # we’ll re-adjust if fixed T differs
        batch_frames.append(frames)
        batch_labels.append(y)
        batch_paths.append(path)

        if len(batch_frames) >= INFER_BATCH_SIZE:
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

    # stats / outputs
    total = cm.sum()
    overall_acc = (correct_total / total) if total > 0 else 0.0
    per_class_acc = (cm.diagonal() / np.maximum(1, per_class_counts)).tolist()
    elapsed_s = perf_counter() - start_wall

    np.savetxt(cm_csv_path, cm, fmt="%d", delimiter=",")

    summary = {
        "model_path": TFLITE_MODEL_PATH,
        "data_dir": DATA_DIR,
        "split": EVAL_SPLIT,
        "num_classes": num_classes,
        "num_videos": int(total),
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "classes": classes,
        "cm_csv": str(cm_csv_path),
        "predictions_csv": str(csv_path),
        "wall_time_sec": float(elapsed_s),
        "avg_inference_ms_per_batch": float(np.mean([r["inference_ms"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_inference_ms_per_clip": float(np.mean([r["inference_ms_per_clip"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_cpu_percent": float(np.mean([r["cpu_percent"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_ram_percent": float(np.mean([r["ram_percent"] for r in metrics_rows])) if metrics_rows else 0.0,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if HAS_MPL:
        try:
            fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Confusion Matrix ({EVAL_SPLIT}) — Acc {overall_acc*100:.2f}%")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            tick_step = max(1, len(classes)//16) if len(classes) > 0 else 1
            ticks = list(range(0, len(classes), tick_step))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels([classes[i] for i in ticks], rotation=90, fontsize=7)
            ax.set_yticklabels([classes[i] for i in ticks], fontsize=7)
            plt.tight_layout(); fig.savefig(cm_png_path); plt.close(fig)
        except Exception as e:
            print(f"[warn] matplotlib plot failed: {e}")

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

    print(f"\nDone. Overall accuracy: {overall_acc*100:.2f}%")
    print(f"Predictions CSV: {csv_path}")
    print(f"Confusion matrix CSV: {cm_csv_path}")
    print(f"Summary JSON: {summary_json}")
    if Path(cm_png_path).exists():
        print(f"Confusion matrix PNG: {cm_png_path}")
    if Path(xlsx_path).exists():
        print(f"Metrics Excel: {xlsx_path}")

if __name__ == "__main__":
    evaluate()
