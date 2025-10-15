# batch_eval.py
# -----------------------------------------------------------
# Batch evaluation for LSA64 "cut" dataset with live metrics.
# - Recursively scans DATA_DIR for NNN_NNN_NNN.* videos.
# - Shows running accuracy, inference ms, CPU%, RAM% in tqdm.
# - Saves predictions CSV, summary JSON, confusion matrix CSV/PNG.
# - NEW: Saves per-batch metrics + summary to Excel.
# -----------------------------------------------------------

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # prevent Qt from needing X
os.environ.setdefault("MPLBACKEND", "Agg")   
import re
import json
import math
import time
from time import perf_counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# --- NEW: metrics deps ---
try:
    import psutil
except ImportError as e:
    raise SystemExit("psutil is required. Install with: pip install psutil") from e

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e

# Try optional readers
HAS_DECORD = False
try:
    import decord
    from decord import VideoReader, cpu
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

HAS_CV2 = False
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from torchvision.io import VideoReader as TVVideoReader
    HAS_TVVR = True
except Exception:
    HAS_TVVR = False
    
try:
    import matplotlib
    matplotlib.use("Agg")  # ensure non-interactive backend
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"[warn] matplotlib plot failed: {e}")

# =========================
# ====== CONFIG AREA ======
# =========================

# Checkpoint (.pt) and optionally its meta json (leave None to auto-find)
MODEL_PATH = r"../models/dynamic.pt"
META_JSON_PATH: Optional[str] = r"../models/dynamic.json"  # e.g., r"models\lsa64_fast_YYYYMMDD_HHMMSS_meta.json"

# Dataset root directory; videos may be in nested subfolders
DATA_DIR = r"../test_vid"

# Which split to evaluate: "val", "test", or "all"
EVAL_SPLIT = "all"

# Cap the number of videos evaluated (None for all)
MAX_VIDEOS: Optional[int] = 1280

# Inference clip spec (match training)
NUM_FRAMES = 16
FRAME_SIZE = 160

# Inference batch size (clips per forward pass; keep small if VRAM is tight)
INFER_BATCH_SIZE = 4

# AMP/TF32 & channels-last for speed
USE_MIXED_PRECISION = True
ENABLE_TF32 = True
USE_CHANNELS_LAST = True

# Output folder (timestamped subfolder will be created)
OUTPUT_DIR = "eval_results_dynamic"

# =========================
# ===== END OF CONFIG =====
# =========================

FNAME_RE = re.compile(r"^(?P<cls>\d{3})_(?P<actor>\d{3})_(?P<rep>\d{3})\.(?:mp4|avi|mov|mkv)$", re.IGNORECASE)
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def set_fast_paths():
    if ENABLE_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    if HAS_CV2:
        try:
            cv2.setNumThreads(0)  # avoid oversubscription
        except Exception:
            pass

# ---------- Model (same as training/infer) ----------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))  # [L,1,D]
    def forward(self, x):  # x: [L,B,D]
        return x + self.pe[:x.size(0)]

class PlainEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, norm_first, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation=activation, batch_first=False, norm_first=norm_first
            ) for _ in range(num_layers)
        ])
    def forward(self, src):
        out = src
        for layer in self.layers: out = layer(out)
        return out

class VideoTransformerClassifier(nn.Module):
    def __init__(self, num_classes: int,
                 d: int = 256, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, use_cls: bool = True, use_channels_last: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # -> [B,512,1,1]
        self.proj = nn.Linear(512, d)
        self.use_cls = use_cls
        self.use_channels_last = use_channels_last
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos = PositionalEncoding(d, max_len=2000)
        self.enc = PlainEncoder(d_model=d, nhead=n_heads, dim_feedforward=d*4,
                                dropout=dropout, activation="gelu", norm_first=True,
                                num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, num_classes)

    def forward(self, video: torch.Tensor):  # video: [B,T,3,H,W]
        B, T, C, H, W = video.shape
        x = video.view(B*T, C, H, W)              # [B*T,3,H,W]
        if self.use_channels_last and x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        feat = self.cnn(x).flatten(1)             # [B*T,512]
        feat = self.proj(feat).view(B, T, -1)     # [B,T,d]
        seq = feat.transpose(0, 1)                # [T,B,d]
        if self.use_cls:
            cls = self.cls_token.expand(-1, B, -1)
            seq = torch.cat([cls, seq], dim=0)    # [T+1,B,d]
        seq = self.pos(seq)
        out = self.enc(seq)                       # [T(+1),B,d]
        pooled = out[0] if self.use_cls else out.mean(dim=0)
        return self.head(self.drop(pooled))       # [B,num_classes]

# ---------- I/O helpers ----------

def load_meta(model_path: str, meta_override: Optional[str]) -> dict:
    if meta_override and Path(meta_override).exists():
        with open(meta_override, "r") as f: return json.load(f)
    # pick the newest *_meta.json in same folder
    cand = sorted(Path(model_path).parent.glob("*_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand: raise FileNotFoundError("No *_meta.json found; set META_JSON_PATH.")
    with open(cand[0], "r") as f: return json.load(f)

def build_model_from_ckpt(ckpt: dict, num_classes: int) -> VideoTransformerClassifier:
    # accept both old and new meta formats
    cfg_root = ckpt.get("config", {})
    cfg = cfg_root.get("MODEL", {}) if "MODEL" in cfg_root else cfg_root
    d = cfg.get("TRANSFORMER_DIM", 256)
    n_layers = cfg.get("TRANSFORMER_LAYERS", 2)
    n_heads = cfg.get("TRANSFORMER_HEADS", 4)
    dropout = cfg.get("TRANSFORMER_DROPOUT", 0.1)
    use_cls = cfg.get("USE_CLS_TOKEN", True)
    m = VideoTransformerClassifier(num_classes, d=d, n_layers=n_layers, n_heads=n_heads,
                                   dropout=dropout, use_cls=use_cls, use_channels_last=USE_CHANNELS_LAST)
    m.load_state_dict(ckpt["model_state"], strict=True)
    return m

def scan_cut_folder(data_dir: str) -> List[Dict]:
    """
    Recursively scan DATA_DIR for files whose *filenames* match
    'XXX_YYY_ZZZ.ext' (ext in VIDEO_EXTS). Subfolder structure is ignored.
    """
    root = Path(data_dir)
    items: List[Dict] = []
    if not root.exists():
        return items

    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        m = FNAME_RE.match(f.name)  # match on filename only
        if not m:
            continue
        items.append({
            "path": str(f.resolve()),
            "cls": m.group("cls"),
            "actor": m.group("actor"),
            "rep": m.group("rep"),
        })
    return items

def uniform_indices(n_total: int, n_samples: int):
    if n_total <= 0: return [0]*n_samples
    if n_samples <= 1: return [min(n_total-1, 0)]
    return [min(n_total-1, int(round(x))) for x in torch.linspace(0, n_total-1, steps=n_samples).tolist()]

# Readers (same priority as fast training)
def sample_frames_decord(path: str, num_frames: int) -> torch.Tensor:
    vr = VideoReader(path, ctx=cpu(0))
    n_total = len(vr)
    idxs = uniform_indices(n_total, num_frames)
    frames = vr.get_batch(idxs).asnumpy()
    return torch.from_numpy(frames)  # [T,H,W,3] uint8

def sample_frames_cv2_seq(path: str, num_frames: int) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n_total <= 0:
        cap.release()
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    idxs = set(uniform_indices(n_total, num_frames))
    frames = []
    i = 0
    ok = True
    while ok and len(frames) < num_frames:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None: break
        if i in idxs:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(frame_rgb))
        i += 1
    cap.release()
    if len(frames) == 0:
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    return torch.stack(frames, dim=0)

def sample_frames_tvvideoreader(path: str, num_frames: int) -> torch.Tensor:
    if not HAS_TVVR:
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    try:
        vr = TVVideoReader(path, "video")
    except Exception:
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    seq = []
    for frame in vr:
        img = frame["data"].permute(1,2,0).contiguous()  # [H,W,C]
        seq.append(img)
    if len(seq) == 0:
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    take = uniform_indices(len(seq), num_frames)
    frames = []
    for k in take:
        img = seq[k]
        img = torch.nn.functional.interpolate(
            img.permute(2,0,1).unsqueeze(0).float(),
            size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False
        ).squeeze(0).permute(1,2,0).byte()
        frames.append(img)
    return torch.stack(frames, dim=0)

def sample_frames(path: str, num_frames: int) -> torch.Tensor:
    if HAS_DECORD:
        return sample_frames_decord(path, num_frames)
    if HAS_CV2:
        return sample_frames_cv2_seq(path, num_frames)
    return sample_frames_tvvideoreader(path, num_frames)

# transforms
class VideoToTensor:
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype != torch.float32:
            frames = frames.float() / 255.0
        return frames.permute(0,3,1,2)  # [T,3,H,W]

class ApplyPerFrame:
    def __init__(self, tfm): self.tfm = tfm
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.tfm(xi) for xi in x], dim=0)

def build_eval_transform():
    img = [
        transforms.Resize(FRAME_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(FRAME_SIZE),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ]
    return transforms.Compose([VideoToTensor(), ApplyPerFrame(transforms.Compose(img))])

# ---------- Evaluation ----------

def choose_items_by_split(all_items: List[Dict], meta: dict, split: str) -> List[Dict]:
    if split == "all": return list(all_items)
    actors = meta.get("actors", {})
    target = actors.get(split, [])
    aset = set(target)
    return [x for x in all_items if x["actor"] in aset]

def build_class_index(meta: dict, ckpt: dict) -> Tuple[List[str], Dict[str,int]]:
    classes = meta.get("classes", []) or ckpt.get("classes", [])
    if not classes:
        raise RuntimeError("Could not recover class list from meta or checkpoint.")
    class_to_idx = {c:i for i,c in enumerate(classes)}
    return classes, class_to_idx

def evaluate():
    set_fast_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load ckpt + meta
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    meta = load_meta(MODEL_PATH, META_JSON_PATH)

    classes, class_to_idx = build_class_index(meta, ckpt)
    num_classes = len(classes)

    model = build_model_from_ckpt(ckpt, num_classes).to(device).eval()
    if USE_CHANNELS_LAST and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    tfm = build_eval_transform()

    # Scan and filter items (recursive)
    items_all = scan_cut_folder(DATA_DIR)
    items = choose_items_by_split(items_all, meta, EVAL_SPLIT)
    if MAX_VIDEOS is not None:
        items = items[:MAX_VIDEOS]
    assert items, f"No videos to evaluate in split={EVAL_SPLIT}."

    # Prepare outputs
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(OUTPUT_DIR) / f"eval_{EVAL_SPLIT}_pt_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    cm_csv_path = out_dir / "confusion_matrix.csv"
    cm_png_path = out_dir / "confusion_matrix.png"
    summary_json = out_dir / "summary.json"
    xlsx_path = out_dir / "metrics.xlsx"  # NEW

    # Confusion matrix + per-class
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_class_counts = np.zeros(num_classes, dtype=np.int64)
    correct_total = 0

    # Batch buffer
    batch_clips, batch_labels, batch_paths = [], [], []

    # --- NEW: metrics collectors ---
    proc = psutil.Process(os.getpid())
    metrics_rows = []  # list of dicts per batch
    start_wall = perf_counter()

    def get_cpu_ram():
        # System CPU% (instantaneous best-effort) and system RAM%
        cpu_pct = psutil.cpu_percent(interval=0.0)
        ram_pct = psutil.virtual_memory().percent
        return float(cpu_pct), float(ram_pct)

    def flush_batch():
        nonlocal correct_total, cm, per_class_counts, metrics_rows
        if not batch_clips: return

        # Build batch tensor on device
        clips = torch.stack(batch_clips, dim=0).to(device, non_blocking=True)  # [B,T,3,H,W]
        B = clips.size(0)

        # --- time the forward pass (includes softmax/argmax below) ---
        t0 = perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_MIXED_PRECISION and device.type=="cuda"):
            logits = model(clips)  # [B,C]
            probs = torch.softmax(logits.float(), dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            confs = probs.max(dim=1).values.cpu().numpy()
        t1 = perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        # write lines & update cm
        with csv_path.open("a", encoding="utf-8") as f:
            for i in range(len(preds)):
                t = int(batch_labels[i])
                p = int(preds[i])
                cm[t, p] += 1
                per_class_counts[t] += 1
                correct_total += int(t == p)
                path = batch_paths[i]
                cls_true = classes[t]
                cls_pred = classes[p]
                f.write(f"{path},{cls_true},{t},{cls_pred},{p},{1 if t==p else 0},{confs[i]:.6f}\n")

        # --- collect metrics after updating accuracy ---
        total_seen = int(cm.sum())
        running_acc = (correct_total / max(1, total_seen))
        cpu_pct, ram_pct = get_cpu_ram()
        now = datetime.now().isoformat(timespec="seconds")

        metrics_rows.append({
            "timestamp": now,
            "batch_size": int(B),
            "clips_seen_total": total_seen,
            "running_accuracy": running_acc,
            "inference_ms": infer_ms,
            "inference_ms_per_clip": infer_ms / max(1, B),
            "cpu_percent": cpu_pct,
            "ram_percent": ram_pct,
        })

        # clear buffers
        batch_clips.clear(); batch_labels.clear(); batch_paths.clear()

        return infer_ms, cpu_pct, ram_pct, running_acc

    # Write CSV header
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("path,true_cls,true_idx,pred_cls,pred_idx,correct,prob_top1\n")

    last_ms = 0.0
    last_cpu = 0.0
    last_ram = 0.0
    last_acc = 0.0

    # Iterate videos
    pbar = tqdm(items, desc=f"Eval [{EVAL_SPLIT}]", total=len(items))
    for it in pbar:
        path = it["path"]
        cls_id = it["cls"]
        if cls_id not in class_to_idx:
            # skip unexpected class id
            continue
        y = class_to_idx[cls_id]

        # load -> sample -> transform
        frames = sample_frames(path, NUM_FRAMES)       # [T,H,W,3] uint8
        clip = tfm(frames)                             # [T,3,S,S]
        batch_clips.append(clip)
        batch_labels.append(y)
        batch_paths.append(path)

        if len(batch_clips) >= INFER_BATCH_SIZE:
            out = flush_batch()
            if out is not None:
                last_ms, last_cpu, last_ram, last_acc = out
            pbar.set_postfix({
                "acc": f"{last_acc*100:.2f}%",
                "ms/batch": f"{last_ms:.1f}",
                "cpu%": f"{last_cpu:.0f}",
                "ram%": f"{last_ram:.0f}",
            })

    # flush last chunk
    out = flush_batch()
    if out is not None:
        last_ms, last_cpu, last_ram, last_acc = out

    # Compute stats
    total = cm.sum()
    overall_acc = (correct_total / total) if total > 0 else 0.0
    per_class_acc = (cm.diagonal() / np.maximum(1, per_class_counts)).tolist()
    elapsed_s = perf_counter() - start_wall

    # Save confusion matrix CSV
    np.savetxt(cm_csv_path, cm, fmt="%d", delimiter=",")

    # Save summary JSON
    summary = {
        "model_path": MODEL_PATH,
        "data_dir": DATA_DIR,
        "split": EVAL_SPLIT,
        "num_classes": num_classes,
        "num_videos": int(total),
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "classes": classes,
        "cm_csv": str(cm_csv_path),
        "predictions_csv": str(csv_path),
        "wall_time_sec": elapsed_s,
        # NEW rollups from per-batch metrics
        "avg_inference_ms_per_batch": float(np.mean([r["inference_ms"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_inference_ms_per_clip": float(np.mean([r["inference_ms_per_clip"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_cpu_percent": float(np.mean([r["cpu_percent"] for r in metrics_rows])) if metrics_rows else 0.0,
        "avg_ram_percent": float(np.mean([r["ram_percent"] for r in metrics_rows])) if metrics_rows else 0.0,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save confusion matrix PNG (labels optional)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Confusion Matrix ({EVAL_SPLIT}) — Acc {overall_acc*100:.2f}%")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        tick_step = max(1, len(classes)//16)  # downsample ticks for readability
        ticks = list(range(0, len(classes), tick_step))
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([classes[i] for i in ticks], rotation=90, fontsize=7)
        ax.set_yticklabels([classes[i] for i in ticks], fontsize=7)
        plt.tight_layout()
        fig.savefig(cm_png_path)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] matplotlib plot failed: {e}")

    # --- NEW: write Excel with per-batch metrics + summary ---
    try:
        df_batches = pd.DataFrame(metrics_rows)
        df_summary = pd.DataFrame([summary])

        # Prefer xlsxwriter if available; fall back to openpyxl
        engine = None
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
    if (Path(cm_png_path).exists()):
        print(f"Confusion matrix PNG: {cm_png_path}")
    if (Path(xlsx_path).exists()):
        print(f"Metrics Excel: {xlsx_path}")

if __name__ == "__main__":
    evaluate()
