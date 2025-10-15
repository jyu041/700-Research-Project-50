# webcam_infer.py
# -----------------------------------------------------------
# Live webcam inference for LSA64 (cut) model trained with
# VideoTransformerClassifier (ResNet18 + Transformer).
# - Set MODEL_PATH (and optionally META_JSON_PATH) below.
# - No CLI args; tweak CONFIG AREA variables at the top.
# - Press 'q' to quit the window.
# -----------------------------------------------------------

import os
import json
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

# =========================
# ====== CONFIG AREA ======
# =========================

# Point this to the .pt checkpoint you want to test
MODEL_PATH = r"models/lsa64_fast_20250917_203048_best_ep8_valacc0.9469_20250917_224608.pt"

# If None, we try to auto-locate the sibling *_meta.json produced during training.
META_JSON_PATH: Optional[str] = None  # e.g., r"models\lsa64_fast_YYYYMMDD_HHMMSS_meta.json"

# Webcam device index
CAM_INDEX = 0

# Inference window
NUM_FRAMES = 16       # should match training (e.g., 16)
FRAME_SIZE = 144      # should match training (e.g., 144)
STRIDE = 2            # push every STRIDE frames (overlap > 0 == smoother)

# Smoothing of predictions
TOPK = 5
EMA_ALPHA = 0.6       # 0=no smoothing, ->1=very smooth
VOTE_HISTORY = 8      # majority vote among last N top1 labels

# Performance knobs
USE_MIXED_PRECISION = True
USE_CUDA = True
CHANNELS_LAST = True
SHOW_FPS_AVG_OVER = 30  # frames

# Overlay
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICK = 2
COLOR_MAIN = (0, 200, 0)   # BGR green
COLOR_SUB = (255, 255, 255)

# =========================
# ===== MODEL DEFNS =======
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(1))  # [L,1,D]

    def forward(self, x):  # x: [L,B,D]
        return x + self.pe[:x.size(0)]

class CheckpointedEncoder(nn.Module):
    """Same structure as training; checkpointing is disabled in inference anyway."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, activation: str, norm_first: bool, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation=activation, batch_first=False, norm_first=norm_first
            ) for _ in range(num_layers)
        ])

    def forward(self, src: torch.Tensor):
        out = src
        for layer in self.layers:
            out = layer(out)
        return out

class VideoTransformerClassifier(nn.Module):
    def __init__(self, num_classes: int,
                 d: int = 256, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1, use_cls: bool = True,
                 use_channels_last: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])   # -> [B,512,1,1]
        self.proj = nn.Linear(512, d)
        self.use_cls = use_cls
        self.use_channels_last = use_channels_last

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos = PositionalEncoding(d, max_len=2000)
        self.enc = CheckpointedEncoder(
            d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout,
            activation="gelu", norm_first=True, num_layers=n_layers
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, num_classes)

    def forward(self, video: torch.Tensor):
        # video: [B,T,3,H,W]
        B, T, C, H, W = video.shape
        x = video.view(B*T, C, H, W)                       # -> [B*T,3,H,W] (4D)
        if self.use_channels_last and x.is_cuda:           # safe: only 4D gets channels_last
            x = x.contiguous(memory_format=torch.channels_last)
        feat = self.cnn(x).flatten(1)                      # [B*T,512]
        feat = self.proj(feat).view(B, T, -1)              # [B,T,d]
        seq = feat.transpose(0,1)                          # [T,B,d]
        if self.use_cls:
            cls = self.cls_token.expand(-1, B, -1)
            seq = torch.cat([cls, seq], dim=0)             # [T+1,B,d]
        seq = self.pos(seq)
        out = self.enc(seq)                                # [T(+1),B,d]
        pooled = out[0] if self.use_cls else out.mean(dim=0)
        return self.head(self.drop(pooled))                # [B,num_classes]


# =========================
# ====== UTILITIES ========
# =========================

def load_meta_for_model(model_path: str, meta_path_override: Optional[str]) -> dict:
    if meta_path_override and Path(meta_path_override).exists():
        with open(meta_path_override, "r") as f:
            return json.load(f)

    # Try to infer: the training scripts used "<run_name>_meta.json"
    mp = Path(model_path)
    parent = mp.parent
    # find any *_meta.json that shares the same run prefix (lsa64_*_YYYYmmdd_HHMMSS)
    meta_candidates = list(parent.glob("*_meta.json"))
    if not meta_candidates:
        raise FileNotFoundError("No *_meta.json found. Please set META_JSON_PATH.")
    # If multiple, pick the newest file in the same folder
    pick = max(meta_candidates, key=lambda p: p.stat().st_mtime)
    with open(pick, "r") as f:
        return json.load(f)

def build_model_from_ckpt(ckpt: dict, num_classes: int) -> VideoTransformerClassifier:
    cfg = ckpt.get("config", {}).get("MODEL", {}) or ckpt.get("config", {})
    d = cfg.get("TRANSFORMER_DIM", 256) if "TRANSFORMER_DIM" in cfg else ckpt.get("config", {}).get("TRANSFORMER_DIM", 256)
    n_layers = cfg.get("TRANSFORMER_LAYERS", 2)
    n_heads = cfg.get("TRANSFORMER_HEADS", 4)
    dropout = cfg.get("TRANSFORMER_DROPOUT", 0.1)
    use_cls = cfg.get("USE_CLS_TOKEN", True)

    model = VideoTransformerClassifier(
        num_classes=num_classes,
        d=d, n_layers=n_layers, n_heads=n_heads, dropout=dropout, use_cls=use_cls,
        use_channels_last=True  # <- enables the safe 4D channels_last path
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model


def softmax_np(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def put_text(img, text, org, color=COLOR_SUB, scale=FONT_SCALE, thick=THICK):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)

# =========================
# ======= INFERENCE =======
# =========================

def main():
    assert Path(MODEL_PATH).exists(), f"MODEL_PATH not found: {MODEL_PATH}"

    # Device / AMP
    device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
    from torch.amp import autocast
    scaler = None  # not needed for eval

    # TF32 + channels_last optional boost on Ampere
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # Load checkpoint + meta
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    meta = load_meta_for_model(MODEL_PATH, META_JSON_PATH)

    classes: List[str] = meta.get("classes", [])
    if not classes and "classes" in ckpt:
        classes = ckpt["classes"]
    assert classes, "Could not recover classes list from meta or checkpoint."

    # Build model from checkpoint config
    num_classes = len(classes)
    model = build_model_from_ckpt(ckpt, num_classes).to(device).eval()
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)

    # Inference preprocess (match training)
    to_tensor = transforms.ToTensor()  # [H,W,C] -> [C,H,W] in [0,1]
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])

    # Frame buffer (sliding window) + smoothing
    frame_buf: deque = deque(maxlen=NUM_FRAMES)
    ema_logits: Optional[np.ndarray] = None   # [num_classes]
    top1_hist: deque = deque(maxlen=VOTE_HISTORY)

    # Capture
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {CAM_INDEX}")

    # FPS averaging
    times = deque(maxlen=SHOW_FPS_AVG_OVER)
    last_infer_time = 0.0

    print("Press 'q' to quit.")
    while True:
        t0 = time.time()
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        # Preprocess frame for the buffer: BGR->RGB, resize, keep uint8 for now
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        frame_buf.append(rgb)

        do_infer = (len(frame_buf) == NUM_FRAMES) and (len(times) % STRIDE == 0)
        pred_label = None
        pred_prob = None

        if do_infer:
            # Build [1,T,3,H,W] tensor
            # Convert to float tensor + normalize per frame
            frames = np.stack(list(frame_buf), axis=0)  # [T,H,W,3] uint8
            frames_t = torch.from_numpy(frames).float().div(255.0)  # [T,H,W,3]
            frames_t = frames_t.permute(0,3,1,2)  # [T,3,H,W]
            # Normalize per frame
            for t in range(frames_t.shape[0]):
                frames_t[t] = normalize(frames_t[t])

            # Add batch and channels_last memory format
            clip = frames_t.unsqueeze(0).to(device, non_blocking=True)  # [1,T,3,H,W]

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_MIXED_PRECISION if device.type=="cuda" else False):
                logits = model(clip)      # [1,num_classes]
            logits_np = logits.squeeze(0).float().cpu().numpy()

            # EMA smoothing on logits
            if ema_logits is None:
                ema_logits = logits_np
            else:
                ema_logits = EMA_ALPHA * ema_logits + (1.0 - EMA_ALPHA) * logits_np

            probs = softmax_np(ema_logits)
            top_idx = int(np.argmax(probs))
            pred_label = classes[top_idx] if 0 <= top_idx < num_classes else str(top_idx)
            pred_prob = float(probs[top_idx])

            # majority vote history (top1 id)
            top1_hist.append(top_idx)
            # Optional: you could use majority vote winner too
            # from collections import Counter; winner = Counter(top1_hist).most_common(1)[0][0]

            last_infer_time = time.time() - t0

        # FPS calc
        t1 = time.time()
        times.append(t1)
        fps = 0.0
        if len(times) >= 2:
            fps = (len(times)-1) / (times[-1] - times[0] + 1e-9)

        # Overlay
        disp = frame_bgr.copy()
        y = 24
        put_text(disp, f"FPS: {fps:5.1f}", (10, y), COLOR_SUB); y += 24
        if last_infer_time:
            put_text(disp, f"Infer time: {last_infer_time*1000:.1f} ms", (10, y), COLOR_SUB); y += 24

        if pred_label is not None:
            put_text(disp, f"Top-1: {pred_label}  ({pred_prob*100:.1f}%)", (10, y), COLOR_MAIN, scale=0.8); y += 28

            # Also show top-k
            # Recompute top-k from ema_logits to display
            probs = softmax_np(ema_logits)
            topk_idx = np.argsort(-probs)[:TOPK]
            for k in range(len(topk_idx)):
                lbl = classes[topk_idx[k]] if 0 <= topk_idx[k] < num_classes else str(topk_idx[k])
                put_text(disp, f"{k+1}. {lbl:<3}  {probs[topk_idx[k]]*100:5.1f}%", (10, y), COLOR_SUB); y += 22

        cv2.imshow("LSA64 Live Inference (press q to quit)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
