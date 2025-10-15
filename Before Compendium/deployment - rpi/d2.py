#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# webcam_infer2.py
# -----------------------------------------------------------
# Live webcam inference for LSA64 (cut) model trained with
# VideoTransformerClassifier (ResNet18 + Transformer).
# - Set MODEL_PATH (and optionally META_JSON_PATH) below.
# - Press 'q' to quit the window.
# - Now shows a performance HUD matching the top script:
#   FPS, CPU%, Temp, RAM MB(%), with cached/smoothed reads.
# -----------------------------------------------------------

import os
import json
import time
import subprocess
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import psutil
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

# =========================
# ====== CONFIG AREA ======
# =========================

# Point this to the .pt checkpoint you want to test
MODEL_PATH = "models/dynamic.pt"

# If None, we try to auto-locate the sibling *_meta.json produced during training.
META_JSON_PATH: Optional[str] = "models/dynamic.json"  # e.g., r"dynamic.json"

# Webcam device index
CAM_INDEX = 0

# Inference window
NUM_FRAMES = 8       # should match training (e.g., 16)
FRAME_SIZE = 112      # should match training (e.g., 144)
STRIDE = 3            # push every STRIDE frames (overlap > 0 == smoother)

# Smoothing of predictions
TOPK = 5
EMA_ALPHA = 0.6       # 0=no smoothing, ->1=very smooth
VOTE_HISTORY = 8      # majority vote among last N top1 labels

# Performance knobs
USE_MIXED_PRECISION = False
USE_CUDA = True
CHANNELS_LAST = False
SHOW_FPS_AVG_OVER = 30  # frames for FPS averaging (separate from HUD calc)

# Overlay
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICK = 2
COLOR_MAIN = (0, 200, 0)   # BGR green
COLOR_SUB = (255, 255, 255)

# =========================
# ===== HUD HELPERS =======
# (copied to match your top script’s behavior & style)
# =========================

_last_stat_t, _last_stats = 0.0, (0.0, 0.0, None)  # (used_mb, ram%, tempC)

def _get_system_stats():
    vm = psutil.virtual_memory()
    used_mb = (vm.total - vm.available) / (1024 * 1024)
    return used_mb, vm.percent, _get_cpu_temp_c()

def _get_system_stats_cached():
    """Returns (used_ram_mb, ram_percent, temp_c), refreshed ~2.5/s."""
    global _last_stat_t, _last_stats
    now = time.time()
    if now - _last_stat_t > 0.4:
        _last_stats = _get_system_stats()
        _last_stat_t = now
    return _last_stats

def _get_cpu_temp_c():
    # psutil path (if supported)
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
        for name, entries in (temps or {}).items():
            for e in entries:
                label = (e.label or name or "").lower()
                if any(k in label for k in ("cpu", "soc", "package", "core", "bcm", "gpu")) and (e.current and e.current > 0):
                    return float(e.current)
        # fallback to any valid temp if above didn't match
        for entries in (temps or {}).values():
            for e in entries:
                if e.current and e.current > 0:
                    return float(e.current)
    except Exception:
        pass

    # common linux sysfs paths
    for p in ("/sys/class/thermal/thermal_zone0/temp", "/sys/class/hwmon/hwmon0/temp1_input"):
        try:
            with open(p, "r") as f:
                v = f.read().strip()
            if v:
                t = float(v)
                return t / 1000.0 if t > 200 else t
        except Exception:
            pass

    # Raspberry Pi vcgencmd
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True).strip()
        if "temp=" in out:
            return float(out.split("temp=")[1].split("'")[0])
    except Exception:
        pass

    return None

# --- CPU % cache (stable like top script) ---
_cpu_inited = False
_last_cpu_t = 0.0
_last_cpu_pct = 0.0

def _get_cpu_percent_cached(period=0.5, alpha=0.4):
    """
    Return a smoothed CPU % updated at most every `period` seconds.
    Uses psutil.cpu_percent(interval=None) and an EMA for stability.
    """
    global _cpu_inited, _last_cpu_t, _last_cpu_pct
    now = time.time()

    if not _cpu_inited:
        psutil.cpu_percent(interval=None)  # prime
        _cpu_inited = True
        _last_cpu_t = now
        return _last_cpu_pct  # 0.0 for the very first call

    if now - _last_cpu_t >= period:
        new = psutil.cpu_percent(interval=None)
        _last_cpu_pct = (1.0 - alpha) * _last_cpu_pct + alpha * new
        _last_cpu_t = now

    return _last_cpu_pct
    
def draw_performance_hud(frame, fps):
    h, w = frame.shape[:2]
    used_mb, ram_percent, temp_c = _get_system_stats_cached()
    cpu_percent = _get_cpu_percent_cached()

    temp_text = f"{temp_c:.1f}°C" if temp_c is not None else "--"
    lines = [
        f"FPS: {fps:.1f}",
        f"CPU: {cpu_percent:.0f}%",
        f"Temp: {temp_text}",
        f"RAM: {used_mb:.0f} MB ({ram_percent:.0f}%)",
    ]
    y = 30
    for line in lines:
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = w - text_w - 15
        color = (0,255,255)
        if "CPU" in line: color = (0,165,255)
        elif "RAM" in line: color = (255,255,0)
        cv2.rectangle(frame, (x-5, y-text_h-5), (x+text_w+5, y+5), (0,0,0), -1)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += text_h + 8

def put_text_bg(img, text, org, color, scale=1.0, thickness=2):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - baseline - 4), (x + w + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_performance_hud(frame, fps):
    """
    Draws FPS, CPU %, CPU temperature, and RAM usage in top-right corner.
    Matches your top script's style and update cadence.
    """
    h, w = frame.shape[:2]

    used_mb, ram_percent, temp_c = _get_system_stats_cached()
    cpu_percent = _get_cpu_percent_cached()

    lines = [
        f"FPS: {fps:.1f}",
        f"CPU: {cpu_percent:.0f}%",
        f"RAM: {used_mb:.0f} MB ({ram_percent:.0f}%)"
    ]
    if temp_c is not None:
        lines.insert(2, f"Temp: {temp_c:.1f}°C")

    y = 30
    for line in lines:
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = w - text_w - 15  # right aligned

        color = (0, 255, 255)   # default yellow
        if "CPU" in line:
            color = (0, 165, 255)     # orange
        elif "RAM" in line:
            color = (255, 255, 0)     # cyan

        cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += text_h + 8

# =========================
# ===== MODEL DEFNS =======
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
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
        x = video.view(B * T, C, H, W)                       # -> [B*T,3,H,W] (4D)
        if self.use_channels_last and x.is_cuda:             # safe: only 4D gets channels_last
            x = x.contiguous(memory_format=torch.channels_last)
        feat = self.cnn(x).flatten(1)                        # [B*T,512]
        feat = self.proj(feat).view(B, T, -1)                # [B,T,d]
        seq = feat.transpose(0, 1)                           # [T,B,d]
        if self.use_cls:
            cls = self.cls_token.expand(-1, B, -1)
            seq = torch.cat([cls, seq], dim=0)               # [T+1,B,d]
        seq = self.pos(seq)
        out = self.enc(seq)                                  # [T(+1),B,d]
        pooled = out[0] if self.use_cls else out.mean(dim=0)
        return self.head(self.drop(pooled))                  # [B,num_classes]

# =========================
# ====== UTILITIES ========
# =========================

def load_meta_for_model(model_path: str, meta_path_override: Optional[str]) -> dict:
    if meta_path_override and Path(meta_path_override).exists():
        with open(meta_path_override, "r") as f:
            return json.load(f)

    mp = Path(model_path)
    parent = mp.parent
    meta_candidates = list(parent.glob("*_meta.json"))
    if not meta_candidates:
        raise FileNotFoundError("No *_meta.json found. Please set META_JSON_PATH.")
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
        use_channels_last=True
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

    # Inference preprocess (mirror training exactly):
    per_frame_transform = transforms.Compose([
        transforms.Resize(FRAME_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(FRAME_SIZE),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Frame buffer (sliding window) + smoothing
    frame_buf: deque = deque(maxlen=NUM_FRAMES)
    ema_logits: Optional[np.ndarray] = None   # [num_classes]
    top1_hist: deque = deque(maxlen=VOTE_HISTORY)

    # Capture
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {CAM_INDEX}")

    # FPS averaging for your existing overlay (left side)
    times = deque(maxlen=SHOW_FPS_AVG_OVER)
    last_infer_time = 0.0

    # HUD FPS timing (match top script’s rolling method)
    hud_times = deque(maxlen=30)
    hud_times.append(time.perf_counter())

    print("Press 'q' to quit.")
    while True:
        t0 = time.time()
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        # Preprocess for buffer: BGR->RGB, keep native size (no stretching here)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_buf.append(rgb)

        do_infer = (len(frame_buf) == NUM_FRAMES) and (len(times) % STRIDE == 0)
        pred_label = None
        pred_prob = None

        if do_infer:
            # Apply transforms PER FRAME first, then stack
            frames_list: List[torch.Tensor] = []
            for f in frame_buf:
                img = cv2.resize(f, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
                arr = img.astype(np.float32) / 255.0
                # normalize in-place
                arr[..., 0] = (arr[..., 0] - 0.485) / 0.229
                arr[..., 1] = (arr[..., 1] - 0.456) / 0.224
                arr[..., 2] = (arr[..., 2] - 0.406) / 0.225
                # to CHW tensor
                ft = torch.from_numpy(arr).permute(2, 0, 1)
                frames_list.append(ft)

            frames_t = torch.stack(frames_list, dim=0)  # [T,3,H,W]
            clip = frames_t.unsqueeze(0).to(device, non_blocking=True)  # [1,T,3,H,W]

            with torch.no_grad(), autocast(device_type="cuda", enabled=USE_MIXED_PRECISION if device.type == "cuda" else False):
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

            last_infer_time = time.time() - t0

        # FPS calc for your existing overlay (left side)
        t1 = time.time()
        times.append(t1)
        fps_left = 0.0
        if len(times) >= 2:
            fps_left = (len(times) - 1) / (times[-1] - times[0] + 1e-9)

        # HUD FPS calc (match top script’s style)
        hud_now = time.perf_counter()
        hud_times.append(hud_now)
        fps_hud = (len(hud_times) - 1) / (hud_times[-1] - hud_times[0] + 1e-9) if len(hud_times) >= 2 else 0.0

        # Compose display
        disp = frame_bgr.copy()

        # Your existing (left) overlay
        y = 24
        if last_infer_time:
            put_text(disp, f"Infer time: {last_infer_time*1000:.1f} ms", (10, y), COLOR_SUB); y += 24

        if pred_label is not None:
            put_text(disp, f"Top-1: {pred_label}  ({pred_prob*100:.1f}%)", (10, y), COLOR_MAIN, scale=0.8); y += 28

            # Also show top-k
            probs = softmax_np(ema_logits)
            topk_idx = np.argsort(-probs)[:TOPK]
            for k in range(len(topk_idx)):
                lbl = classes[topk_idx[k]] if 0 <= topk_idx[k] < len(classes) else str(topk_idx[k])
                put_text(disp, f"{k+1}. {lbl:<3}  {probs[topk_idx[k]]*100:5.1f}%", (10, y), COLOR_SUB); y += 22

        # Performance HUD (top-right) — identical look & cadence to top script
        draw_performance_hud(disp, fps_hud)

        cv2.imshow("LSA64 Live Inference (press q to quit)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
