#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
webcam_infer_tflite.py
Live webcam inference using a TFLite LSA64 (cut) model.
- Robust handling of dynamic input shapes (T can be -1)
- Single Interpreter (no destructor spam)
- Optional hotfix for tflite_runtime Delegate.__del__ bug
- Keeps your HUD, EMA smoothing, and top-k readout
"""

import os, json, time, subprocess, gc
from collections import deque
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
import psutil

# ---- Try tflite_runtime first; fallback to TF Lite ----
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite_runtime"
except Exception:
    from tensorflow.lite import Interpreter as TFInterpreter  # type: ignore
    tflite = None
    TFLITE_BACKEND = "tensorflow"

# -------------------------
# Small hotfix for the known tflite_runtime delegate __del__ error
# (Harmless warning; this just silences it.)
# -------------------------
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
# ====== CONFIG AREA ======
# =========================
MODEL_TFLITE = "models/dynamic.tflite"
META_JSON_PATH: Optional[str] = "models/dynamic.json"

CAM_INDEX   = 0
FRAME_SIZE  = 112
NUM_FRAMES  = 8          # fallback if model uses dynamic T (-1)
STRIDE      = 2

TOPK        = 5
EMA_ALPHA   = 0.6
VOTE_HISTORY= 8
SHOW_FPS_AVG_OVER = 30

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICK      = 2
COLOR_MAIN = (0, 200, 0)   # BGR green
COLOR_SUB  = (255, 255, 255)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# =========================
# ===== HUD HELPERS =======
# =========================
_last_stat_t, _last_stats = 0.0, (0.0, 0.0, None)  # (used_mb, ram%, tempC)
_cpu_inited, _last_cpu_t, _last_cpu_pct = False, 0.0, 0.0

def _get_system_stats():
    vm = psutil.virtual_memory()
    used_mb = (vm.total - vm.available) / (1024 * 1024)
    return used_mb, vm.percent, _get_cpu_temp_c()

def _get_system_stats_cached():
    global _last_stat_t, _last_stats
    now = time.time()
    if now - _last_stat_t > 0.4:
        _last_stats = _get_system_stats()
        _last_stat_t = now
    return _last_stats

def _get_cpu_temp_c():
    # psutil route
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
    # sysfs fallback
    for p in ("/sys/class/thermal/thermal_zone0/temp", "/sys/class/hwmon/hwmon0/temp1_input"):
        try:
            v = float(open(p).read().strip())
            return v/1000.0 if v > 200 else v
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

def put_text(img, text, org, color=COLOR_SUB, scale=FONT_SCALE, thick=THICK):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)

def draw_performance_hud(frame, fps):
    h, w = frame.shape[:2]
    used_mb, ram_percent, temp_c = _get_system_stats_cached()
    cpu_percent = _get_cpu_percent_cached()
    lines = [f"FPS: {fps:.1f}", f"CPU: {cpu_percent:.0f}%", f"RAM: {used_mb:.0f} MB ({ram_percent:.0f}%)"]
    if temp_c is not None: lines.insert(2, f"Temp: {temp_c:.1f}°C")
    y = 30
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, FONT, 0.6, 2)
        x = w - tw - 15
        color = (0,255,255)
        if "CPU" in line: color=(0,165,255)
        elif "RAM" in line: color=(255,255,0)
        cv2.rectangle(frame, (x-5, y-th-5), (x+tw+5, y+5), (0,0,0), -1)
        cv2.putText(frame, line, (x, y), FONT, 0.6, color, 2)
        y += th + 8

# =========================
# ======= UTILITIES =======
# =========================
def softmax_np(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def preprocess_frame(frame_bgr, size=FRAME_SIZE):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1))  # [3,H,W]

def load_classes(model_path: str) -> List[str]:
    if META_JSON_PATH and Path(META_JSON_PATH).exists():
        return (json.load(open(META_JSON_PATH)) or {}).get("classes", [])
    mp = Path(model_path)
    cands = list(mp.parent.glob("*_meta.json"))
    if cands:
        return (json.load(open(max(cands, key=lambda p: p.stat().st_mtime))) or {}).get("classes", [])
    raise FileNotFoundError("No classes found. Provide META_JSON_PATH or place a *_meta.json next to the model.")

def get_model_io_shape(interpreter) -> np.ndarray:
    """Return a plain Python-list shape, preferring shape_signature."""
    inp = interpreter.get_input_details()[0]
    sig = inp.get("shape_signature", None)
    shp = inp.get("shape", None)
    arr = np.array(sig if (isinstance(sig, np.ndarray) and sig.size>0) else shp).astype(int)
    return arr.tolist()

# =========================
# ======= MAIN ============
# =========================
def main():
    assert Path(MODEL_TFLITE).exists(), f"MODEL_TFLITE not found: {MODEL_TFLITE}"
    classes = load_classes(MODEL_TFLITE)
    assert classes, "Class list is empty."

    # Build interpreter (let runtime pick XNNPACK if bundled)
    if tflite is not None:
        interpreter = tflite.Interpreter(  # type: ignore[attr-defined]
            model_path=MODEL_TFLITE,
            num_threads=max(1, (os.cpu_count() or 2) - 1),
        )
    else:
        interpreter = TFInterpreter(model_path=MODEL_TFLITE, num_threads=max(1, (os.cpu_count() or 2) - 1))  # type: ignore

    interpreter.allocate_tensors()

    # Resolve expected input shape
    shape = get_model_io_shape(interpreter)  # [N,C,H,W,T] (your model)
    assert len(shape) == 5, f"Expected 5D input, got {shape}"
    N, C, Hm, Wm, Tm = [int(x) for x in shape]  # coerce to Python int
    if Tm < 0:  # dynamic time dimension
        Tm = int(NUM_FRAMES)
    if Hm < 0 or Wm < 0:
        Hm = Wm = int(FRAME_SIZE)

    print(f"✅ Model loaded. Input: N={N},C={C},H=W={Hm},T={Tm}")

    inp_info  = interpreter.get_input_details()[0]
    out_info  = interpreter.get_output_details()[0]

    # Video capture
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {CAM_INDEX}")

    # Buffers and state
    frame_buf: deque = deque(maxlen=int(Tm))   # <-- FIX: make sure it's a real Python int
    ema_logits: Optional[np.ndarray] = None
    times = deque(maxlen=SHOW_FPS_AVG_OVER)
    hud_times = deque(maxlen=30); hud_times.append(time.perf_counter())
    frame_i = 0
    last_infer_ms = 0.0

    last_pred_label: Optional[str] = None
    last_pred_prob: float = 0.0
    last_topk: Optional[np.ndarray] = None
    last_pred_time: float = 0.0
    last_probs: Optional[np.ndarray] = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Build sliding window (preprocess once per frame)
        frame_buf.append(preprocess_frame(frame, size=Hm))
        frame_i += 1

        do_infer = (len(frame_buf) == Tm) and ((frame_i % STRIDE) == 0)
        if do_infer:
            # Build clip & run inference
            clip = np.stack(list(frame_buf), axis=0)[None, ...].astype(np.float32)
            clip = np.transpose(clip, (0, 2, 3, 4, 1))
            interpreter.set_tensor(inp_info["index"], clip)

            t0 = time.time()
            interpreter.invoke()
            last_infer_ms = (time.time() - t0) * 1000.0

            logits = interpreter.get_tensor(out_info["index"])[0]
            if ema_logits is None:
                ema_logits = logits
            else:
                ema_logits = EMA_ALPHA * ema_logits + (1.0 - EMA_ALPHA) * logits

            probs = softmax_np(ema_logits)
            top_idx = int(np.argmax(probs))

            # ✅ Save for display outside inference step
            last_pred_label = classes[top_idx]
            last_pred_prob = float(probs[top_idx])
            last_topk = np.argsort(-probs)[:TOPK]
            last_pred_time = last_infer_ms
            last_probs = probs.copy()  
            
            
        # HUD FPS (smoothed)
        hud_times.append(time.perf_counter())
        fps_hud = (len(hud_times) - 1) / (hud_times[-1] - hud_times[0] + 1e-9) if len(hud_times) >= 2 else 0.0


        # --- HUD DRAWING: use last stable values ---
        disp = frame.copy()
        y = 24

        # infer time
        if last_pred_time > 0:
            put_text(disp, f"Infer: {last_pred_time:.1f} ms", (10, y))
            y += 24

        # top-1 + top-k
        if last_pred_label is not None and last_probs is not None:
            put_text(disp, f"Top-1: {last_pred_label} ({last_pred_prob*100:.1f}%)",
                     (10, y), COLOR_MAIN, 0.8)
            y += 28
            if last_topk is not None:
                for k, idx in enumerate(last_topk):
                    put_text(disp, f"{k+1}. {classes[int(idx)]}  {last_probs[int(idx)]*100:5.1f}%", (10, y))
                    y += 22

        draw_performance_hud(disp, fps_hud)
        cv2.imshow("LSA64 Live Inference (TFLite)", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    del interpreter
    gc.collect()

if __name__ == "__main__":
    main()
