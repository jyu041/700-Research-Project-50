#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, time, argparse, numpy as np
from collections import deque
from pathlib import Path
import psutil, subprocess

# ----------------------- labels -----------------------
ASL_LABELS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','space'
]

# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 (PyTorch) + TFLite classifier for ASL on hand crop")
    # YOLO detector (.pt)
    p.add_argument("--det_model", type=str, default="models/best_hand.pt", help="Path to YOLOv8 .pt model")
    p.add_argument("--conf",      type=float, default=0.8, help="YOLO detection confidence")
    p.add_argument("--iou",       type=float, default=0.45, help="YOLO NMS IoU")
    p.add_argument("--imgsz",     type=int,   default=256,  help="YOLO inference size")

    # Classifier (.tflite)
    p.add_argument("--cls_model", type=str, default="models/model.tflite", help="Path to TFLite classifier")
    p.add_argument("--cls_thr",   type=float, default=0.60, help="Min confidence to display a label")
    p.add_argument("--cls_threads", type=int, default=0, help="TFLite interpreter threads (0=auto)")

    # Source / display
    p.add_argument("--source", type=str, default="0", help="0 for webcam or path to video")
    p.add_argument("--view", action="store_true", help="Show window")
    p.add_argument("--save", action="store_true", help="Save annotated video")
    p.add_argument("--save_dir", type=str, default="runs/yolo_pt_tflite_cls")
    p.add_argument("--smooth_k", type=int, default=5, help="SMA window for classifier probs")

    # Crop behavior
    p.add_argument("--square_pad", type=float, default=1, help="Expand bbox to square by this factor")

    return p.parse_args()

# ----------------------- UI helpers -----------------------
def put_text_bg(img, text, org, color=(255,255,255), scale=0.8, thickness=2):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x-4, y-h-baseline-4), (x+w+4, y+4), (0,0,0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def expand_square(x1, y1, x2, y2, img_w, img_h, pad=1.25):
    cx = 0.5*(x1 + x2); cy = 0.5*(y1 + y2)
    w  = max(1, x2 - x1); h  = max(1, y2 - y1)
    s  = pad * max(w, h)
    nx1 = int(max(0, cx - s/2)); ny1 = int(max(0, cy - s/2))
    nx2 = int(min(img_w-1, cx + s/2)); ny2 = int(min(img_h-1, cy + s/2))
    return nx1, ny1, nx2, ny2
# ----------------------- helpers --------------------------
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
    """
    h, w = frame.shape[:2]

    # --- Collect stats ---
    cpu_percent = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_mb = (vm.total - vm.available) / (1024 * 1024)
    ram_percent = vm.percent

    # Try to get CPU temperature (safe fallback if not available)
    temp_c = None
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
        if temps:
            for name, entries in temps.items():
                for e in entries:
                    if "cpu" in (e.label or name).lower():
                        temp_c = e.current
                        break
                if temp_c is not None:
                    break
    except Exception:
        pass

    # --- Text lines ---
    lines = [
        f"FPS: {fps:.1f}",
        f"CPU: {cpu_percent:.0f}%",
        f"RAM: {ram_mb:.0f} MB ({ram_percent:.0f}%)"
    ]
    if temp_c is not None:
        lines.insert(2, f"Temp: {temp_c:.1f}°C")

    # --- Draw top-right ---
    y = 30
    for line in lines:
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = w - text_w - 15  # align to right edge with 15px margin

        color = (0, 255, 255)  # default yellow
        if "CPU" in line:
            color = (0, 165, 255)  # orange
        elif "RAM" in line:
            color = (255, 255, 0)  # cyan

        cv2.rectangle(frame, (x-5, y-text_h-5), (x+text_w+5, y+5), (0,0,0), -1)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += text_h + 8

def _get_system_stats():
    vm = psutil.virtual_memory()
    used_mb = (vm.total - vm.available) / (1024*1024)
    return used_mb, vm.percent, _get_cpu_temp_c()

# ----------------------- TFLite classifier -----------------------
def _get_tflite_interpreter(model_path, threads):
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
            raise RuntimeError("No TFLite interpreter available. Install tflite_runtime or tensorflow.") from e
    threads = threads if (threads and threads > 0) else max(1, (os.cpu_count() or 4) - 1)
    itp = Runtime.Interpreter(model_path=model_path, num_threads=threads)
    itp.allocate_tensors()
    return itp

class TFLiteClassifier:
    def __init__(self, tflite_path, threads=0):
        self.interpreter = _get_tflite_interpreter(tflite_path, threads)
        self.in_details  = self.interpreter.get_input_details()[0]
        self.out_details = self.interpreter.get_output_details()[0]
        self.input_index = self.in_details['index']
        self.output_index= self.out_details['index']
        self.in_dtype    = self.in_details['dtype']
        self.out_dtype   = self.out_details['dtype']
        self.in_scale, self.in_zero = (self.in_details.get('quantization') or (0.0, 0))
        self.out_scale, self.out_zero = (self.out_details.get('quantization') or (0.0, 0))
        self.in_shape = self.in_details['shape']  # [1,H,W,C]
        _, self.in_h, self.in_w, self.in_c = self.in_shape

        # warm-up
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=self.in_dtype)
        self.interpreter.set_tensor(self.input_index, dummy)
        self.interpreter.invoke()

    @property
    def target_hw(self):
        return (int(self.in_w), int(self.in_h))  # (W,H) for cv2.resize

    def _quantize_in(self, x_float01):
        if np.issubdtype(self.in_dtype, np.integer):
            if self.in_scale and self.in_scale > 0:
                x = x_float01 / self.in_scale + self.in_zero
            else:
                x = x_float01 * 255.0
            x = np.clip(x, 0, 255).astype(self.in_dtype)
        else:
            x = x_float01.astype(self.in_dtype)
        return x

    def _dequantize_out(self, y):
        if np.issubdtype(y.dtype, np.integer) and self.out_scale:
            y = (y.astype(np.float32) - self.out_zero) * self.out_scale
        return y.astype(np.float32)

    def _softmax_if_needed(self, y):
        # If looks like probs in [0,1] summing ~1, return as-is; else softmax logits
        s = float(np.sum(y))
        if np.isfinite(s) and 0.98 <= s <= 1.02 and np.all((y >= -1e-6) & (y <= 1.0+1e-6)):
            return y
        y = y - np.max(y)
        y = np.exp(y)
        y /= (np.sum(y) + 1e-9)
        return y

    def predict(self, rgb_float01):
        x = self._quantize_in(rgb_float01)[None, ...]  # [1,H,W,C]
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output_index)[0]
        y = self._dequantize_out(y)
        y = self._softmax_if_needed(y)
        return y

# ----------------------- Main -----------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load YOLOv8 (PyTorch)
    from ultralytics import YOLO
    yolo = YOLO(args.det_model)

    # Load TFLite classifier
    clf = TFLiteClassifier(args.cls_model, threads=args.cls_threads)
    inW, inH = clf.target_hw
    print(f"[INFO] Classifier expects (W,H,C) = ({inW},{inH},3)")

    # Open source
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: cannot open source: {args.source}")
        return

    writer = None
    out_path = None

    fps_hist = deque(maxlen=30)
    last_label, last_conf = "—", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()

        # Run YOLO detection on this frame
        # Note: using the direct call is slightly faster than .predict
        results = yolo(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )

        disp = frame.copy()
        H, W = frame.shape[:2]

        # Get detections (choose the largest hand)
        boxes = []
        scores = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()         # (N,4)
                confs = r.boxes.conf.cpu().numpy()        # (N,)
                # classes = r.boxes.cls.cpu().numpy()     # not used, assuming single hand class
                boxes = xyxy
                scores = confs

        if len(boxes) > 0:
            # pick largest
            areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
            i = int(np.argmax(areas))
            x1,y1,x2,y2 = map(int, boxes[i])

            # draw YOLO bbox
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)

            # make a square crop with padding so classifier sees consistent framing
            sx1, sy1, sx2, sy2 = expand_square(x1, y1, x2, y2, W, H, pad=args.square_pad)
            roi = frame[sy1:sy2, sx1:sx2]
            if roi.size > 0:
                # preprocess for classifier
                img = cv2.resize(roi, (inW, inH), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = np.ascontiguousarray(img)

                # run classifier
                probs = clf.predict(img)
                # smooth a bit
                # (simple EMA to avoid a big deque if you prefer)
                if not hasattr(main, "_ema"):
                    main._ema = probs
                else:
                    main._ema = 0.6*main._ema + 0.4*probs
                probs_s = main._ema

                idx = int(np.argmax(probs_s))
                conf = float(probs_s[idx])
                lab  = ASL_LABELS[idx] if idx < len(ASL_LABELS) else "Unknown"

                last_label, last_conf = lab, conf

                # draw label if confident
                if conf >= args.cls_thr:
                    put_text_bg(disp, f"{lab} ({conf:.2f})", (10, 40), (0,255,0), 0.9, 2)
                else:
                    put_text_bg(disp, f"Uncertain ({conf:.2f})", (10, 40), (0,165,255), 0.9, 2)

                # optional: show the square crop box
                cv2.rectangle(disp, (sx1,sy1), (sx2,sy2), (255,255,0), 1)
            else:
                put_text_bg(disp, "Empty ROI", (10, 40), (0,0,255), 0.9, 2)
        else:
            put_text_bg(disp, "No hand detected", (10, 40), (0,0,255), 0.9, 2)

        # Performance HUD (bottom-left): RAM, CPU temp, FPS
        t1 = time.perf_counter()
        fps_hist.append(1.0 / max(1e-6, (t1 - t0)))
        fps = np.mean(fps_hist) if len(fps_hist) > 0 else 0.0
            
        draw_performance_hud(disp, fps)


        # View
        if args.view:
            cv2.imshow("YOLOv8 (PyTorch) + TFLite ASL Classifier", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save video
        if args.save:
            if writer is None:
                os.makedirs(args.save_dir, exist_ok=True)
                out_path = str(Path(args.save_dir) / "annotated.mp4")
                h, w = disp.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
            writer.write(disp)

    cap.release()
    if writer:
        writer.release()
        print(f"[✓] Saved video → {out_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
