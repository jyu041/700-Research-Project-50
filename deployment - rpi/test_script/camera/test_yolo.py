#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, cv2, numpy as np
import argparse
from pathlib import Path

# -------------------- Args (minimal) --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Simple TFLite YOLO hand demo: box + confidence (top-left), FPS (bottom-right)")
    p.add_argument("--model", type=str, default="../models/v5n_100epoch_img416.tflite", help="Path to YOLO .tflite model")
    p.add_argument("--source", type=str, default="0", help="0 (webcam) or path to video/image")
    p.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    p.add_argument("--iou",  type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--threads", type=int, default=0, help="TFLite num threads (0=auto)")
    return p.parse_args()

# -------------------- TFLite helper --------------------
def _get_tflite_interpreter(model_path, threads):
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
            raise RuntimeError("No TFLite interpreter available. Install `tflite_runtime` or `tensorflow`.") from e

    threads = threads if (threads and threads > 0) else max(1, (os.cpu_count() or 4) - 1)
    print(f"Loading TFLite model: {model_path}  (threads={threads})")
    itp = Runtime.Interpreter(model_path=model_path, num_threads=threads)
    itp.allocate_tensors()
    return itp

def _nms_xyxy(boxes, scores, iou_thres=0.45, top_k=300):
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1][:top_k]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

class TFLiteYOLODetector:
    """
    Expects output shaped like (N,6) or (1,N,6):
      either [x1,y1,x2,y2,score,cls] OR [cx,cy,w,h,score,cls]
    Accepts normalized or absolute coords. Scales back to original frame size.
    """
    def __init__(self, model_path, conf=0.3, iou=0.45, threads=0):
        self.interpreter = _get_tflite_interpreter(model_path, threads)
        self.in_details  = self.interpreter.get_input_details()[0]
        self.out_details = self.interpreter.get_output_details()
        self.input_index = self.in_details['index']
        self.in_dtype    = self.in_details['dtype']
        self.in_scale, self.in_zero = (self.in_details.get('quantization') or (0.0, 0))
        self.in_shape = self.in_details['shape']  # [1,H,W,C]
        _, self.in_h, self.in_w, self.in_c = self.in_shape
        self.out_idx = [d['index'] for d in self.out_details]
        self.conf = float(conf)
        self.iou  = float(iou)
        # warm-up
        dummy = np.zeros((1, self.in_h, self.in_w, self.in_c), dtype=self.in_dtype)
        self.interpreter.set_tensor(self.input_index, dummy)
        self.interpreter.invoke()

    def _quantize(self, x_float01):
        if np.issubdtype(self.in_dtype, np.integer):
            if self.in_scale and self.in_scale > 0:
                x = x_float01 / self.in_scale + self.in_zero
            else:
                x = x_float01 * 255.0
            x = np.clip(x, 0, 255).astype(self.in_dtype)
        else:
            x = x_float01.astype(self.in_dtype)
        return x

    def _gather_head(self):
        outs = [self.interpreter.get_tensor(i) for i in self.out_idx]
        outs.sort(key=lambda a: int(np.prod(a.shape)), reverse=True)
        return outs[0]

    def _parse(self, raw, in_size):
        arr = raw
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] < 6:
            return (np.zeros((0,4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32))

        A,B,C,D,S,CLS = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4], arr[:,5]
        iw, ih = in_size
        # Heuristic: are numbers normalized?
        max_coord = float(max(A.max(initial=0), B.max(initial=0), C.max(initial=0), D.max(initial=0)))
        norm = max_coord <= 2.2
        if norm:
            A, B, C, D = A*iw, B*ih, C*iw, D*ih

        # Guess xyxy vs cxcywh
        xyxy_like = np.mean((C > A) & (D > B)) > 0.6
        if xyxy_like:
            x1, y1, x2, y2 = A, B, C, D
        else:
            x1 = A - C/2.0; y1 = B - D/2.0
            x2 = A + C/2.0; y2 = B + D/2.0

        boxes  = np.stack([x1,y1,x2,y2], axis=1).astype(np.float32)
        scores = S.astype(np.float32)
        clses  = CLS.astype(np.int32)
        return boxes, scores, clses

    def detect(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        iw, ih = self.in_w, self.in_h

        img = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = self._quantize(img)[None, ...]

        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        raw = self._gather_head()

        boxes_in, scores, classes = self._parse(raw, (iw, ih))
        if boxes_in.size == 0:
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)

        keep = scores >= self.conf
        if not np.any(keep):
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)
        boxes_in, scores = boxes_in[keep], scores[keep]

        keep_idx = _nms_xyxy(boxes_in, scores, iou_thres=self.iou)
        boxes_in, scores = boxes_in[keep_idx], scores[keep_idx]

        # scale back to original
        sx, sy = W/float(iw), H/float(ih)
        boxes_out = boxes_in.copy()
        boxes_out[:, [0,2]] *= sx
        boxes_out[:, [1,3]] *= sy

        # clamp
        boxes_out[:,0] = np.clip(boxes_out[:,0], 0, W-1)
        boxes_out[:,1] = np.clip(boxes_out[:,1], 0, H-1)
        boxes_out[:,2] = np.clip(boxes_out[:,2], 0, W-1)
        boxes_out[:,3] = np.clip(boxes_out[:,3], 0, H-1)
        return boxes_out, scores

# -------------------- Drawing helpers --------------------
def put_text_bg(img, text, org, color=(255,255,255), scale=0.8, thickness=2):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x-4, y-h-baseline-4), (x+w+4, y+4), (0,0,0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def put_text_bottom_right(img, text, margin=10, color=(0,255,255), scale=0.8, thickness=2):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    H, W = img.shape[:2]
    x = W - tw - margin
    y = H - bl - margin
    cv2.rectangle(img, (x-4, y-th-bl-4), (x+tw+4, y+4), (0,0,0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

# -------------------- Main (always view) --------------------
def main():
    args = parse_args()
    det = TFLiteYOLODetector(args.model, conf=args.conf, iou=args.iou, threads=args.threads)

    # Source (webcam by default)
    src = 0 if args.source == "0" else args.source
    # Single image path? Just run once and show
    if isinstance(src, str) and Path(src).exists() and Path(src).is_file():
        frame = cv2.imread(src)
        boxes, scores = det.detect(frame)
        disp = frame.copy()
        if len(boxes) > 0:
            # choose largest
            areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
            i = int(np.argmax(areas))
            x1,y1,x2,y2 = map(int, boxes[i])
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
            put_text_bg(disp, f"Confidence: {float(scores[i]):.2f}", (10, 30), (0,255,0))
        else:
            put_text_bg(disp, "No hand detected", (10, 30), (0,0,255))
        put_text_bottom_right(disp, "FPS: N/A")
        cv2.imshow("Hand (TFLite)", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Video / webcam loop
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: could not open source: {src}")
        return

    prev = time.perf_counter()
    ema_fps = None
    alpha = 0.12  # EMA factor

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.perf_counter()
        dt = now - prev
        prev = now
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        ema_fps = inst_fps if ema_fps is None else (alpha * inst_fps + (1 - alpha) * ema_fps)

        # Detect (every frame for simplicity/stability)
        boxes, scores = det.detect(frame)
        disp = frame.copy()

        if len(boxes) > 0:
            areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
            i = int(np.argmax(areas))
            x1,y1,x2,y2 = map(int, boxes[i])
            conf = float(scores[i])
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
            # Confidence top-left
            put_text_bg(disp, f"Confidence: {conf:.2f}", (10, 30), (0,255,0))
        else:
            put_text_bg(disp, "No hand detected", (10, 30), (0,0,255))

        # FPS bottom-right
        fps_show = ema_fps if ema_fps is not None else inst_fps
        put_text_bottom_right(disp, f"FPS: {fps_show:.1f}")

        # Always view
        cv2.imshow("Hand (TFLite)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
