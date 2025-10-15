#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, cv2, numpy as np
import argparse
from datetime import datetime
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----------------------- defaults -----------------------
ASL_LABELS_FALLBACK = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','space'
]
SAVE_ROOT = "captures"
LOG_PATH  = os.path.join(SAVE_ROOT, "log.csv")

# ----------------------- args -----------------------
def parse_arguments():
    p = argparse.ArgumentParser(description='ASL/BSL Interpreter (merge close hands, single top-left display)')
    p.add_argument('--model_path',   type=str, default='model_all.h5')
    p.add_argument('--yolo_weights', type=str, default='best_hand.pt')
    p.add_argument('--labels_path',  type=str, default='labels.txt')
    p.add_argument('--camera_id',    type=int, default=0)
    p.add_argument('--width',        type=int, default=640)
    p.add_argument('--height',       type=int, default=480)
    p.add_argument('--confidence',   type=float, default=0.7)
    p.add_argument('--det_conf',     type=float, default=0.30)
    p.add_argument('--det_iou',      type=float, default=0.45)
    return p.parse_args()

# ----------------------- helpers -----------------------
def load_labels(path_txt):
    if os.path.exists(path_txt):
        with open(path_txt, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    print(f"⚠️ labels file not found: {path_txt} — using fallback list")
    return ASL_LABELS_FALLBACK[:]

def ensure_dirs():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp","filename","label","confidence","x1","y1","x2","y2"]
            )

def load_and_prep_model(path):
    model = load_model(path)
    input_shape = model.input_shape[1:3]
    if input_shape is None or None in input_shape:
        input_shape = (224,224)
    print(f"✅ Loaded classifier. Input shape: {input_shape}")
    return model, input_shape

def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
    inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0,inter_x2-inter_x1), max(0,inter_y2-inter_y1)
    inter = iw*ih
    if inter == 0: return 0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a+area_b-inter+1e-6)

def should_merge(b1,b2):
    # vertical overlap and small horizontal gap OR decent IoU
    (lx1,ly1,lx2,ly2),(rx1,ry1,rx2,ry2) = sorted([b1,b2], key=lambda b:(b[0]+b[2])/2)
    ov_y = max(0, min(ly2, ry2) - max(ly1, ry1))
    h1, h2 = ly2-ly1, ry2-ry1
    v_overlap_ratio = ov_y / max(1, min(h1,h2))
    gap = max(0, rx1 - lx2)
    w1, w2 = lx2-lx1, rx2-rx1
    gap_ratio = gap / max(1, min(w1,w2))
    return (v_overlap_ratio >= 0.5 and gap_ratio <= 0.35) or (iou(b1,b2) >= 0.15)

def union_box(b1,b2):
    return (min(b1[0],b2[0]), min(b1[1],b2[1]),
            max(b1[2],b2[2]), max(b1[3],b2[3]))

def smooth_box(prev, curr, alpha=0.7):
    if prev is None: return curr
    return tuple(int(alpha*p+(1-alpha)*c) for p,c in zip(prev,curr))

def crop_and_prep(frame, box, target_size):
    H,W = frame.shape[:2]
    x1,y1,x2,y2 = map(int,box)
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(W-1,x2), min(H-1,y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None
    img = cv2.resize(roi, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
    return img

def detect_hands(yolo, frame, conf, iou_thresh):
    res = yolo.predict(source=frame, conf=conf, iou=iou_thresh, max_det=4, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None and len(res.boxes) else []
    boxes = [tuple(map(int,b)) for b in boxes]
    boxes = sorted(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes[:2]

# ----------------------- main -----------------------
def main():
    args = parse_arguments()
    ensure_dirs()
    model, in_shape = load_and_prep_model(args.model_path)
    labels = load_labels(args.labels_path)
    if len(labels) != model.output_shape[-1]:
        print(f"⚠️ Label length {len(labels)} != model outputs {model.output_shape[-1]}")

    yolo = YOLO(args.yolo_weights)
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("❌ Could not open webcam"); return

    print("✅ Webcam ready. 'q' = quit, 's' = save screenshot.")
    prev_box = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame,1)
        disp = frame.copy()

        boxes = detect_hands(yolo, disp, args.det_conf, args.det_iou)
        chosen_box = None

        if len(boxes) == 2 and should_merge(boxes[0], boxes[1]):
            chosen_box = union_box(boxes[0], boxes[1])
        elif len(boxes) >= 1:
            # if two hands far apart, pick the largest box only
            chosen_box = boxes[0]

        if chosen_box is not None:
            chosen_box = smooth_box(prev_box, chosen_box, alpha=0.7)
            prev_box = chosen_box

            inp = crop_and_prep(frame, chosen_box, in_shape)
            if inp is not None:
                pred = model.predict(np.expand_dims(inp,0), verbose=0)[0]
                idx = int(np.argmax(pred))
                conf = float(pred[idx])
                lab  = labels[idx] if idx < len(labels) else "Unknown"
                shown = lab if conf >= args.confidence else "Uncertain"

                # Top-left overlay only
                cv2.putText(disp, f"{shown}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(disp, f"{conf:.2f}", (10,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            prev_box = None
            cv2.putText(disp, "No hand detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.putText(disp, "Controls: 'q'=quit, 's'=save", (10, disp.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
        cv2.imshow("ASL/BSL Interpreter (merged)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            out_path = os.path.join(SAVE_ROOT, f"frame_{ts}.jpg")
            cv2.imwrite(out_path, disp)
            print(f"[SAVED] {out_path}")

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
