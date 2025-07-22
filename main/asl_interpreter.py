#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, cv2, numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from datetime import datetime
import argparse

# ----------------------- constants & labels -----------------------
ASL_LABELS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','space'
]
SAVE_ROOT = "captures"                # where screenshots go
LOG_PATH  = os.path.join(SAVE_ROOT, "log.csv")

# ----------------------- helper functions -------------------------
def parse_arguments():
    p = argparse.ArgumentParser(description='ASL Sign Interpreter using YOLO and webcam')
    p.add_argument('--model_path',  type=str, default='./model.h5')
    p.add_argument('--yolo_weights',type=str, default='best_hand.pt')
    p.add_argument('--camera_id',   type=int, default=0)
    p.add_argument('--width',       type=int, default=640)
    p.add_argument('--height',      type=int, default=480)
    p.add_argument('--confidence',  type=float, default=0.7)
    return p.parse_args()

def load_and_prep_model(path):
    print(f"Loading ASL classification model from {path} …")
    model = load_model(path)
    input_shape = model.input_shape[1:3]
    print(f"Model expects input shape: {input_shape}")
    return model, input_shape

def ensure_dirs():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp","filename","label","confidence","x1","y1","x2","y2"]
            )

def detect_hand_and_preprocess(frame, yolo_model, target_size):
    """Return frame w/ box drawn, pre‑processed ROI, bbox, raw ROI."""
    results = yolo_model.predict(source=frame, conf=0.3, verbose=False)
    boxes   = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    if len(boxes) == 0:
        return frame, None, None, None  # <‑‑ RETURN 4 VALUES

    largest = max(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
    x1,y1,x2,y2 = map(int, largest)
    x1,y1 = max(x1,0), max(y1,0)
    x2,y2 = min(x2,frame.shape[1]), min(y2,frame.shape[0])
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame, None, None, None  # <‑‑ RETURN 4 VALUES

    img = cv2.resize(roi, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return frame, img, (x1,y1,x2,y2), roi

def draw_prediction(frame, pred_vec, labels, thresh):
    idx  = int(np.argmax(pred_vec))
    conf = float(pred_vec[idx])
    lab  = labels[idx] if idx < len(labels) else "Unknown"

    if conf < thresh:
        lab_show, color = "Uncertain", (0,0,255)
    else:
        lab_show, color = f"Predicted: {lab}", (0,255,0)

    cv2.putText(frame, lab_show, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Confidence: {conf:.2f}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return lab, conf

# ---------- NEW: save the entire frame (screenshot) ----------
def save_screenshot(label, confidence, bbox, frame):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder    = os.path.join(SAVE_ROOT, label)
    os.makedirs(folder, exist_ok=True)
    filename  = f"{label}_{timestamp}.jpg"
    path      = os.path.join(folder, filename)
    cv2.imwrite(path, frame)

    with open(LOG_PATH,"a",newline="") as f:
        csv.writer(f).writerow([timestamp, filename, label, f"{confidence:.4f}", *(bbox or (0,0,0,0))])

    print(f"[SAVED] screenshot → {path}  (conf {confidence:.2f})")

# --------------------------- main loop ----------------------------
def main():
    args = parse_arguments(); ensure_dirs()
    model, in_shape = load_and_prep_model(args.model_path)
    yolo  = YOLO(args.yolo_weights)

    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened(): 
        print("Error: Could not open webcam."); return

    print("Webcam initialized. Press 'q' to quit, 's' to save screenshot.")

    while True:
        ok, frame = cap.read()
        if not ok: print("Error: Failed to read frame."); break
        frame = cv2.flip(frame, 1)

        disp, proc, bbox, _ = detect_hand_and_preprocess(frame, yolo, in_shape)

        label, conf = "uncertain", 0.0
        if proc is not None:
            pred = model.predict(np.expand_dims(proc,0), verbose=0)[0]
            label, conf = draw_prediction(disp, pred, ASL_LABELS, args.confidence)
        else:
            cv2.putText(disp, "No hand detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.putText(disp, "Controls: 'q'=quit, 's'=save", (10, disp.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)
        cv2.imshow("ASL Sign Interpreter", disp)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('s'):
            label_to_use = label if conf >= args.confidence else "uncertain"
            save_screenshot(label_to_use, conf, bbox, disp)

    cap.release(); cv2.destroyAllWindows(); print("Interpreter closed.")

if __name__ == "__main__":
    main()
