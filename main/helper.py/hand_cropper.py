# infer_static_images.py
# Runs YOLO hand detection + TF classifier on a folder of images.
# Saves annotated outputs + crops + CSV log — with optional label filtering

import os, csv, cv2, numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# ----------------------- constants & labels -----------------------
ASL_LABELS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','space'
]

# ----------------------- args -----------------------
def parse_arguments():
    p = argparse.ArgumentParser(description="Static image inference: YOLO hand + TF classifier")
    p.add_argument('--model_path',    type=str, default='model_v2.h5', help='Path to Keras .h5/.keras model')
    p.add_argument('--yolo_weights',  type=str, default='best.pt', help='YOLO hand detector weights (.pt)')
    p.add_argument('--imgsz',         type=int, default=640,   help='YOLO inference size')
    p.add_argument('--det_conf',      type=float, default=0.50,help='YOLO conf threshold')
    p.add_argument('--det_iou',       type=float, default=0.45,help='YOLO NMS IoU')
    p.add_argument('--clf_conf',      type=float, default=0.95,help='Classifier confidence threshold')
    p.add_argument('--hand_only',     action='store_true',     help="Filter to classes named like 'hand'")
    p.add_argument('--input_dir',     type=str, default=r"C:\Users\Tony\Downloads\captures", help='Folder of images')
    p.add_argument('--output_dir',    type=str, default="asl_sampled\\Q", help='Folder to save results')
    p.add_argument('--target_label',  type=str, default="Q",
                   help='Only save images if predicted label matches this (e.g., "E"). Case-insensitive.')
    return p.parse_args()

# ----------------------- utils -----------------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def draw_box(img, box, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), (255, 0, 0), -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def detect_best_hand_bgr(img_bgr, yolo, imgsz, conf, iou, hand_class_ids=None):
    """Return (xyxy box or None) of highest-confidence kept detection."""
    res = yolo.predict(source=img_bgr, imgsz=imgsz, conf=conf, iou=iou, max_det=10, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    cls  = res.boxes.cls.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    sco  = res.boxes.conf.cpu().numpy()

    idxs = list(range(len(cls)))
    if hand_class_ids is not None:
        idxs = [i for i,c in enumerate(cls) if int(c) in hand_class_ids]
    if not idxs:
        return None

    best_i = idxs[int(np.argmax(sco[idxs]))]
    return xyxy[best_i]

def prep_for_classifier(img_bgr_roi, target_hw):
    h, w = target_hw
    rgb = cv2.cvtColor(cv2.resize(img_bgr_roi, (w, h)), cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0

# ----------------------- main -----------------------
def main():
    args = parse_arguments()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    ann_dir = out_dir / "annotated"
    ensure_dir(out_dir); ensure_dir(ann_dir)

    # CSV log
    log_path = out_dir / "log.csv"
    new_log  = not log_path.exists()
    logf = open(log_path, "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["timestamp","filename","pred_label","pred_conf","x1","y1","x2","y2"])

    # load models
    clf = load_model(args.model_path)
    target_hw = clf.input_shape[1:3]  # (H, W)
    yolo = YOLO(args.yolo_weights)

    # resolve hand class ids (optional)
    hand_class_ids = None
    if args.hand_only and hasattr(yolo, "names"):
        hand_class_ids = [i for i,n in yolo.names.items() if "hand" in str(n).lower()]
        if not hand_class_ids:
            print("⚠️ No class named like 'hand' found, using all classes.")
            hand_class_ids = None

    # normalize target_label if given
    target_label = args.target_label.upper() if args.target_label else None

    # images
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    imgs = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in exts])
    if not imgs:
        print("No input images found."); return

    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Skip unreadable: {p}")
            continue

        box = detect_best_hand_bgr(img, yolo, args.imgsz, args.det_conf, args.det_iou, hand_class_ids)
        if box is None:
            print(f"{p.name}: no hand detected (skipped)")
            continue

        H, W = img.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            print(f"{p.name}: empty ROI (skipped)")
            continue

        # classify
        inp = prep_for_classifier(roi, target_hw)
        pred = clf.predict(np.expand_dims(inp, 0), verbose=0)[0]
        idx  = int(np.argmax(pred))
        conf = float(pred[idx])
        label= ASL_LABELS[idx] if idx < len(ASL_LABELS) else str(idx)

        # If a target_label is set, skip saving if not matching
        if target_label is not None and label.upper() != target_label:
            print(f"{p.name}: predicted {label}, not target {target_label} → skipped")
            continue

        # build annotation text (respect conf threshold)
        shown_label = label if conf >= args.clf_conf else "uncertain"
        ann = img.copy()
        draw_box(ann, (x1, y1, x2, y2), f"{shown_label} {conf:.2f}")

        # save annotated and crop
        out_name = p.with_suffix(".jpg").name
        cv2.imwrite(str(ann_dir / out_name), ann)
        cv2.imwrite(str(out_dir / f"{p.stem}.jpg"), roi)

        # log
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logw.writerow([ts, out_name, label, f"{conf:.4f}", x1, y1, x2, y2])

        print(f"{p.name}: saved ({label}, {conf:.2f})")

    logf.close()
    print("✅ Done.")

if __name__ == "__main__":
    main()
