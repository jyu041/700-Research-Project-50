#!/usr/bin/env python3
"""
Real-time Hand Tracking with your webcam using MediaPipe + OpenCV.

Features:
- Tracks up to N hands (default: 2) in real time
- Draws 21 hand landmarks and connections
- Shows handedness (Left/Right) with confidence
- Displays FPS and basic pinch distance (thumb tip ↔ index tip)
- CLI flags for camera index, resolution, and model settings

Install dependencies:
    pip install opencv-python mediapipe

Run:
    python hand_tracker.py --camera 0 --width 1280 --height 720 --max-hands 2
Press 'q' or ESC to quit.
"""

import argparse
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import mediapipe as mp


@dataclass
class Config:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    max_hands: int = 2
    model_complexity: int = 1  # 0=Lite, 1=Full, 2=Heavy
    min_detection_conf: float = 0.5
    min_tracking_conf: float = 0.5
    selfie: bool = True  # Flip frame horizontally for a selfie view


class FPSTimer:
    def __init__(self, avg_over: int = 10):
        self.past = []
        self.avg_over = avg_over
        self.last = time.time()

    def tick(self) -> float:
        now = time.time()
        dt = now - self.last
        self.last = now
        if dt <= 0:
            return 0.0
        fps = 1.0 / dt
        self.past.append(fps)
        if len(self.past) > self.avg_over:
            self.past.pop(0)
        return sum(self.past) / len(self.past)


def put_text(
    img,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    bg: Optional[Tuple[int, int, int]] = (0, 0, 0),
):
    """Utility: draw text with optional background for readability."""
    if bg is not None:
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x, y - h - baseline), (x + w, y + baseline // 2), bg, -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main(cfg: Config):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # Set up camera
    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    # Try to set resolution (may be ignored by some webcams/drivers)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cfg.camera_index}. Try --camera 1 or 2.")

    # MediaPipe Hands model
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=cfg.max_hands,
        model_complexity=cfg.model_complexity,
        min_detection_confidence=cfg.min_detection_conf,
        min_tracking_confidence=cfg.min_tracking_conf,
    )

    fps_timer = FPSTimer(avg_over=20)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from camera. Exiting...")
                break

            if cfg.selfie:
                frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Inference
            results = hands.process(rgb)

            h, w = frame.shape[:2]

            # Draw results
            if results.multi_hand_landmarks:
                # Pair landmarks with handedness (they are aligned lists)
                handedness_list = results.multi_handedness or []
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    # Handedness label (Left/Right) with confidence
                    label = None
                    if idx < len(handedness_list):
                        category = handedness_list[idx].classification[0]
                        label = f"{category.label} {category.score:.2f}"  # e.g., "Right 0.98"

                    # Put label near index fingertip (landmark 8)
                    lm = hand_landmarks.landmark
                    ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                    if label:
                        put_text(frame, label, (ix + 10, iy - 10))

                    # Simple example: pinch distance (thumb tip=4, index tip=8)
                    tx, ty = int(lm[4].x * w), int(lm[4].y * h)
                    ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                    pinch = ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5
                    put_text(frame, f"Pinch: {int(pinch)}px", (ix + 10, iy + 20), scale=0.5)

            # FPS
            fps = fps_timer.tick()
            put_text(frame, f"FPS: {fps:5.1f}", (10, 30))

            # Instructions
            put_text(frame, "Press 'q' or ESC to quit | 'f' toggle selfie flip", (10, h - 10), scale=0.5, bg=(0, 0, 0))

            cv2.imshow("Hand Tracker (MediaPipe + OpenCV)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or 'q'
                break
            elif key in (ord('f'), ord('F')):
                cfg.selfie = not cfg.selfie

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Webcam Hand Tracker using MediaPipe + OpenCV")
    p.add_argument("--camera", type=int, default=0, help="Camera index (0 is default)")
    p.add_argument("--width", type=int, default=1280, help="Capture width")
    p.add_argument("--height", type=int, default=720, help="Capture height")
    p.add_argument("--max-hands", type=int, default=2, help="Max number of hands to detect")
    p.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2], help="Model complexity (0=Lite, 1=Full, 2=Heavy)")
    p.add_argument("--min-detection-conf", type=float, default=0.5, help="Minimum detection confidence")
    p.add_argument("--min-tracking-conf", type=float, default=0.5, help="Minimum tracking confidence")
    p.add_argument("--no-selfie", action="store_true", help="Disable selfie (mirrored) view")
    args = p.parse_args()

    return Config(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        max_hands=args.max_hands,
        model_complexity=args.model_complexity,
        min_detection_conf=args.min_detection_conf,
        min_tracking_conf=args.min_tracking_conf,
        selfie=not args.no_selfie,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
