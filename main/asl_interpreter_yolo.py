# asl_interpreter_yolo_only.py
import os
import cv2
import argparse
from datetime import datetime
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO-only ASL Sign Interpreter using webcam')
    parser.add_argument('--model_path', type=str, default='best_group.pt',
                        help='Path to YOLOv8 model (.pt)')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Webcam device ID')
    parser.add_argument('--width', type=int, default=640,
                        help='Video frame width')
    parser.add_argument('--height', type=int, default=480,
                        help='Video frame height')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--record', action='store_true',
                        help='Record video with detections')
    return parser.parse_args()

def display_prediction(frame, box, label, conf, threshold):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0) if conf >= threshold else (0, 0, 255)
    text = f"{label} ({conf:.2f})" if conf >= threshold else "Uncertain"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def create_video_writer(width, height):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"asl_yolo_only_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, 20, (width, height)), path

def main():
    args = parse_arguments()
    print("Loading YOLO model...")
    model = YOLO(args.model_path)

    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    print("Webcam ready. Press 'q' to quit.")
    video_writer, out_path = (None, None)
    if args.record:
        video_writer, out_path = create_video_writer(args.width, args.height)
        print(f"Recording started: {out_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror effect
        results = model(frame, verbose=False)[0]

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf)
                class_id = int(box.cls)
                label = model.names[class_id]
                display_prediction(frame, (x1, y1, x2, y2), label, conf, args.confidence)

        # UI instructions
        cv2.putText(frame, "Press 'q' to quit", (10, args.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("ASL Interpreter (YOLO-only)", frame)

        if args.record:
            video_writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Closed.")

if __name__ == "__main__":
    main()
