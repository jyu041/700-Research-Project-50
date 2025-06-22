import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from datetime import datetime
from ultralytics import YOLO

# Define ASL alphabet labels
ASL_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'space'
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='ASL Sign Interpreter using YOLO and webcam')
    parser.add_argument('--model_path', type=str, default='./model.h5',
                        help='Path to the trained model (.h5 file)')
    parser.add_argument('--yolo_weights', type=str, default='best_hand.pt',
                        help='Path to YOLOv8 weights (default: best_hand.pt)')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                        help='Width of the video frame (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Height of the video frame (default: 480)')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Confidence threshold for predictions (default: 0.7)')
    parser.add_argument('--record', action='store_true',
                        help='Record the video with predictions')
    return parser.parse_args()

def load_and_prep_model(model_path):
    print(f"Loading ASL classification model from {model_path}...")
    model = load_model(model_path)
    model.summary()
    input_shape = model.input_shape[1:3]  # Exclude batch size
    print(f"Model expects input shape: {input_shape}")
    return model, input_shape

def create_video_writer(width, height, fps=20):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"asl_interpretation_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Recording video to {output_path}")
    return writer

def detect_hand_and_preprocess(frame, yolo_model, target_size):
    """Use YOLO to detect hand and return cropped+processed image for classification."""
    results = yolo_model.predict(source=frame, conf=0.3, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    if len(boxes) == 0:
        return frame, None

    # Pick the largest box (assume it's the main hand)
    largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    x1, y1, x2, y2 = map(int, largest_box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

    # Draw detection box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop ROI and preprocess
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame, None

    img = cv2.resize(roi, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = img / 255.0
    # img = img[..., np.newaxis]  # Make it (H, W, 1)
    return frame, img

def display_prediction(frame, prediction, labels, threshold):
    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx]
    predicted_label = labels[predicted_idx] if predicted_idx < len(labels) else "Unknown"

    if confidence < threshold:
        label_text = "Uncertain"
        color = (0, 0, 255)
    else:
        label_text = f"Predicted: {predicted_label}"
        color = (0, 255, 0)

    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

def main():
    args = parse_arguments()

    model, input_shape = load_and_prep_model(args.model_path)
    yolo_model = YOLO(args.yolo_weights)

    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    video_writer = create_video_writer(args.width, args.height) if args.record else None
    print("Webcam initialized. Press 'q' to quit, 's' to save a frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        display_frame, processed_img = detect_hand_and_preprocess(frame, yolo_model, input_shape)

        if processed_img is not None:
            input_tensor = np.expand_dims(processed_img, axis=0)
            prediction = model.predict(input_tensor, verbose=0)[0]
            result_frame = display_prediction(display_frame, prediction, ASL_LABELS, args.confidence)
        else:
            result_frame = display_frame
            cv2.putText(result_frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show controls
        cv2.putText(result_frame, "Controls: 'q'=quit, 's'=save", (10, result_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('ASL Sign Interpreter', result_frame)

        if video_writer:
            video_writer.write(result_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"asl_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"Saved frame to {filename}")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Interpreter closed.")

if __name__ == "__main__":
    main()
