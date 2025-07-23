#!/usr/bin/env python3
"""
ASL Sign Interpreter for Raspberry Pi with hand detection using TFLite
Replicates the functionality of asl_interpreter.py for Raspberry Pi
"""

import argparse
import numpy as np
import tflite_runtime.interpreter as tflite
from datetime import datetime
import os
import cv2
import csv
from PIL import Image, ImageDraw, ImageFont
from picamera2 import Picamera2
import time

# Define ASL alphabet labels (matching original code)
ASL_LABELS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','space'
]

SAVE_ROOT = "captures"
LOG_PATH = os.path.join(SAVE_ROOT, "log.csv")


def parse_arguments():
    parser = argparse.ArgumentParser(description='ASL Sign Interpreter on Raspberry Pi with hand detection')
    parser.add_argument('--asl_model_path', type=str, default='./model.tflite',
                        help='Path to the ASL classification TFLite model (.tflite)')
    parser.add_argument('--yolo_model_path', type=str, default='./best_hand.tflite',
                        help='Path to the YOLO hand detection TFLite model (.tflite)')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Confidence threshold for ASL predictions (default: 0.7)')
    parser.add_argument('--hand_confidence', type=float, default=0.3,
                        help='Confidence threshold for hand detection (default: 0.3)')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera height (default: 480)')
    parser.add_argument('--record', action='store_true',
                        help='Save annotated frames to disk instead of displaying')
    return parser.parse_args()


def load_tflite_model(path):
    """Load a TFLite model and return interpreter and details"""
    print(f"Loading TFLite model from {path}...")
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_shape = input_details['shape']
    input_index = input_details['index']
    output_index = output_details['index']
    input_dtype = input_details['dtype']
    
    return interpreter, input_index, output_index, input_shape, input_dtype


def ensure_dirs():
    """Create necessary directories and log file"""
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "filename", "label", "confidence", "x1", "y1", "x2", "y2"]
            )


def detect_hands_tflite(image_array, yolo_interpreter, yolo_input_idx, yolo_output_idx, 
                       yolo_input_shape, yolo_input_dtype, confidence_threshold=0.3):
    """
    Detect hands using TFLite YOLO model
    Returns: list of bounding boxes [x1, y1, x2, y2] with confidence > threshold
    """
    # Prepare input for YOLO
    input_height, input_width = yolo_input_shape[1], yolo_input_shape[2]
    original_height, original_width = image_array.shape[:2]
    
    # Resize image for YOLO input
    resized_image = cv2.resize(image_array, (input_width, input_height))
    
    # Normalize to [0, 1] if float32, keep as uint8 if uint8
    if yolo_input_dtype == np.float32:
        input_tensor = (resized_image / 255.0).astype(np.float32)
    else:
        input_tensor = resized_image.astype(yolo_input_dtype)
    
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # Run inference
    yolo_interpreter.set_tensor(yolo_input_idx, input_tensor)
    yolo_interpreter.invoke()
    
    # Get output
    output_data = yolo_interpreter.get_tensor(yolo_output_idx)
    
    # Process YOLO output (this may need adjustment based on your specific YOLO export format)
    boxes = []
    
    # YOLOv8 TFLite output format: [batch, num_detections, 85] where 85 = 4(bbox) + 1(conf) + 80(classes)
    # For hand detection, we typically only have 1 class
    if len(output_data.shape) == 3:
        detections = output_data[0]  # Remove batch dimension
        
        for detection in detections:
            # YOLO format: [center_x, center_y, width, height, confidence, class_scores...]
            if len(detection) >= 5:
                center_x, center_y, width, height, conf = detection[:5]
                
                if conf > confidence_threshold:
                    # Convert to pixel coordinates
                    x1 = int((center_x - width/2) * original_width)
                    y1 = int((center_y - height/2) * original_height)
                    x2 = int((center_x + width/2) * original_width)
                    y2 = int((center_y + height/2) * original_height)
                    
                    # Clamp to image boundaries
                    x1 = max(0, min(x1, original_width))
                    y1 = max(0, min(y1, original_height))
                    x2 = max(0, min(x2, original_width))
                    y2 = max(0, min(y2, original_height))
                    
                    boxes.append([x1, y1, x2, y2, conf])
    
    return boxes


def preprocess_roi_for_asl(roi_array, target_size):
    """Preprocess ROI for ASL classification"""
    if roi_array.size == 0:
        return None
    
    # Resize to target size
    resized = cv2.resize(roi_array, target_size)
    
    # Convert BGR to RGB and normalize
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_image.astype(np.float32) / 255.0
    
    return normalized


def draw_prediction_on_frame(frame, label, confidence, bbox, threshold):
    """Draw prediction and bounding box on frame"""
    # Draw bounding box if available
    if bbox:
        x1, y1, x2, y2 = bbox[:4]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Determine display text and color
    if confidence < threshold:
        display_text = "Uncertain"
        color = (0, 0, 255)  # Red
    else:
        display_text = f"Predicted: {label}"
        color = (0, 255, 0)  # Green
    
    # Draw text
    cv2.putText(frame, display_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame


def save_screenshot(label, confidence, bbox, frame):
    """Save screenshot with metadata logging"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder = os.path.join(SAVE_ROOT, label)
    os.makedirs(folder, exist_ok=True)
    filename = f"{label}_{timestamp}.jpg"
    path = os.path.join(folder, filename)
    
    cv2.imwrite(path, frame)
    
    # Log to CSV
    with open(LOG_PATH, "a", newline="") as f:
        bbox_coords = bbox[:4] if bbox else (0, 0, 0, 0)
        csv.writer(f).writerow([
            timestamp, filename, label, f"{confidence:.4f}", *bbox_coords
        ])
    
    print(f"[SAVED] screenshot → {path} (conf {confidence:.2f})")


def main():
    args = parse_arguments()
    ensure_dirs()
    
    # Load models
    asl_interpreter, asl_input_idx, asl_output_idx, asl_input_shape, asl_input_dtype = load_tflite_model(args.asl_model_path)
    yolo_interpreter, yolo_input_idx, yolo_output_idx, yolo_input_shape, yolo_input_dtype = load_tflite_model(args.yolo_model_path)
    
    # ASL model input size (height, width)
    asl_input_size = (asl_input_shape[2], asl_input_shape[1])  # (width, height) for cv2.resize
    
    # Initialize camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration({
        "size": (args.width, args.height),
        "format": "RGB888"
    })
    picam2.configure(config)
    picam2.start()
    
    frame_count = 0
    if args.record:
        os.makedirs('annotated_frames', exist_ok=True)
    
    print("ASL Interpreter started. Controls:")
    print("- 's' key: Save screenshot")
    print("- 'q' key or Ctrl+C: Quit")
    
    try:
        while True:
            # Capture frame
            frame_rgb = picam2.capture_array()  # RGB format
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            frame_display = frame_bgr.copy()
            
            # Detect hands
            hand_boxes = detect_hands_tflite(
                frame_bgr, yolo_interpreter, yolo_input_idx, yolo_output_idx,
                yolo_input_shape, yolo_input_dtype, args.hand_confidence
            )
            
            label, confidence, bbox = "uncertain", 0.0, None
            
            if hand_boxes:
                # Use the largest detected hand
                largest_box = max(hand_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                x1, y1, x2, y2 = largest_box[:4]
                bbox = largest_box
                
                # Extract ROI
                roi = frame_bgr[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Preprocess ROI for ASL classification
                    processed_roi = preprocess_roi_for_asl(roi, asl_input_size)
                    
                    if processed_roi is not None:
                        # Run ASL classification
                        input_tensor = np.expand_dims(processed_roi, axis=0).astype(asl_input_dtype)
                        asl_interpreter.set_tensor(asl_input_idx, input_tensor)
                        asl_interpreter.invoke()
                        
                        predictions = asl_interpreter.get_tensor(asl_output_idx)[0]
                        pred_idx = int(np.argmax(predictions))
                        confidence = float(predictions[pred_idx])
                        label = ASL_LABELS[pred_idx] if pred_idx < len(ASL_LABELS) else "Unknown"
            
            # Draw results on frame
            if bbox:
                frame_display = draw_prediction_on_frame(
                    frame_display, label, confidence, bbox, args.confidence
                )
            else:
                cv2.putText(frame_display, "No hand detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add controls text
            cv2.putText(frame_display, "Controls: 'q'=quit, 's'=save", 
                       (10, frame_display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display or save frame
            if args.record:
                fname = f"annotated_frames/frame_{frame_count:05d}.jpg"
                cv2.imwrite(fname, frame_display)
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Recorded {frame_count} frames...")
            else:
                cv2.imshow("ASL Sign Interpreter", frame_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    label_to_use = label if confidence >= args.confidence else "uncertain"
                    save_screenshot(label_to_use, confidence, bbox, frame_display)
            
            time.sleep(0.03)  # Small delay to prevent excessive CPU usage
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.close()
        cv2.destroyAllWindows()
        print("Interpreter closed.")


if __name__ == '__main__':
    main()