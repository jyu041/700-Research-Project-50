# asl_interpreter.py
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from datetime import datetime

# Define ASL alphabet labels
ASL_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ASL Sign Interpreter using webcam')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model (.h5 file)')
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
    """Load the trained model and prepare it for inference."""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    model.summary()
    
    # Get input shape expected by the model
    input_shape = model.input_shape[1:3]  # Excluding batch dimension
    print(f"Model expects input shape: {input_shape}")
    
    return model, input_shape

def preprocess_frame(frame, target_size):
    """Preprocess frame for model inference."""
    # Extract region of interest (center square of the frame)
    h, w = frame.shape[:2]
    
    # Create a square ROI in the center
    size = min(h, w)
    x = (w - size) // 2
    y = (h - size) // 2
    roi = frame[y:y+size, x:x+size]
    
    # Draw the ROI rectangle
    cv2.rectangle(frame, (x, y), (x+size, y+size), (0, 255, 0), 2)
    
    # Preprocess ROI for the model
    img = cv2.resize(roi, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img / 255.0  # Normalize to [0,1]
    
    return frame, img

def display_prediction(frame, prediction, labels, threshold):
    """Display prediction on the frame."""
    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx]
    
    # Display the predicted label and confidence
    predicted_label = labels[predicted_idx] if predicted_idx < len(labels) else "Unknown"
    
    # If confidence is below threshold, show "Uncertain"
    if confidence < threshold:
        label_text = "Uncertain"
        confidence_color = (0, 0, 255)  # Red for low confidence
    else:
        label_text = f"Predicted: {predicted_label}"
        confidence_color = (0, 255, 0)  # Green for high confidence
    
    # Add prediction text to frame
    cv2.putText(frame, label_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, confidence_color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, confidence_color, 2)
    
    # Add helper text
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def create_video_writer(width, height, fps=20):
    """Create a video writer for recording."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"asl_interpretation_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Recording video to {output_path}")
    return writer

def main():
    args = parse_arguments()
    
    # Load the model
    model, input_shape = load_and_prep_model(args.model_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam initialized. Starting ASL interpretation...")
    
    # Set up video recording if requested
    video_writer = None
    if args.record:
        video_writer = create_video_writer(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    # Main loop
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)
        
        # Preprocess the frame
        display_frame, processed_img = preprocess_frame(frame, (input_shape[0], input_shape[1]))
        
        # Make prediction
        input_tensor = np.expand_dims(processed_img, axis=0)
        prediction = model.predict(input_tensor, verbose=0)[0]
        
        # Display prediction on frame
        result_frame = display_prediction(display_frame, prediction, ASL_LABELS, args.confidence)
        
        # Add help text for controlling the application
        instructions = [
            "Controls:",
            "- Press 'q' to quit",
            "- Press 's' to save current frame",
            "- Position your hand in the green box"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(result_frame, instruction, (10, result_frame.shape[0] - 100 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('ASL Sign Interpreter', result_frame)
        
        # Record frame if requested
        if video_writer is not None:
            video_writer.write(result_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' key to quit
        if key == ord('q'):
            break
            
        # 's' key to save current frame
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"asl_frame_{timestamp}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"Saved frame to {filename}")
    
    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("ASL Sign Interpreter closed.")

if __name__ == "__main__":
    main()