#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-Optimized Dynamic ASL System
Matches your working TensorFlow 2.19.0 + tf_keras setup
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import random
from datetime import datetime
import argparse
import logging
import pickle
from collections import deque

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports (matching your working environment)
import tensorflow as tf
import tf_keras as keras
from tf_keras.models import Sequential, load_model
from tf_keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Model
from tf_keras.optimizers import Adam
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ML imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# MediaPipe
import mediapipe as mp

# GPU Setup - Force GPU usage like your working setup
print("TensorFlow version:", tf.__version__)
print("tf_keras version:", keras.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"GPUs detected: {len(gpus)}")

if gpus:
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set default GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        
        # Test GPU
        with tf.device('/GPU:0'):
            test = tf.constant([1.0, 2.0, 3.0, 4.0])
            result = tf.reduce_sum(test)
            print(f"GPU test successful: {result.numpy()}")
            
        print("✅ GPU enabled and tested successfully")
        
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("❌ No GPU detected - check your TensorFlow-GPU installation")

# Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MediaPipeExtractor:
    """Optimized MediaPipe feature extractor"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Use only hand features for reliability and speed
        self.features_per_hand = 21 * 3  # 21 landmarks * 3 coordinates
        self.total_features = self.features_per_hand * 2  # Both hands = 126 features
    
    def extract_features(self, frame):
        """Extract hand features efficiently"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features = np.zeros(self.total_features, dtype=np.float32)
        
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Extract landmark coordinates
                coords = []
                for landmark in hand_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                
                # Assign to correct hand
                hand_label = handedness.classification[0].label
                if hand_label == "Right":
                    features[0:self.features_per_hand] = coords
                else:  # Left hand
                    features[self.features_per_hand:] = coords
        
        return features

def load_dataset_optimized(data_dir, json_file, max_classes=20, max_videos_per_class=30):
    """Load dataset with optimized parameters"""
    data_path = Path(data_dir)
    
    with open(json_file, 'r') as f:
        wlasl_data = json.load(f)
    
    video_paths = []
    labels = []
    class_counts = {}
    
    # Filter for good classes with enough data
    for entry in wlasl_data:
        gloss = entry['gloss']
        
        # Skip if we have enough classes
        if len(class_counts) >= max_classes and gloss not in class_counts:
            continue
        
        if gloss not in class_counts:
            class_counts[gloss] = 0
        
        # Skip if class has enough videos
        if class_counts[gloss] >= max_videos_per_class:
            continue
        
        # Process instances
        for instance in entry['instances']:
            if instance.get('split') != 'train':
                continue
            
            video_id = instance['video_id']
            video_path = data_path / f"{video_id}.mp4"
            
            if video_path.exists():
                video_paths.append(str(video_path))
                labels.append(gloss)
                class_counts[gloss] += 1
    
    logger.info(f"Dataset: {len(video_paths)} videos, {len(class_counts)} classes")
    logger.info(f"Classes: {list(class_counts.keys())}")
    
    return video_paths, labels

def extract_video_features_optimized(video_path, sequence_length=25):
    """Extract features from video with optimization"""
    extractor = MediaPipeExtractor()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((sequence_length, extractor.total_features), dtype=np.float32)
    
    features_list = []
    frame_count = 0
    skip_frames = 1  # Process every frame for better accuracy
    max_frames = 50  # Reasonable limit
    
    while len(features_list) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % (skip_frames + 1) == 0:
            # Resize for speed but maintain quality
            frame = cv2.resize(frame, (320, 240))
            features = extractor.extract_features(frame)
            features_list.append(features)
        
        frame_count += 1
    
    cap.release()
    
    if len(features_list) == 0:
        return np.zeros((sequence_length, extractor.total_features), dtype=np.float32)
    
    features_array = np.array(features_list, dtype=np.float32)
    
    # Sequence length handling
    if len(features_array) > sequence_length:
        # Uniform sampling
        indices = np.linspace(0, len(features_array)-1, sequence_length, dtype=int)
        features_array = features_array[indices]
    elif len(features_array) < sequence_length:
        # Pad with repetition of last frame
        last_frame = features_array[-1]
        padding_needed = sequence_length - len(features_array)
        padding = np.tile(last_frame, (padding_needed, 1))
        features_array = np.vstack([features_array, padding])
    
    return features_array

def create_gpu_optimized_model(input_shape, num_classes):
    """Create model optimized for GPU training"""
    
    # Use functional API for better GPU utilization
    inputs = Input(shape=input_shape, name='sequence_input')
    
    # Bidirectional LSTM layers - optimized for GPU
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(inputs)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(x)
    
    # Dense layers with proper regularization
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='GPUOptimizedASL')
    
    return model

def train_gpu_model(config):
    """Train model with GPU optimization"""
    logger.info("🚀 Starting GPU-optimized training...")
    
    # Load dataset
    video_paths, labels = load_dataset_optimized(
        config['data_dir'], 
        config['json_file'],
        config['max_classes'],
        config['max_videos_per_class']
    )
    
    if len(video_paths) == 0:
        logger.error("No videos found!")
        return
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    logger.info(f"📊 Processing {len(video_paths)} videos for {num_classes} classes")
    
    # Extract all features (show progress)
    logger.info("🎬 Extracting features from videos...")
    X = []
    y = []
    
    for i, (video_path, label_idx) in enumerate(zip(video_paths, y_encoded)):
        if (i + 1) % 20 == 0 or i == len(video_paths) - 1:
            logger.info(f"Progress: {i+1}/{len(video_paths)} videos processed")
        
        features = extract_video_features_optimized(video_path, config['sequence_length'])
        X.append(features)
        y.append(label_idx)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    logger.info(f"📈 Dataset prepared: {X.shape}")
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    logger.info(f"🔄 Split - Train: {X_train.shape}, Validation: {X_val.shape}")
    
    # Create model
    input_shape = (config['sequence_length'], 126)
    
    # Force model creation on GPU
    with tf.device('/GPU:0'):
        model = create_gpu_optimized_model(input_shape, num_classes)
        
        # Compile with GPU-optimized settings
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    model.summary()
    
    # GPU-optimized callbacks
    callbacks = [
        ModelCheckpoint(
            config['model_save_path'],
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    logger.info("🏋️ Starting GPU training...")
    
    # Train with GPU optimization
    with tf.device('/GPU:0'):
        history = model.fit(
            X_train, y_train,
            batch_size=config['batch_size'],
            epochs=config['num_epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
    
    # Save label encoder
    label_path = config['model_save_path'].replace('.h5', '_labels.pkl')
    with open(label_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Plot results
    plot_training_results(history)
    
    logger.info(f"✅ Training completed! Model saved to {config['model_save_path']}")
    logger.info(f"📝 Labels saved to {label_path}")
    
    return model, label_encoder

def plot_training_results(history):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gpu_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def realtime_gpu_recognition(model_path):
    """Real-time recognition using GPU model"""
    logger.info(f"🔄 Loading GPU model from {model_path}")
    
    # Load model and labels
    model = load_model(model_path)
    
    label_path = model_path.replace('.h5', '_labels.pkl')
    with open(label_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    classes = list(label_encoder.classes_)
    logger.info(f"✅ Model loaded: {len(classes)} classes")
    logger.info(f"🏷️ Classes: {classes}")
    
    # Initialize components
    extractor = MediaPipeExtractor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("❌ Cannot open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Feature buffer
    sequence_length = 25
    feature_buffer = deque(maxlen=sequence_length)
    
    logger.info("🎥 Real-time recognition started")
    logger.info("Controls: 'q' = quit, 's' = save prediction")
    
    prediction_history = deque(maxlen=5)  # Smooth predictions
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("❌ Failed to read frame")
            break
        
        frame = cv2.flip(frame, 1)  # Mirror effect
        
        # Extract features
        features = extractor.extract_features(frame)
        feature_buffer.append(features)
        
        # Make prediction when buffer is full
        if len(feature_buffer) == sequence_length:
            # Prepare input for model
            sequence = np.array(list(feature_buffer))
            input_batch = np.expand_dims(sequence, axis=0)
            
            # GPU prediction
            with tf.device('/GPU:0'):
                predictions = model.predict(input_batch, verbose=0)[0]
            
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]
            predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
            
            # Add to history for smoothing
            prediction_history.append((predicted_class, confidence))
            
            # Get most common recent prediction
            if len(prediction_history) >= 3:
                recent_predictions = [p[0] for p in list(prediction_history)[-3:]]
                most_common = max(set(recent_predictions), key=recent_predictions.count)
                avg_confidence = np.mean([p[1] for p in list(prediction_history)[-3:] if p[0] == most_common])
            else:
                most_common = predicted_class
                avg_confidence = confidence
            
            # Display prediction
            if avg_confidence >= 0.7:
                color = (0, 255, 0)  # Green for high confidence
                status = "CONFIDENT"
            elif avg_confidence >= 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
                status = "UNCERTAIN"
            else:
                color = (0, 100, 255)  # Orange for low confidence
                status = "LOW CONF"
            
            # Main prediction text
            cv2.putText(frame, f"Sign: {most_common}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Confidence: {avg_confidence:.2f} ({status})", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        else:
            # Building sequence
            progress = len(feature_buffer)
            cv2.putText(frame, f"Loading sequence... {progress}/{sequence_length}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Feature activity indicator
        active_features = np.count_nonzero(features)
        cv2.putText(frame, f"Hand features: {active_features}/126", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(frame, "Controls: 'q'=quit, 's'=save", (10, frame.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow("GPU ASL Recognition", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(feature_buffer) == sequence_length:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            logger.info(f"💾 Saved prediction: {most_common} (conf: {avg_confidence:.2f}) at {timestamp}")
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("🏁 Real-time recognition stopped")

def main():
    parser = argparse.ArgumentParser(description='GPU-Optimized Dynamic ASL Recognition')
    parser.add_argument('--mode', choices=['train', 'realtime'], default='train',
                       help='Operation mode')
    parser.add_argument('--data_dir', type=str, default='./dataset/videos/',
                       help='Directory containing video files')
    parser.add_argument('--json_file', type=str, default='./dataset/WLASL_v0.3.json',
                       help='WLASL JSON metadata file')
    parser.add_argument('--model_path', type=str, default='gpu_optimized_asl.h5',
                       help='Path to save/load model')
    
    args = parser.parse_args()
    
    # GPU-optimized configuration
    config = {
        'data_dir': args.data_dir,
        'json_file': args.json_file,
        'sequence_length': 25,
        'batch_size': 64,  # Large batch for GPU efficiency
        'num_epochs': 30,
        'learning_rate': 0.001,
        'max_classes': 20,
        'max_videos_per_class': 30,
        'model_save_path': args.model_path
    }
    
    if args.mode == 'train':
        if not os.path.exists(config['data_dir']):
            logger.error(f"❌ Data directory not found: {config['data_dir']}")
            logger.info("Please update the data_dir path")
            return
        
        if not os.path.exists(config['json_file']):
            logger.error(f"❌ JSON file not found: {config['json_file']}")
            logger.info("Please update the json_file path")
            return
        
        train_gpu_model(config)
        
    elif args.mode == 'realtime':
        if not os.path.exists(args.model_path):
            logger.error(f"❌ Model not found: {args.model_path}")
            logger.info("Please train a model first using --mode train")
            return
        
        realtime_gpu_recognition(args.model_path)

if __name__ == "__main__":
    main()