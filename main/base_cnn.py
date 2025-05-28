# python base_cnn.py --batch_size 64 --epochs 50 --learning_rate 0.0005 --optimizer sgd --patience 10 --model_name CustomASLModel

import os
import sys
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import cv2
import random

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is available and will be used for training")
else:
    print("No GPU available, using CPU for training")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CNN model for ASL classification')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of epochs to train (default: 30)')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'sgd', 'rmsprop'], 
                        help='Optimizer for training (default: adam)')
    parser.add_argument('--patience', type=int, default=5, 
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--model_name', type=str, default='ASL_CNN', 
                        help='Model name (default: ASL_CNN)')
    
    return parser.parse_args()


def load_data(data_dir):
    """Load and preprocess the image data."""
    images = []
    labels = []
    class_names = []
    
    # Get class folders
    class_folders = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_names = class_folders
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    
    # Load images from each class folder
    for class_idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_folder)
        print(f"Loading class {class_folder} ({class_idx})...")
        
        image_files = os.listdir(class_path)
        print(f"Found {len(image_files)} images in class {class_folder}")
        
        # Limit number of images per class to 100
        image_files = image_files[:100]  # Only load a maximum of 100 images per class
        print(f"Loading {len(image_files)} images (max 100) from class {class_folder}")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = cv2.resize(img, (64, 64))  # Resize to a consistent size
                img = img / 255.0  # Normalize to [0,1]
                
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
    
    return np.array(images), np.array(labels), class_names

def create_cnn_model(input_shape, num_classes):
    """Create a CNN model architecture."""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def get_optimizer(optimizer_name, learning_rate):
    """Get the specified optimizer with learning rate."""
    if optimizer_name.lower() == 'adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return cm

def save_metrics(metrics_dict, save_path):
    """Save metrics to CSV file."""
    df = pd.DataFrame(metrics_dict)
    df.to_csv(save_path, index=False)
    return df

def main():
    args = parse_arguments()
    
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'dataset', 'set_1', 'asl_alphabet_train')
    
    # Create timestamp for saving model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(project_root, 'Models', f"{args.model_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Model will be saved to: {model_dir}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, class_names = load_data(data_dir)
    
    # Data splitting (70% train, 15% validation, 15% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15/0.85, random_state=SEED, stratify=y_train_val
    )
    
    print(f"Dataset sizes: Train={X_train.shape[0]}, Validation={X_val.shape[0]}, Test={X_test.shape[0]}")
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create and compile model
    input_shape = X_train[0].shape
    num_classes = len(class_names)
    print(f"Input shape: {input_shape}, Number of classes: {num_classes}")
    
    model = create_cnn_model(input_shape, num_classes)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    checkpoint_path = os.path.join(model_dir, 'model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=args.patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        steps_per_epoch=len(X_train) // args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save model
    model_path = os.path.join(model_dir, 'model.h5')
    tf.keras.models.save_model(model, model_path)
    
    # Save as .pth format for compatibility (TensorFlow doesn't use .pth natively, 
    # but we'll create an empty file to match the required directory structure)
    with open(os.path.join(model_dir, 'model.pth'), 'w') as f:
        f.write('TensorFlow model saved as .h5 format. This is a placeholder file.')
    
    # Evaluate on test set
    print("Evaluating model...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test F1 score (weighted): {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot and save confusion matrix
    cm_path = os.path.join(model_dir, 'confusion_matrix.png')
    cm = plot_confusion_matrix(y_test, y_pred, class_names, cm_path)
    
    # Save metrics
    metrics_dict = {
        'accuracy': [accuracy],
        'f1_score': [f1],
        **{f"{class_name}_precision": [class_report[class_name]['precision']] for class_name in class_names},
        **{f"{class_name}_recall": [class_report[class_name]['recall']] for class_name in class_names},
        **{f"{class_name}_f1-score": [class_report[class_name]['f1-score']] for class_name in class_names},
        **{f"{class_name}_support": [class_report[class_name]['support']] for class_name in class_names}
    }
    
    metrics_path = os.path.join(model_dir, 'metrics.csv')
    save_metrics(metrics_dict, metrics_path)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.close()
    
    print(f"Model and results saved to {model_dir}")

if __name__ == "__main__":
    main()