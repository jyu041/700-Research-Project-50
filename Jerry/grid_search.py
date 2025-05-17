# grid_search.py
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
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import itertools
import seaborn as sns
import pandas as pd
import cv2
import random
import json
import time

# Set random seeds for reproducibility
SEED = 42 # [0, 1, 42]

# Maximum numer of images to load from each class
MAX_IMAGES_PER_CLASS = 100 

# Hyper-Parameter tuning
def get_parameter_grid():
    """Define the grid of parameters to search."""
    param_grid = {
        'batch_size': [16, 32, 64],
        'epochs': [30, 50],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'dropout_conv': [0.1, 0.25, 0.4],
        'dropout_dense': [0.3, 0.5, 0.7],
        'conv_filters': [(32, 64, 128), (64, 128, 256)],
        'kernel_size': [3, 5],
        'dense_units': [256, 512, 1024],
        'use_batch_norm': [True, False],
        'patience': [5, 10],
        'rotation_range': [10, 20],
        'width_shift_range': [0.1, 0.2],
        'height_shift_range': [0.1, 0.2],
        'shear_range': [0.1, 0.2],
        'zoom_range': [0.1, 0.2],
        'horizontal_flip': [True]
    }
    return param_grid


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
    parser = argparse.ArgumentParser(description='Grid Search for CNN hyperparameters')
    parser.add_argument('--output_dir', type=str, default='grid_search_results',
                        help='Directory to save grid search results (default: grid_search_results)')
    parser.add_argument('--max_trials', type=int, default=10,
                        help='Maximum number of trials to run (default: 10)')
    parser.add_argument('--random_search', action='store_true',
                        help='Use random search instead of grid search')
    
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
        
        # Limit number of images per class to X
        image_files = image_files[:MAX_IMAGES_PER_CLASS]  # Only load a maximum of X images per class
        print(f"Loading {len(image_files)} images (max {MAX_IMAGES_PER_CLASS}) from class {class_folder}")
        
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

def analyze_predictions(y_pred, class_names):
    """Analyze the distribution of predictions."""
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_distribution = dict(zip(unique, counts))
    
    print("Prediction distribution:")
    for idx, count in pred_distribution.items():
        if idx < len(class_names):
            print(f"  {class_names[idx]}: {count} predictions")
    
    print(f"Number of classes predicted: {len(unique)} out of {len(class_names)}")

def create_cnn_model(input_shape, num_classes, conv_filters=(32, 64, 128), 
                    kernel_size=3, dropout_conv=0.25, dropout_dense=0.5, 
                    dense_units=512, use_batch_norm=True):
    """
    Create a CNN model architecture with configurable hyperparameters.
    
    Args:
        input_shape: Shape of the input images
        num_classes: Number of output classes
        conv_filters: Tuple of filters for each convolutional block
        kernel_size: Size of convolutional kernels
        dropout_conv: Dropout rate for convolutional layers
        dropout_dense: Dropout rate for dense layers
        dense_units: Number of units in the dense layer
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        A compiled keras model
    """
    model = Sequential()
    
    
    
    # First convolutional block
    model.add(Conv2D(conv_filters[0], (kernel_size, kernel_size), 
                    activation='relu', padding='same', input_shape=input_shape))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Conv2D(conv_filters[0], (kernel_size, kernel_size), 
                    activation='relu', padding='same'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_conv))
    
    # Second convolutional block
    model.add(Conv2D(conv_filters[1], (kernel_size, kernel_size), 
                    activation='relu', padding='same'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Conv2D(conv_filters[1], (kernel_size, kernel_size), 
                    activation='relu', padding='same'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_conv))
    
    # Third convolutional block
    model.add(Conv2D(conv_filters[2], (kernel_size, kernel_size), 
                    activation='relu', padding='same'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Conv2D(conv_filters[2], (kernel_size, kernel_size), 
                    activation='relu', padding='same'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_conv))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_dense))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')) 
    
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

def train_model(X_train, y_train, X_val, y_val, params, trial_dir):
    """
    Train a model with the given parameters.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        params: Dictionary of hyperparameters
        trial_dir: Directory to save model and results
        
    Returns:
        History object, trained model, and validation accuracy
    """
    # Extract parameters
    batch_size = params['batch_size']
    epochs = params['epochs']
    learning_rate = params['learning_rate']
    optimizer_name = params['optimizer']
    dropout_conv = params['dropout_conv']
    dropout_dense = params['dropout_dense']
    conv_filters = params['conv_filters']
    kernel_size = params['kernel_size']
    dense_units = params['dense_units']
    use_batch_norm = params['use_batch_norm']
    patience = params['patience']
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=params.get('rotation_range', 10),
        width_shift_range=params.get('width_shift_range', 0.1),
        height_shift_range=params.get('height_shift_range', 0.1),
        shear_range=params.get('shear_range', 0.1),
        zoom_range=params.get('zoom_range', 0.1),
        horizontal_flip=params.get('horizontal_flip', True),
        fill_mode='nearest'
    )
    
    # Create and compile model
    input_shape = X_train[0].shape
    num_classes = len(np.unique(y_train))
    

    
    model = create_cnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_filters=conv_filters,
        kernel_size=kernel_size,
        dropout_conv=dropout_conv,
        dropout_dense=dropout_dense,
        dense_units=dense_units,
        use_batch_norm=use_batch_norm
    )
    
    optimizer = get_optimizer(optimizer_name, learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_path = os.path.join(trial_dir, 'model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0  # Less verbose
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=0  # Less verbose
    )
    
    # Train model
    print(f"Training model with parameters: {params}")
    
    class_weights = {}
    total_samples = len(y_train)
    unique, counts = np.unique(y_train, return_counts=True)
    
    for i, c in enumerate(unique):
        class_weights[c] = total_samples / (len(unique) * counts[i])
        
        
    reduce_lr = ReduceLROnPlateau(
      monitor='val_loss',
      factor=0.5,
      patience=3,
      min_lr=1e-6,
      verbose=1
    )
    
    history = model.fit(
      datagen.flow(X_train, y_train, batch_size=batch_size),
      steps_per_epoch=len(X_train) // batch_size,
      epochs=epochs,
      validation_data=(X_val, y_val),
      callbacks=[checkpoint, early_stopping, reduce_lr],
      verbose=1,
      class_weight=class_weights  # Add class weights
    )
    
    # Get validation accuracy
    val_accuracy = max(history.history['val_accuracy'])
    
    return history, model, val_accuracy

def evaluate_model(model, X_test, y_test, class_names, trial_dir):
    """
    Evaluate the model and save metrics.
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels
        class_names: List of class names
        trial_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of metrics
    """
    # Evaluate on test set
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Output only major metrics to console
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test F1 score (weighted): {f1:.4f}")
    
    # Get detailed classification report
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    # Plot and save confusion matrix
    cm_path = os.path.join(trial_dir, 'confusion_matrix.png')
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    
    # Prepare metrics dictionary
    metrics_dict = {
        'accuracy': [accuracy],
        'f1_score': [f1],
        **{f"{class_name}_precision": [class_report[class_name]['precision']] for class_name in class_names},
        **{f"{class_name}_recall": [class_report[class_name]['recall']] for class_name in class_names},
        **{f"{class_name}_f1-score": [class_report[class_name]['f1-score']] for class_name in class_names},
        **{f"{class_name}_support": [class_report[class_name]['support']] for class_name in class_names}
    }
    
    # Save metrics
    metrics_path = os.path.join(trial_dir, 'metrics.csv')
    df = pd.DataFrame(metrics_dict)
    df.to_csv(metrics_path, index=False)
    analyze_predictions(y_pred, class_names)
    return metrics_dict

def plot_training_history(history, trial_dir):
    """Plot and save training history."""
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
    plt.savefig(os.path.join(trial_dir, 'training_history.png'))
    plt.close()



def random_param_selection(param_grid, count):
    """
    Randomly select parameter combinations.
    
    Args:
        param_grid: Dictionary of parameter lists
        count: Number of combinations to select
        
    Returns:
        List of parameter dictionaries
    """
    all_keys = list(param_grid.keys())
    selected_params = []
    
    for _ in range(count):
        param_dict = {}
        for key in all_keys:
            param_dict[key] = random.choice(param_grid[key])
        selected_params.append(param_dict)
    
    return selected_params

def generate_parameter_combinations(param_grid, max_trials, random_search=False):
    """
    Generate parameter combinations for grid search.
    
    Args:
        param_grid: Dictionary of parameter lists
        max_trials: Maximum number of trials to run
        random_search: Whether to use random search
        
    Returns:
        List of parameter dictionaries
    """
    if random_search:
        return random_param_selection(param_grid, max_trials)
    
    # For full grid search, calculate total combinations
    total_combinations = 1
    for key, values in param_grid.items():
        total_combinations *= len(values)
    
    print(f"Total parameter combinations: {total_combinations}")
    
    if total_combinations <= max_trials:
        # Generate all combinations if under max_trials
        keys = param_grid.keys()
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    else:
        # Otherwise use random selection
        print(f"Too many combinations. Using random selection of {max_trials} combinations.")
        return random_param_selection(param_grid, max_trials)

def save_results_summary(results, output_dir):
    """
    Save a summary of all grid search results.
    
    Args:
        results: List of dictionaries with trial results
        output_dir: Directory to save summary
    """
    # Convert to DataFrame for easy manipulation
    results_df = pd.DataFrame(results)
    
    # Sort by validation accuracy
    results_df = results_df.sort_values('val_accuracy', ascending=False)
    
    # Save as CSV
    results_df.to_csv(os.path.join(output_dir, 'grid_search_results.csv'), index=False)
    
    # Plot top 10 parameter combinations
    plt.figure(figsize=(12, 6))
    top_results = results_df.head(10)
    sns.barplot(x=top_results.index, y='val_accuracy', data=top_results)
    plt.title('Top 10 Parameter Combinations by Validation Accuracy')
    plt.xlabel('Trial Index')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_10_combinations.png'))
    
    # Create visualizations for parameter importance
    create_parameter_importance_plots(results_df, output_dir)
    
def create_parameter_importance_plots(results_df, output_dir):
    """
    Create plots showing the effect of different parameters on validation accuracy.
    
    Args:
        results_df: DataFrame with grid search results
        output_dir: Directory to save plots
    """
    # List of parameters to analyze individually
    params_to_analyze = [
        'batch_size', 'learning_rate', 'optimizer', 'dropout_conv', 
        'dropout_dense', 'dense_units', 'use_batch_norm'
    ]
    
    # Create a figure with subplots for each parameter
    n_params = len(params_to_analyze)
    n_cols = 2
    n_rows = (n_params + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4 * n_rows))
    
    for i, param in enumerate(params_to_analyze):
        plt.subplot(n_rows, n_cols, i + 1)
        
        if results_df[param].dtype == bool:
            # Boolean parameters
            sns.boxplot(x=param, y='val_accuracy', data=results_df)
        elif results_df[param].dtype == object and not results_df[param].str.contains(',').any():
            # Categorical parameters (not tuples)
            sns.boxplot(x=param, y='val_accuracy', data=results_df)
        else:
            # Numerical parameters or tuples
            if isinstance(results_df[param].iloc[0], tuple):
                # Convert tuples to strings for grouping
                results_df[f'{param}_str'] = results_df[param].astype(str)
                sns.boxplot(x=f'{param}_str', y='val_accuracy', data=results_df)
            else:
                # Plot correlation for numerical parameters
                sns.scatterplot(x=param, y='val_accuracy', data=results_df)
                
        plt.title(f'Effect of {param} on Validation Accuracy')
        plt.xlabel(param)
        plt.ylabel('Validation Accuracy')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_importance.png'))
    plt.close()

def main():
    args = parse_arguments()
    
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'dataset', 'set_1', 'asl_alphabet_train')
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, args.output_dir, f"grid_search_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Results will be saved to: {output_dir}")
    
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
    
    # Define parameter grid
    param_grid = get_parameter_grid()
    
    # Generate parameter combinations
    param_combinations = generate_parameter_combinations(
        param_grid, 
        args.max_trials, 
        args.random_search
    )
    
    print(f"Running {len(param_combinations)} parameter combinations")
    
    # Save parameter combinations
    with open(os.path.join(output_dir, 'parameter_combinations.json'), 'w') as f:
        json.dump(param_combinations, f, indent=2, default=str)
    
    # Run grid search
    results = []
    
    for trial_idx, params in enumerate(param_combinations):
        print(f"\n==== Trial {trial_idx+1}/{len(param_combinations)} ====")
        
        # Create trial directory
        trial_dir = os.path.join(output_dir, f"trial_{trial_idx:03d}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Save parameters
        with open(os.path.join(trial_dir, 'params.json'), 'w') as f:
            json.dump(params, f, indent=2, default=str)
        
        try:
            # Train model
            start_time = datetime.datetime.now()
            history, model, val_accuracy = train_model(
                X_train, y_train, X_val, y_val, params, trial_dir
            )
            
            # Plot training history
            plot_training_history(history, trial_dir)
            
            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test, class_names, trial_dir)
            
            # Calculate training time
            end_time = datetime.datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Record results
            trial_results = {
                'trial_idx': trial_idx,
                'val_accuracy': val_accuracy,
                'test_accuracy': metrics['accuracy'][0],
                'f1_score': metrics['f1_score'][0],
                'training_time': training_time,
                **params
            }
            
            results.append(trial_results)
            
            # Print summary
            print(f"Trial {trial_idx} completed in {training_time:.1f} seconds")
            print(f"Validation accuracy: {val_accuracy:.4f}")
            print(f"Test accuracy: {metrics['accuracy'][0]:.4f}")
            print(f"Results saved to {trial_dir}")
            
        except Exception as e:
            print(f"Error in trial {trial_idx}: {str(e)}")
            # Record failure
            results.append({
                'trial_idx': trial_idx,
                'val_accuracy': 0,
                'test_accuracy': 0,
                'f1_score': 0,
                'training_time': 0,
                'error': str(e),
                **params
            })
    
    # Save overall results
    save_results_summary(results, output_dir)
    
    # Find best parameters
    best_trial = max(results, key=lambda x: x['val_accuracy'])
    print("\n==== Grid Search Completed ====")
    print(f"Best validation accuracy: {best_trial['val_accuracy']:.4f}")
    print(f"Best test accuracy: {best_trial['test_accuracy']:.4f}")
    print("Best parameters:")
    for key, value in best_trial.items():
        if key not in ['trial_idx', 'val_accuracy', 'test_accuracy', 'f1_score', 'training_time']:
            print(f"  {key}: {value}")
    
    print(f"\nAll results saved to {output_dir}")
    save_path = os.path.join(output_dir, f"trial_{best_trial['trial_idx']:03d}")
    print(f"Best model saved to {save_path}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
# Basic usage (will run 10 random trials by default)
# python grid_search.py

# Run full grid search with more trials
# python grid_search.py --max_trials 50

# Force random search even with fewer combinations
# python grid_search.py --random_search

# Specify custom output directory
# python grid_search.py --output_dir my_grid_search_results