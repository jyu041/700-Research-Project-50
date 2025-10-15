import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mobilenet import build_model
import tkinter as tk
from tkinter import filedialog

# Folder selection dialog
tk.Tk().withdraw()

def main():
    print("📁 Please select your dataset folder...")
    dataset_path = filedialog.askdirectory(title="Select Dataset Folder")

    if not dataset_path:
        print("❌ No folder selected.")
        return

    print("🔍 Scanning dataset...")

    # Create ImageDataGenerator with train/validation split
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    # Image settings
    img_height, img_width = 256, 256
    batch_size = 30

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = train_gen.num_classes
    print(f"✅ Classes detected: {num_classes}")

    print("🔧 Building MobileNet...")
    model = build_model((img_height, img_width, 3), num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("🚀 Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        verbose=1
    )

    print("📊 Evaluating model...")
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"\n✅ Final accuracy: {acc:.4f}")
    print(f"Final loss:       {loss:.4f}")

if __name__ == "__main__":
    main()
