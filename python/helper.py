from PyQt5.QtWidgets import QSlider, QStyle, QFileDialog
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
import os, cv2
import numpy as np

class ClickableSlider(QSlider):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            val = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(),
                event.x(), self.width()
            )
            self.setValue(val)
            event.accept()
        super().mousePressEvent(event)


def update_ratio_label(label, value):
    test = 100 - value
    label.setText(f"Train/Test Ratio: {value}/{test}")

def update_batch_label(label, value):
    label.setText(f"Batch Size: {value}")

def update_epoch_label(label, value):
    label.setText(f"Epochs: {value}")

def load_images_with_opencv(base_path, img_height=64, img_width=64):
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    class_map = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(base_path, class_name)
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue  # skip unreadable images
                img = cv2.resize(img, (img_width, img_height))
                images.append(img)
                labels.append(class_map[class_name])

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_map

class DataLoaderWorker(QObject):
    progress = pyqtSignal(int)              # New signal for progress %
    finished = pyqtSignal(object)           # Emits (images, labels, class_map)

    def __init__(self, folder, img_height=64, img_width=64):
        super().__init__()
        self.folder = folder
        self.img_height = img_height
        self.img_width = img_width

    def run(self):
        images = []
        labels = []
        class_names = sorted([d for d in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, d))])
        class_map = {name: idx for idx, name in enumerate(class_names)}

        all_files = []
        for class_name in class_names:
            class_path = os.path.join(self.folder, class_name)
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append((class_name, os.path.join(class_path, filename)))

        total = len(all_files)
        for i, (class_name, img_path) in enumerate(all_files):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (self.img_width, self.img_height))
            images.append(img)
            labels.append(class_map[class_name])
            self.progress.emit(int((i + 1) / total * 100))  # Emit real-time progress %

        images = np.array(images)
        labels = np.array(labels)
        self.finished.emit((images, labels, class_map))