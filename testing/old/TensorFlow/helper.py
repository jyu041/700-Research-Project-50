from PyQt5.QtWidgets import QSlider, QStyle, QFileDialog
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
import tensorflow as tf

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

def load_dataset(path, img_height=64, img_width=64, batch_size=32):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    return dataset

def handle_load_data(progress_bar, parent=None):
    folder = QFileDialog.getExistingDirectory(parent, "Select Dataset Folder")
    dataset = None
    if folder:
        progress_bar.setValue(10)
        dataset = load_dataset(folder)
        progress_bar.setValue(100)
    return dataset

class DataLoaderWorker(QObject):
    finished = pyqtSignal(object)  # sends the loaded dataset when done

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        dataset = load_dataset(self.folder)
        self.finished.emit(dataset)