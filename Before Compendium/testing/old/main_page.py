import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QProgressBar, QStackedWidget, 
    QVBoxLayout, QHBoxLayout, QComboBox, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QThread
from helper import *
from training_page import *

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.stack = QStackedWidget(self)
        self.main_widget = QWidget()
        self.training_page = TrainingPage(self.stack)
        self.stack.addWidget(self.main_widget)
        self.stack.addWidget(self.training_page)

        self.initUI()
        main_stack_layout = QVBoxLayout(self)
        main_stack_layout.addWidget(self.stack)
        self.setLayout(main_stack_layout)

    def initUI(self):
        main_layout = QVBoxLayout()
        # Top Row: Load Data + Progress Bar
        load_layout = QHBoxLayout()
        self.load_button = QPushButton('Load Data')
        self.load_button.setFixedWidth(100)
        self.load_button.clicked.connect(self.trigger_load_data)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(150)
        self.progress_bar.setValue(0)
        load_layout.addWidget(self.load_button, stretch=0)
        load_layout.addWidget(self.progress_bar, stretch=1)
        main_layout.addLayout(load_layout)

        # Second Row: Train + Dropdown
        test_layout = QHBoxLayout()
        self.train_button = QPushButton('Train')
        self.train_button.setFixedWidth(100)
        self.train_button.clicked.connect(self.show_training_page)
        self.dropdown = QComboBox()
        self.dropdown.setStyleSheet("""
            QComboBox::drop-down {
                padding-right: 10px;
            }
        """)
        self.dropdown.setMinimumWidth(150)
        self.dropdown.addItems(["mobilenet", "model2", "model3"])
        test_layout.addWidget(self.train_button,stretch=0)
        test_layout.addWidget(self.dropdown, stretch=1)
        main_layout.addLayout(test_layout)

        # Train/Test Ratio
        self.ratio_label = QLabel("Train/Test: 80/20")  # this is the label
        self.ratio_slider = ClickableSlider(Qt.Horizontal)  # this is the slider
        self.ratio_slider.setRange(0, 100)
        self.ratio_slider.setValue(80)
        self.train_button.setEnabled(False)
        self.ratio_slider.valueChanged.connect(
            lambda value: update_ratio_label(self.ratio_label, value)
        )
        main_layout.addWidget(self.ratio_label)
        main_layout.addWidget(self.ratio_slider)


        # Batch Size
        self.batch_label = QLabel('Batch Size: 30')
        self.batch_slider = ClickableSlider(Qt.Horizontal)
        self.batch_slider.setRange(10, 50)
        self.batch_slider.setValue(30)
        self.batch_slider.valueChanged.connect(
        lambda value: update_batch_label(self.batch_label, value))
        main_layout.addWidget(self.batch_label)
        main_layout.addWidget(self.batch_slider)

        # Epoch
        self.epoch_label = QLabel('Epoch: 10')
        self.epoch_slider = ClickableSlider(Qt.Horizontal)
        self.epoch_slider.setRange(1, 100)
        self.epoch_slider.setValue(10)
        self.epoch_slider.valueChanged.connect(
        lambda value: update_epoch_label(self.epoch_label, value))
        main_layout.addWidget(self.epoch_label)
        main_layout.addWidget(self.epoch_slider)

        # Test Button
        self.test_button = QPushButton('Test')
        self.test_button.setFixedWidth(200)
        main_layout.addWidget(self.test_button, alignment=Qt.AlignCenter)

        # Set the main layout
        self.setWindowTitle("Main Page")
        self.setLayout(main_layout)
        self.main_widget.setLayout(main_layout)
        self.adjustSize()

    def trigger_load_data(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if not folder:
            return 
        
        self.current_progress = 0
        self.progress_bar.setValue(0)
        self.load_button.setEnabled(False)

        # Setup thread and worker
        self.thread = QThread()
        self.worker = DataLoaderWorker(folder)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_bar.setValue)  # << Add this
        self.worker.finished.connect(self.on_data_loaded)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.worker.finished.connect(lambda: self.load_button.setEnabled(True))

    
    def on_data_loaded(self, data):
        self.images, self.labels, self.class_map = data
        if len(self.images) == 0 or len(self.labels) == 0:
            QMessageBox.critical(self, "Error", "No images found or unable to load.")
            self.train_button.setEnabled(False)
            return
        self.train_button.setEnabled(True)
        self.progress_bar.setValue(100)

    
    def show_training_page(self):
        self.stack.setCurrentWidget(self.training_page)

        epoch_count = self.epoch_slider.value()
        selected_model = self.dropdown.currentText()
        test_ratio = 1 - self.ratio_slider.value() / 100
        batch_size = self.batch_slider.value()

        self.adjustSize()
        QTimer.singleShot(0, lambda: self.training_page.start_training(
            epoch_count, self.images, self.labels,
            selected_model, test_ratio, batch_size
        ))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

