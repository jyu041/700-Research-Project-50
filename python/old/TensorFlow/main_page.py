import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QProgressBar,
    QVBoxLayout, QHBoxLayout, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, QThread
from helper import *

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        # Top Row: Load Data + Progress Bar
        load_layout = QHBoxLayout()
        self.load_button = QPushButton('Load Data')
        self.load_button.setFixedWidth(100)
        self.load_button.clicked.connect(self.trigger_load_data)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(150)
        load_layout.addWidget(self.load_button, stretch=0)
        load_layout.addWidget(self.progress_bar, stretch=1)
        main_layout.addLayout(load_layout)

        # Second Row: Train + Dropdown
        test_layout = QHBoxLayout()
        self.train_button = QPushButton('Train')
        self.train_button.setFixedWidth(100)
        self.dropdown = QComboBox()
        self.dropdown.setStyleSheet("""
            QComboBox::drop-down {
                padding-right: 10px;
            }
        """)
        self.dropdown.setMinimumWidth(150)
        self.dropdown.addItems(["model1", "model2", "model3"])
        test_layout.addWidget(self.train_button,stretch=0)
        test_layout.addWidget(self.dropdown, stretch=1)
        main_layout.addLayout(test_layout)

        # Train/Test Ratio
        self.ratio_label = QLabel("Train/Test: 80/20")  # this is the label
        self.ratio_slider = ClickableSlider(Qt.Horizontal)  # this is the slider
        self.ratio_slider.setRange(0, 100)
        self.ratio_slider.setValue(80)
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
        self.epoch_slider.setRange(1, 30)
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
        self.adjustSize()

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.animate_progress)

        self.current_progress = 0
        self.max_progress = 100

    def trigger_load_data(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.current_progress = 0
            self.progress_bar.setValue(0)

            self.progress_timer.start(50)  # 50ms tick

            # Setup thread and worker
            self.thread = QThread()
            self.worker = DataLoaderWorker(folder)
            self.worker.moveToThread(self.thread)

            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.on_data_loaded)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def animate_progress(self):
        if self.current_progress < self.max_progress:
            self.current_progress += 1
            self.progress_bar.setValue(self.current_progress)
        else:
            self.progress_timer.stop()
    
    def on_data_loaded(self, dataset):
        self.dataset = dataset
        self.progress_bar.setValue(100)
        self.progress_timer.stop()





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

