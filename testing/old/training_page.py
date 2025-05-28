# … imports …
from PyQt5.QtCore import Qt, QThread, pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QProgressBar, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from training_worker import *


class TrainingPage(QWidget):
    def __init__(self, parent_stack):
        super().__init__()
        self.parent_stack = parent_stack
        self._build_ui()

    # ------------------------- UI scaffold ---------------------------
    def _build_ui(self):
        v = QVBoxLayout(self)                       # v is now the page’s layout

        # --- status + info ---
        self.status = QLabel("", alignment=Qt.AlignCenter)
        v.addWidget(self.status)

        self.info = QLabel("", alignment=Qt.AlignCenter)
        v.addWidget(self.info)

        self.epoch_bar = QProgressBar()
        self.epoch_bar.setValue(0)
        v.addWidget(self.epoch_bar)

        # --- Matplotlib canvas ---
        self.canvas = FigureCanvas(Figure(figsize=(5, 4)))
        v.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_title("Training Loss and Accuracy")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss/Acc")
        self.loss_line, = self.ax.plot([], [], 'r-', label="Loss")
        self.acc_line , = self.ax.plot([], [], 'b-', label="Accuracy")
        self.ax.legend()

        # --- Back button ---
        back = QPushButton("Back", clicked=self._back)
        back.setFixedWidth(200)
        v.addWidget(back, alignment=Qt.AlignCenter)
        self._back_btn = back


    # --------------------- navigation helpers ------------------------
    def _back(self):
        self.parent_stack.setCurrentIndex(0)

    # ----------------------- public API ------------------------------
    def start_training(self, epochs, X, y, model, test_ratio, batch_size):
        # reset UI
        self.info.setText("")
        self.status.setText("")
        self.epoch_bar.setValue(0)

        # 🔁 Properly reset the plot
        self.ax.clear()
        self.ax.set_title("Training Loss and Accuracy")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss/Acc")
        self.losses, self.accs = [], []

        # ⚠️ Recreate the plot lines *after clearing*
        self.loss_line, = self.ax.plot([], [], 'r-', label="Loss")
        self.acc_line , = self.ax.plot([], [], 'b-', label="Accuracy")
        self.ax.legend()
        self.canvas.draw_idle()

        self._back_btn.setDisabled(True)

        # spin up worker thread
        self._thread = QThread(self)
        self._worker = TrainingWorker(epochs, X, y, model, test_ratio, batch_size)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)

        # connect signals
        self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._worker.batch_percent.connect(self._on_batch, Qt.QueuedConnection)
        self._worker.status.connect(self.status.setText)
        self._worker.done.connect(self._on_done)
        self._worker.done.connect(self._thread.quit)

        self._thread.start()

    # -------------------- slots in GUI thread ------------------------
    @pyqtSlot(int, float, float)
    def _on_progress(self, epoch, loss, acc):
        self.losses.append(loss)
        self.accs.append(acc)

        x_vals = list(range(1, len(self.losses) + 1))
        self.loss_line.set_data(x_vals, self.losses)
        self.acc_line.set_data(x_vals, self.accs)

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

        self.info.setText(f"Loss {loss:.4f}  •  Acc {acc:.4f}")


    @pyqtSlot(list, list)
    def _on_done(self, losses, accs):
        self.status.setText("✅ Training complete")
        self._back_btn.setDisabled(False)

    @pyqtSlot(int, int)
    def _on_batch(self, epoch, pct):
        self.epoch_bar.setValue(pct)
        self.info.setText(f"Epoch {epoch} • {pct}%")
        if pct == 100:
            self.epoch_bar.setFormat(f"Epoch {epoch} complete")
