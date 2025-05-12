from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np, importlib, tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback


class TrainingWorker(QObject):
    # ---------------------------- signals ----------------------------
    progress = pyqtSignal(int, float, float)      # epoch, loss, acc
    batch_percent = pyqtSignal(int, int)
    status   = pyqtSignal(str)
    done     = pyqtSignal(list, list)             # full curves
    # -----------------------------------------------------------------

    def __init__(self, max_epochs, images, labels,
                 model_name, test_ratio, batch_size):
        super().__init__()
        self.max_epochs = max_epochs
        self.images     = images.astype("float32") / 255.0
        self.labels     = labels
        self.model_name = model_name
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.losses, self.accs = [], []

    # ---------------------------- worker entry -----------------------
    def run(self):
        try:
            self.status.emit("🧹 Pre-processing")
            y = to_categorical(self.labels,
                               len(np.unique(self.labels)))

            # shuffle + split
            idx = np.random.permutation(len(self.images))
            split = int(len(idx) * (1 - self.test_ratio))
            x_tr, x_te = self.images[idx[:split]], self.images[idx[split:]]
            y_tr, y_te = y[idx[:split]], y[idx[split:]]

            self.status.emit(f"⚙️ Building {self.model_name}")
            model = importlib.import_module(
                     f"models.{self.model_name.lower()}").build_model(
                     (64, 64, 3), y.shape[1])

            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            steps_per_epoch = int(np.ceil(len(x_tr) / self.batch_size))

            # ---------- epoch callback (GUI-safe) ----------
            class EpochCallback(Callback):
                def __init__(self, outer): super().__init__(); self.o = outer
                def on_epoch_end(self, ep, logs=None):
                    loss = logs.get('loss', 0.0)
                    acc  = logs.get('accuracy', 0.0)
                    self.o.losses.append(loss)
                    self.o.accs.append(acc)
                    self.o.progress.emit(ep + 1, loss, acc)

            class BatchCallback(Callback):                       
                def __init__(self, outer, steps):
                    super().__init__(); self.o = outer; self.steps = steps
                def on_epoch_begin(self, epoch, logs=None):
                    self.curr_epoch = epoch
                def on_train_batch_end(self, batch, logs=None):
                    pct = int((batch + 1) / self.steps * 100)
                    self.o.batch_percent.emit(self.curr_epoch + 1, pct)

            self.status.emit("🚀 Training …")
            model.fit(
                x_tr, y_tr,
                validation_data=(x_te, y_te),
                epochs=self.max_epochs,
                batch_size=self.batch_size,
                verbose=0,
                callbacks=[EpochCallback(self),
                        BatchCallback(self, steps_per_epoch)] 
            )

            self.done.emit(self.losses, self.accs)

        except Exception as e:
            self.status.emit(f"❌ {e}")
            self.done.emit([], [])
