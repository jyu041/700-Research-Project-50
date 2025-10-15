import os, random, datetime, argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ───────────────── reproducibility ─────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ──────────────────────── CLI ──────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser("ASL classifier – no Albumentations")
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--optimizer",   choices=["adam", "sgd", "rmsprop"], default="adam")
    p.add_argument("--patience",    type=int,   default=6)
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--img_cap",     type=int,   default=1000, help="max images / class")
    p.add_argument("--model_name",  default="MobileNetASL")
    p.add_argument("--data_root",
        default=r"C:\Users\Tony\Downloads\compsys700\700-Research-Project-50\dataset\all",
        help="root folder with class sub-dirs")
    return p.parse_args()

# ───────────────────── scan disk (paths only) ─────────────────────────────
def list_files(root, img_cap):
    paths, labels, class_names = [], [], []
    for cls_idx, cls in enumerate(sorted(os.listdir(root))):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        class_names.append(cls)
        files = [os.path.join(d, f) for f in os.listdir(d)
                 if f.lower().endswith(('.jpg', '.png'))]
        random.shuffle(files)
        files = files[:img_cap]
        paths.extend(files); labels.extend([cls_idx]*len(files))
    paths, labels = np.array(paths), np.array(labels)
    tr_val_p, tst_p, tr_val_l, tst_l = train_test_split(
        paths, labels, test_size=0.15, random_state=SEED, stratify=labels)
    tr_p, val_p, tr_l, val_l = train_test_split(
        tr_val_p, tr_val_l, test_size=0.15/0.85, random_state=SEED,
        stratify=tr_val_l)
    return tr_p, tr_l, val_p, val_l, tst_p, tst_l, class_names

# ───────────────────── simple on-the-fly aug ──────────────────────────────
def augment_image(img):
    # flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    # rotate ±20°
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2),
                                angle, 1.0)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                         borderMode=cv2.BORDER_REFLECT_101)
    # brightness / contrast jitter
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)   # contrast
        beta  = random.uniform(-15, 15)    # brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

class PathSequence(Sequence):
    def __init__(self, paths, labels, bs, img_size,
                 augment=False, shuffle=True):
        self.paths, self.labels = np.array(paths), np.array(labels)
        self.bs = bs; self.sz = img_size
        self.augment = augment; self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self): return len(self.paths)//self.bs
    def on_epoch_end(self):
        if self.shuffle:
            idx = np.arange(len(self.paths)); np.random.shuffle(idx)
            self.paths, self.labels = self.paths[idx], self.labels[idx]
    def __getitem__(self, idx):
        sl = slice(idx*self.bs, (idx+1)*self.bs)
        imgs = [self._read(p) for p in self.paths[sl]]
        return np.stack(imgs), self.labels[sl]
    def _read(self, path):
        img = cv2.imread(path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.sz, self.sz))
        if self.augment: img = augment_image(img)
        return img.astype(np.float32)/255.0

# ───────────────────────── model ───────────────────────────────────────────
def build_model(img_size, n_cls):
    base = MobileNetV2(include_top=False, weights="imagenet",
                       input_tensor=Input(shape=(img_size, img_size, 3)))
    base.trainable = True
    for l in base.layers[:20]: l.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    out = Dense(n_cls, activation="softmax")(x)
    return Model(base.input, out)

def get_opt(name, lr):
    return {"adam": Adam, "sgd": lambda l: SGD(l, momentum=0.9),
            "rmsprop": RMSprop}[name](learning_rate=lr)

# ───────────────────────── helpers ─────────────────────────────────────────
def save_cm(y_true, y_pred, names, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names)
    plt.xlabel('Pred'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.tight_layout(); plt.savefig(path); plt.close()

# ─────────────────────────── main ──────────────────────────────────────────
def main():
    args = cli()
    run_tag   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("Models", f"{args.model_name}_{run_tag}")
    os.makedirs(model_dir, exist_ok=True)

    tr_p, tr_l, val_p, val_l, tst_p, tst_l, classes = \
        list_files(args.data_root, args.img_cap)

    # preload small val/test sets
    def load(paths):
        arr=[]
        for p in paths:
            img=cv2.imread(p); img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(args.img_size,args.img_size))
            arr.append(img.astype(np.float32)/255.0)
        return np.array(arr)
    X_val  = load(val_p);  X_test = load(tst_p)

    train_gen = PathSequence(tr_p,tr_l,args.batch_size,args.img_size,
                             augment=True,shuffle=True)

    # class weights
    cw = compute_class_weight('balanced',
            classes=np.unique(tr_l), y=tr_l)
    class_weight = dict(enumerate(cw))

    # choose loss (old TF fallback)
    try:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                      label_smoothing=0.1)
    except TypeError:
        print("⚠️  TF too old for label_smoothing – disabled.")
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model = build_model(args.img_size, len(classes))
    model.compile(optimizer=get_opt(args.optimizer, args.learning_rate),
                  loss=loss_fn, metrics=['accuracy'])
    model.summary()

    ckpt = os.path.join(model_dir,"best.h5")
    cbs = [ ModelCheckpoint(ckpt, monitor='val_loss',
                            save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=args.patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2,
                              min_lr=1e-6, verbose=1) ]

    print("🚀 Training …")
    hist = model.fit(train_gen,
                     steps_per_epoch=len(train_gen),
                     epochs=args.epochs,
                     validation_data=(X_val, val_l),
                     class_weight=class_weight,
                     callbacks=cbs, verbose=1)

    # quick fine-tune (optional – skip if RAM/GPU tight)
    model.compile(optimizer=Adam(1e-5), loss=loss_fn, metrics=['accuracy'])
    model.fit(train_gen, steps_per_epoch=len(train_gen), epochs=10,
              validation_data=(X_val,val_l), class_weight=class_weight,
              callbacks=cbs, verbose=1)

    print("Evaluating …")
    y_pred = np.argmax(model.predict(X_test,batch_size=args.batch_size),axis=1)
    acc=accuracy_score(tst_l,y_pred); f1=f1_score(tst_l,y_pred,average='weighted')
    print(f"Test acc {acc:.4f}  |  F1 {f1:.4f}")
    print(classification_report(tst_l,y_pred,target_names=classes))

    save_cm(tst_l,y_pred,classes,os.path.join(model_dir,"cm.png"))
    pd.DataFrame(hist.history).to_csv(os.path.join(model_dir,"history.csv"),
                                      index=False)
    print(f"✔️  Saved to {model_dir}")

if __name__=="__main__":
    main()
