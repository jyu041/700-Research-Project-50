
# train_fast.py
# -----------------------------------------------------------
# LSA64 (cut) trainer — faster throughput on Windows
# - Sequential OpenCV sampling (no per-frame cap.set() seeks)
# - Optional decord; TVVideoReader as last resort
# - torch.amp autocast/GradScaler
# - Micro-batch + grad accumulation
# - 2 DataLoader workers by default (tweak at top)
# -----------------------------------------------------------

import os
import re
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Optional decoders/readers
HAS_DECORD = False
try:
    import decord  # pip install decord
    from decord import VideoReader, cpu
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

HAS_CV2 = False
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from torchvision.io import VideoReader as TVVideoReader
    HAS_TVVR = True
except Exception:
    HAS_TVVR = False

# =========================
# ====== CONFIG AREA ======
# =========================

CUT_DIR = "./datasets/lsa64_cut/all_cut"
OUTPUT_DIR = "models"
SEED = 42

# Subject-held-out
N_VAL_ACTORS = 1
N_TEST_ACTORS = 1
ACTORS_VAL = []
ACTORS_TEST = []

# Video sampling & preprocessing
NUM_FRAMES = 16                # 16
FRAME_SIZE = 144               # 144
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Augmentations
USE_RANDOM_RESIZED_CROP = False # False
RANDOM_CROP_SCALE = (0.75, 1.0)
RANDOM_CROP_RATIO = (3/4, 4/3)
COLOR_JITTER = None            # None
H_FLIP = False

# Training (speed-tuned)
NUM_EPOCHS = 30                # 30
MICRO_BATCH_SIZE = 8           # 8
ACCUM_STEPS = 8                # 8
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0
USE_MIXED_PRECISION = True
FREEZE_CNN_EPOCHS = 0

# Transformer head
TRANSFORMER_DIM = 256
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_DROPOUT = 0.1
USE_CLS_TOKEN = True
USE_GRAD_CHECKPOINTING = False  # speed up

# DataLoader
NUM_WORKERS = 2                 # 2
PREFETCH_FACTOR = 2             # 2
PERSISTENT_WORKERS = True
PIN_MEMORY = True

# Checkpointing
SAVE_BEST_ONLY = True
SAVE_EVERY_N_EPOCHS = None

# =========================
# ===== END OF CONFIG =====
# =========================

FNAME_RE = re.compile(r"^(?P<cls>\d{3})_(?P<actor>\d{3})_(?P<rep>\d{3})\.(?:mp4|avi|mov|mkv)$", re.IGNORECASE)

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def scan_files(cut_dir: str):
    items = []
    for f in Path(cut_dir).iterdir():
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            m = FNAME_RE.match(f.name)
            if not m:
                continue
            items.append({
                "path": str(f),
                "cls": m.group("cls"),
                "actor": m.group("actor"),
                "rep": m.group("rep"),
            })
    classes = sorted({x["cls"] for x in items})
    actors = sorted({x["actor"] for x in items})
    return items, classes, actors

def actor_split(items, actors, n_val, n_test, actors_val_override=None, actors_test_override=None):
    actors_sorted = list(actors)
    if actors_val_override:
        val = sorted(set(actors_val_override))
    else:
        val = actors_sorted[-n_val:] if n_val > 0 else []
    rem = [a for a in actors_sorted if a not in val]
    if actors_test_override:
        test = sorted(set(actors_test_override))
    else:
        test = rem[-n_test:] if n_test > 0 else []
    train = [a for a in actors_sorted if a not in set(val + test)]

    def pick(actor_list):
        aset = set(actor_list)
        return [x for x in items if x["actor"] in aset]

    return pick(train), pick(val), pick(test), train, val, test

def uniform_indices(n_total: int, n_samples: int):
    if n_total <= 0:
        return [0] * n_samples
    if n_samples <= 1:
        return [min(n_total - 1, 0)]
    return [min(n_total - 1, int(round(x)))
            for x in torch.linspace(0, n_total - 1, steps=n_samples).tolist()]

# ---------- Readers ----------

def sample_frames_decord(path: str, num_frames: int) -> torch.Tensor:
    vr = VideoReader(path, ctx=cpu(0))
    n_total = len(vr)
    idxs = uniform_indices(n_total, num_frames)
    frames = vr.get_batch(idxs)  # decord NDArray, not a numpy array
    frames = frames.asnumpy()    # convert to np.ndarray
    return torch.from_numpy(frames)  # [T,H,W,3] uint8

def sample_frames_cv2_seq(path: str, num_frames: int) -> torch.Tensor:
    # Sequential read, collect only target indices (fast; no random seeks)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n_total <= 0:
        cap.release()
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    idxs = uniform_indices(n_total, num_frames)
    want = set(idxs)
    frames = []
    i = 0
    ok = True
    while ok and len(frames) < len(idxs):
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        if i in want:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(frame_rgb))
        i += 1
    cap.release()
    if len(frames) == 0:
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    # pad if needed
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    return torch.stack(frames, dim=0)

def sample_frames_tvvideoreader(path: str, num_frames: int) -> torch.Tensor:
    if not HAS_TVVR:
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    try:
        vr = TVVideoReader(path, "video")
    except Exception:
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    seq = []
    for frame in vr:
        img = frame["data"].permute(1, 2, 0).contiguous()  # [H,W,C] uint8
        seq.append(img)
    if len(seq) == 0:
        return torch.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=torch.uint8)
    take = uniform_indices(len(seq), num_frames)
    frames = []
    for k in take:
        img = seq[k]
        img = torch.nn.functional.interpolate(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            size=(FRAME_SIZE, FRAME_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).byte()
        frames.append(img)
    return torch.stack(frames, dim=0)

def sample_frames(path: str, num_frames: int) -> torch.Tensor:
    if HAS_DECORD:
        return sample_frames_decord(path, num_frames)
    if HAS_CV2:
        return sample_frames_cv2_seq(path, num_frames)
    return sample_frames_tvvideoreader(path, num_frames)

# ---------- Transforms ----------

class VideoToTensor:
    def __call__(self, video_frames: torch.Tensor) -> torch.Tensor:
        if video_frames.dtype != torch.float32:
            video_frames = video_frames.float() / 255.0
        return video_frames.permute(0, 3, 1, 2)

class ApplyPerFrame:
    def __init__(self, tfm): self.tfm = tfm
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.tfm(xi) for xi in x], dim=0)

def build_transforms(train: bool):
    img = []
    if train and USE_RANDOM_RESIZED_CROP:
        img.append(transforms.RandomResizedCrop(FRAME_SIZE, scale=RANDOM_CROP_SCALE,
                                                ratio=RANDOM_CROP_RATIO,
                                                interpolation=InterpolationMode.BICUBIC))
    else:
        img += [transforms.Resize(FRAME_SIZE, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(FRAME_SIZE)]
    if train and COLOR_JITTER:
        b, c, s, h = COLOR_JITTER
        img.append(transforms.ColorJitter(b, c, s, h))
    if train and H_FLIP:
        img.append(transforms.RandomHorizontalFlip(0.5))
    img.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]))
    return transforms.Compose([VideoToTensor(), ApplyPerFrame(transforms.Compose(img))])

# ---------- Dataset ----------

class LSA64CutFlatDataset(Dataset):
    def __init__(self, items: List[Dict], class_to_idx: Dict[str,int], num_frames: int, train: bool):
        self.items = items
        self.class_to_idx = class_to_idx
        self.num_frames = num_frames
        self.tfm = build_transforms(train=train)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx: int):
        it = self.items[idx]
        frames = sample_frames(it["path"], self.num_frames)    # [T,H,W,3] uint8
        frames = self.tfm(frames)                               # [T,3,S,S] float
        label = self.class_to_idx[it["cls"]]
        return frames, label

def collate_batch(batch):
    vids, labels = zip(*batch)
    return torch.stack(vids, 0), torch.tensor(labels, dtype=torch.long)

# ---------- Model ----------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(1))
    def forward(self, x):  # x: [L,B,D]
        return x + self.pe[:x.size(0)]

class CheckpointedEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, activation: str, norm_first: bool,
                 num_layers: int, use_ckpt: bool):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation=activation, batch_first=False, norm_first=norm_first
            ) for _ in range(num_layers)
        ])
        self.use_ckpt = use_ckpt
    def _ckpt(self, fn, *args):
        try:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        except TypeError:
            return torch.utils.checkpoint.checkpoint(fn, *args)
    def forward(self, src):
        out = src
        for layer in self.layers:
            if self.use_ckpt and self.training:
                out = self._ckpt(layer, out)
            else:
                out = layer(out)
        return out

class VideoTransformerClassifier(nn.Module):
    def __init__(self, num_classes: int, d: int = TRANSFORMER_DIM, n_layers: int = TRANSFORMER_LAYERS,
                 n_heads: int = TRANSFORMER_HEADS, dropout: float = TRANSFORMER_DROPOUT,
                 use_cls: bool = USE_CLS_TOKEN, use_ckpt: bool = USE_GRAD_CHECKPOINTING):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])   # -> [B,512,1,1]
        self.proj = nn.Linear(512, d)
        self.use_cls = use_cls
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1,1,d))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos = PositionalEncoding(d, max_len=2000)
        self.enc = CheckpointedEncoder(
            d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout,
            activation="gelu", norm_first=True, num_layers=n_layers, use_ckpt=use_ckpt
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, num_classes)
    def forward(self, video):
        B,T,C,H,W = video.shape
        x = video.view(B*T, C, H, W)
        feat = self.cnn(x).flatten(1)
        feat = self.proj(feat).view(B, T, -1)
        seq = feat.transpose(0,1)
        if self.use_cls:
            cls = self.cls_token.expand(-1, B, -1)
            seq = torch.cat([cls, seq], dim=0)
        seq = self.pos(seq)
        out = self.enc(seq)
        pooled = out[0] if self.use_cls else out.mean(dim=0)
        return self.head(self.drop(pooled))

# ---------- Utils ----------

def cosine_with_warmup(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    progress = (step - warmup) / float(max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using DECORD: {HAS_DECORD} | OpenCV: {HAS_CV2} | TVVideoReader: {HAS_TVVR}")
    print("Scanning files...")
    items, classes, actors = scan_files(CUT_DIR)
    assert len(items) > 0, f"No videos found in {CUT_DIR}"
    classes_sorted = sorted(classes)
    class_to_idx = {c:i for i,c in enumerate(classes_sorted)}

    train_items, val_items, test_items, a_train, a_val, a_test = actor_split(
        items, actors, N_VAL_ACTORS, N_TEST_ACTORS, ACTORS_VAL, ACTORS_TEST
    )
    print(f"Actors -> train:{a_train} | val:{a_val} | test:{a_test}")
    print(f"Split sizes -> train:{len(train_items)} | val:{len(val_items)} | test:{len(test_items)}")

    train_ds = LSA64CutFlatDataset(train_items, class_to_idx, NUM_FRAMES, train=True)
    val_ds   = LSA64CutFlatDataset(val_items,   class_to_idx, NUM_FRAMES, train=False)

    train_loader_kwargs = dict(
        batch_size=MICRO_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        drop_last=True, collate_fn=collate_batch,
    )
    if NUM_WORKERS > 0:
        train_loader_kwargs.update(dict(prefetch_factor=PREFETCH_FACTOR,
                                        persistent_workers=PERSISTENT_WORKERS))
    train_loader = DataLoader(train_ds, **train_loader_kwargs)

    val_workers = NUM_WORKERS // 2 if NUM_WORKERS > 0 else 0
    val_loader_kwargs = dict(
        batch_size=MICRO_BATCH_SIZE, shuffle=False,
        num_workers=val_workers, pin_memory=PIN_MEMORY,
        drop_last=False, collate_fn=collate_batch,
    )
    if val_workers > 0:
        val_loader_kwargs.update(dict(prefetch_factor=PREFETCH_FACTOR,
                                      persistent_workers=PERSISTENT_WORKERS))
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = VideoTransformerClassifier(num_classes=len(classes_sorted)).to(device)

    def set_cnn_frozen(freeze: bool):
        for p in model.cnn.parameters():
            p.requires_grad = not freeze
    if FREEZE_CNN_EPOCHS > 0:
        set_cnn_frozen(True)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda", enabled=USE_MIXED_PRECISION)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    global_step = 0
    accum = 0
    best_val_acc = 0.0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"lsa64_fast_{run_id}"
    meta = {
        "run_name": run_name, "timestamp": run_id,
        "classes": classes_sorted, "class_to_idx": class_to_idx,
        "actors": {"train": a_train, "val": a_val, "test": a_test},
    }
    meta_path = Path(OUTPUT_DIR) / f"{run_name}_meta.json"
    with open(meta_path, "w") as f: json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    for epoch in range(1, NUM_EPOCHS + 1):
        if FREEZE_CNN_EPOCHS > 0 and epoch == FREEZE_CNN_EPOCHS + 1:
            set_cnn_frozen(False)

        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]", leave=False)
        sum_loss, sum_correct, sum_seen = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for vids, labels in train_bar:
            vids = vids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr = cosine_with_warmup(global_step, total_steps, BASE_LR, warmup_steps)
            for pg in optimizer.param_groups: pg["lr"] = lr

            with autocast("cuda", enabled=USE_MIXED_PRECISION):
                logits = model(vids)
                loss = criterion(logits, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()
            accum += 1

            if accum % ACCUM_STEPS == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                sum_correct += (preds == labels).sum().item()
                sum_seen += vids.size(0)
                sum_loss += (loss.item() * ACCUM_STEPS) * vids.size(0)

            global_step += 1
            train_bar.set_postfix({"lr": f"{lr:.2e}",
                                   "loss": f"{(sum_loss/max(1,sum_seen)):.4f}",
                                   "acc": f"{(sum_correct/max(1,sum_seen))*100:.2f}%"})

        train_loss = sum_loss / max(1, sum_seen)
        train_acc = sum_correct / max(1, sum_seen)

        # Validate
        model.eval()
        v_loss, v_correct, v_seen = 0.0, 0, 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]", leave=False)
            for vids, labels in val_bar:
                vids = vids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast("cuda", enabled=USE_MIXED_PRECISION):
                    logits = model(vids)
                    loss = criterion(logits, labels)
                v_loss += loss.item() * vids.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_seen += vids.size(0)
                val_bar.set_postfix({"loss": f"{(v_loss/max(1,v_seen)):.4f}",
                                     "acc": f"{(v_correct/max(1,v_seen))*100:.2f}%"})
        val_loss = v_loss / max(1, v_seen)
        val_acc = v_correct / max(1, v_seen)

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        # Checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "classes": classes_sorted,
            "class_to_idx": class_to_idx,
            "actors": {"train": a_train, "val": a_val, "test": a_test},
        }
        def save_ckpt(suffix):
            path = Path(OUTPUT_DIR) / f"{run_name}_{suffix}_{timestamp}.pt"
            torch.save(ckpt, path)
            print(f"Saved checkpoint: {path}")
            return path
        saved = False
        if SAVE_BEST_ONLY:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_ckpt(f"best_ep{epoch}_valacc{val_acc:.4f}")
                saved = True
        else:
            save_ckpt(f"ep{epoch}_valacc{val_acc:.4f}")
            saved = True
        if (SAVE_EVERY_N_EPOCHS is not None) and (epoch % SAVE_EVERY_N_EPOCHS == 0) and not saved:
            save_ckpt(f"ep{epoch}")

    final_path = Path(OUTPUT_DIR) / f'{run_name}_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
    torch.save({
        "epoch": NUM_EPOCHS,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "classes": classes_sorted,
        "class_to_idx": class_to_idx,
        "actors": {"train": a_train, "val": a_val, "test": a_test},
    }, final_path)
    print(f"Saved final checkpoint: {final_path}")
    print("Training complete.")

if __name__ == "__main__":
    main()
