import os
import csv
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from unet import UNet
from utils.dice_score import dice_loss

# =========================
# Konfiguration
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

NUM_CLASSES = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-4
VAL_SPLIT = 0.1
NUM_WORKERS = 4
BALANCE_DOMAINS = True

# Pfade anpassen
KITTI_IMAGE_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
KITTI_MASK_DIR  = "/mnt/data1/Hussein-thesis-repo/Kitti-2-Carla-masks"

CARLA_IMAGE_DIR = {
    "Town01": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town01/generated2/images_rgb",
    "Town02": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town02/generated2/images_rgb",
    "Town03": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town03/generated2/images_rgb",
    "Town04": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town04/generated2/images_rgb",
    "Town05": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town05/generated2/images_rgb",
}
CARLA_MASK_DIR  = "/mnt/data1/Hussein-thesis-repo/Code-Results/Carla-ID-masks"

CHECKPOINT_DIR  = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mixed"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "unet_kitti_plus_carla_best.pth")
LAST_PATH       = os.path.join(CHECKPOINT_DIR, "unet_kitti_plus_carla_last.pth")
HISTORY_CSV     = os.path.join(CHECKPOINT_DIR, "loss_history.csv")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================
# Transforms
# =========================
transform_img = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)
])

# =========================
# Datasets
# =========================
class FolderSegDataset(Dataset):
    """
    Bild Masken Paarung 1 zu 1. Masken als ID Bild uint8 mit Klassen 0 bis 23. 0 wird als void behandelt.
    """
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None, domain_name=""):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.domain_name = domain_name

        imgs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.filenames = []
        for f in imgs:
            base = os.path.splitext(f)[0]
            mask_png = os.path.join(mask_dir, base + ".png")
            if os.path.exists(mask_png):
                self.filenames.append(f)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path  = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(fname)[0] + ".png")

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()
        return image, mask, self.domain_name

def build_carla_concat(image_dirs_dict, mask_root, transform_img, transform_mask):
    carla_datasets = []
    total_pairs = 0
    for town, img_dir in image_dirs_dict.items():
        mask_dir = os.path.join(mask_root, town)
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            print(f"übersprungen {town} images:{os.path.isdir(img_dir)} masks:{os.path.isdir(mask_dir)}")
            continue
        ds = FolderSegDataset(
            image_dir=img_dir,
            mask_dir=mask_dir,
            transform_img=transform_img,
            transform_mask=transform_mask,
            domain_name=f"CARLA_{town}"
        )
        print(f"{town} Paare {len(ds)}")
        total_pairs += len(ds)
        if len(ds) > 0:
            carla_datasets.append(ds)
    assert len(carla_datasets) > 0, "keine CARLA Town Datasets gefunden"
    print(f"CARLA Gesamtpaare {total_pairs}")
    return ConcatDataset(carla_datasets)

# Einzel Datasets
kitti_ds = FolderSegDataset(
    KITTI_IMAGE_DIR, KITTI_MASK_DIR,
    transform_img=transform_img, transform_mask=transform_mask, domain_name="KITTI"
)

carla_ds = build_carla_concat(
    image_dirs_dict=CARLA_IMAGE_DIR,
    mask_root=CARLA_MASK_DIR,
    transform_img=transform_img,
    transform_mask=transform_mask
)

# Zusammenführen
full_ds = ConcatDataset([kitti_ds, carla_ds])

# =========================
# Split und Loader
# =========================
val_len = max(1, int(len(full_ds) * VAL_SPLIT))
train_len = len(full_ds) - val_len
train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                generator=torch.Generator().manual_seed(SEED))

def make_loader(dataset, shuffle, batch_size=BATCH_SIZE, balance_domains=False):
    if not balance_domains or not shuffle:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # Gleichgewicht zwischen KITTI und CARLA
    domain_labels = []
    for idx in dataset.indices:
        if idx < len(kitti_ds):
            domain_labels.append(0)  # KITTI
        else:
            domain_labels.append(1)  # CARLA
    domain_labels = np.array(domain_labels)
    n0 = int((domain_labels == 0).sum())
    n1 = int((domain_labels == 1).sum())
    w0 = 0.5 / n0 if n0 > 0 else 0.0
    w1 = 0.5 / n1 if n1 > 0 else 0.0
    sample_weights = np.where(domain_labels == 0, w0, w1).astype(np.float32)

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=NUM_WORKERS, pin_memory=True)

train_loader = make_loader(train_ds, shuffle=True,  balance_domains=BALANCE_DOMAINS)
val_loader   = make_loader(val_ds,   shuffle=False, balance_domains=False)

print(f"KITTI samples {len(kitti_ds)}  CARLA samples {len(carla_ds)}")
print(f"Train {len(train_ds)}  Val {len(val_ds)}")

# =========================
# Modell und Optimierung
# =========================
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
ce_loss = nn.CrossEntropyLoss(ignore_index=0)

def step_batch(images, masks):
    preds = model(images)
    loss = ce_loss(preds, masks) + dice_loss(preds, masks, multiclass=True)
    return loss

# =========================
# Logging Hilfen
# =========================
def init_history_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "epoch", "train_loss", "val_loss", "best_so_far", "lr"])

def append_history_csv(path, epoch, train_loss, val_loss, best_so_far, lr):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            int(best_so_far),
            f"{lr:.8f}"
        ])

def save_checkpoint(state_path, model, optimizer, epoch, best_val, train_hist, val_hist, extra=None):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val,
        "history": {
            "train_loss": train_hist,
            "val_loss": val_hist
        }
    }
    if extra:
        payload.update(extra)
    torch.save(payload, state_path)

# =========================
# Training
# =========================
init_history_csv(HISTORY_CSV)
train_history, val_history = [], []
best_val = math.inf

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    running = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train")
    for images, masks, _domain in pbar:
        images, masks = images.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = step_batch(images, masks)
        loss.backward()
        optimizer.step()
        running += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    train_loss = running / max(1, len(train_loader))

    # Val
    model.eval()
    running = 0.0
    with torch.no_grad():
        for images, masks, _domain in val_loader:
            images, masks = images.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
            loss = step_batch(images, masks)
            running += loss.item()
    val_loss = running / max(1, len(val_loader))

    train_history.append(float(train_loss))
    val_history.append(float(val_loss))
    is_best = val_loss < best_val
    append_history_csv(HISTORY_CSV, epoch, train_loss, val_loss, is_best, LR)

    print(f"Epoch {epoch}  train {train_loss:.4f}  val {val_loss:.4f}")

    # Last immer speichern
    save_checkpoint(
        LAST_PATH, model, optimizer, epoch,
        best_val if best_val != math.inf else None,
        train_history, val_history,
        extra={"learning_rate": LR}
    )

    # Best bei Verbesserung speichern
    if is_best:
        best_val = val_loss
        save_checkpoint(
            CHECKPOINT_PATH, model, optimizer, epoch, best_val,
            train_history, val_history,
            extra={"learning_rate": LR}
        )
        print(f"saved best model {CHECKPOINT_PATH}")

print(f"Loss CSV {HISTORY_CSV}")
print(f"Last Checkpoint {LAST_PATH}")
print(f"Best Checkpoint {CHECKPOINT_PATH}")