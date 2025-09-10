import os
import csv
from datetime import datetime
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

from unet import UNet
from utils.dice_score import dice_loss

# ==================== Konfiguration ====================
SEED          = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

NUM_CLASSES   = 24
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 4
EPOCHS        = 50
LR            = 1e-4
NUM_WORKERS   = 4
VAL_SPLIT     = 0.1

TOWN_RGB = {
    "Town01": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town01/generated2/images_rgb",
    "Town02": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town02/generated2/images_rgb",
    "Town03": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town03/generated2/images_rgb",
    "Town04": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town04/generated2/images_rgb",
    "Town05": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town05/generated2/images_rgb",
}
MASK_ROOT = "/mnt/data1/Hussein-thesis-repo/Code-Results/Carla-ID-masks"

CHECKPOINT_PATH = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mit-synthData/unet_carla_best.pth"
RESULTS_DIR     = os.path.dirname(CHECKPOINT_PATH)
LAST_PATH       = os.path.join(RESULTS_DIR, "unet_carla_last.pth")
HISTORY_CSV     = os.path.join(RESULTS_DIR, "loss_history.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== Transforms ====================
transform_img = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_mask = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)
])

# ==================== Dataset ====================
class CarlaPairsDataset(Dataset):
    def __init__(self, pairs, transform_img=None, transform_mask=None):
        self.pairs = pairs
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask_img = Image.open(mask_path)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask_img = self.transform_mask(mask_img)

        mask = torch.from_numpy(np.array(mask_img, dtype=np.uint8)).long()
        return image, mask

def collect_pairs():
    pairs = []
    for town, rgb_dir in TOWN_RGB.items():
        rgb_dir = Path(rgb_dir)
        mask_dir = Path(MASK_ROOT) / town
        if not rgb_dir.exists() or not mask_dir.exists():
            print(f"Ordner fehlt → {town} | images:{rgb_dir.exists()} masks:{mask_dir.exists()}")
            continue
        mask_map = {p.name: p for p in mask_dir.glob("*.png")}
        cnt = 0
        for img_path in rgb_dir.glob("*.png"):
            m = mask_map.get(img_path.name)
            if m is not None:
                pairs.append((str(img_path), str(m)))
                cnt += 1
        print(f"{town}: {cnt} Paare")
    random.shuffle(pairs)
    print(f"Gesamtpaare: {len(pairs)}")
    return pairs

# ==================== CSV Logging Helpers ====================
def init_history_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "epoch", "train_loss", "val_loss", "lr"])

def append_history_csv(path, epoch, train_loss, val_loss, lr):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{lr:.8f}"
        ])

def save_checkpoint(state_path, model, optimizer, epoch, best_val, train_hist, val_hist):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val,
        "history": {
            "train_loss": train_hist,
            "val_loss": val_hist
        },
        "learning_rate": optimizer.param_groups[0]["lr"]
    }
    torch.save(payload, state_path)

# ==================== DataLoader ====================
pairs = collect_pairs()
assert len(pairs) > 0, "Keine (RGB,ID)-Paare gefunden. Pfade prüfen!"

full_dataset = CarlaPairsDataset(pairs, transform_img=transform_img, transform_mask=transform_mask)

if VAL_SPLIT > 0:
    val_len = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_len = len(full_dataset) - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(SEED))
    print(f"Split: train={train_len}, val={val_len}")
else:
    train_ds, val_ds = full_dataset, None

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True) if val_ds else None

# ==================== Modell, Optimizer, Loss ====================
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
ce_loss = nn.CrossEntropyLoss(ignore_index=0)

def step_batch(images, masks):
    preds = model(images)
    return ce_loss(preds, masks) + dice_loss(preds, masks, multiclass=True)

# ==================== Training Loop mit CSV + History ====================
init_history_csv(HISTORY_CSV)
best_val = float('inf')
train_hist, val_hist = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
    for images, masks in loop:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)

        loss  = step_batch(images, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += loss.item()
        loop.set_postfix(loss=loss.item())

    train_loss = running / max(1, len(train_loader))
    val_loss = None

    if val_loader is not None:
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
                images = images.to(DEVICE, non_blocking=True)
                masks  = masks.to(DEVICE, non_blocking=True)
                val_running += step_batch(images, masks).item()
        val_loss = val_running / max(1, len(val_loader))

    train_hist.append(float(train_loss))
    val_hist.append(float(val_loss) if val_loss is not None else None)
    append_history_csv(HISTORY_CSV, epoch, train_loss, val_loss, optimizer.param_groups[0]["lr"])

    if val_loss is not None:
        print(f"[Epoch {epoch}] train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
    else:
        print(f"[Epoch {epoch}] train_loss: {train_loss:.4f}")

    # immer Last-Checkpoint
    save_checkpoint(LAST_PATH, model, optimizer, epoch, best_val if best_val != float('inf') else None, train_hist, val_hist)

    # Best speichern
    crit = val_loss if val_loss is not None else train_loss
    if crit < best_val:
        best_val = crit
        save_checkpoint(CHECKPOINT_PATH, model, optimizer, epoch, best_val, train_hist, val_hist)
        print(f"gespeichert best → {CHECKPOINT_PATH}")

print("\nTraining fertig.")
print(f"Loss CSV: {HISTORY_CSV}")
print(f"Last Checkpoint: {LAST_PATH}")
print(f"Best Checkpoint: {CHECKPOINT_PATH}")
