import os
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from KittiCarlaDataset import KittiCarlaDataset
from unet import UNet
from utils.dice_score import dice_loss

# === Konstanten ===
NUM_CLASSES = 24
IMAGE_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
MASK_DIR = "/mnt/data1/Hussein-thesis-repo/Kitti-2-Carla-masks"
CHECKPOINT_PATH = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mit-Carla-Masks/unet_best.pth"
LAST_PATH = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mit-Carla-Masks/unet_last.pth"
HISTORY_CSV = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mit-Carla-Masks/loss_history.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
transform_img = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_mask = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)
])

# === Dataset und DataLoader ===
dataset = KittiCarlaDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    transform_img=transform_img,
    transform_mask=transform_mask
)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# === Modell Optimizer Loss ===
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
ce_loss = nn.CrossEntropyLoss(ignore_index=0)

# === Logging Helfer ===
def init_history_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "epoch", "train_loss", "lr"])

def append_history_csv(path, epoch, train_loss, lr):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.now().isoformat(timespec="seconds"), epoch, f"{train_loss:.6f}", f"{lr:.8f}"])

def save_checkpoint(state_path, model, optimizer, epoch, best_loss, train_history, extra=None):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_train_loss": best_loss,
        "history": {
            "train_loss": train_history
        }
    }
    if extra:
        payload.update(extra)
    torch.save(payload, state_path)

# === Training Loop mit History im Checkpoint und CSV Log ===
best_loss = float("inf")
epochs = 50
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
init_history_csv(HISTORY_CSV)

train_history = []

for epoch in range(1, epochs + 1):
    model.train()
    running = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

    for images, masks in loop:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True).long()

        preds = model(images)
        loss = ce_loss(preds, masks) + dice_loss(preds, masks, multiclass=True)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running / max(1, len(train_loader))
    train_history.append(float(avg_loss))
    append_history_csv(HISTORY_CSV, epoch, avg_loss, optimizer.param_groups[0]["lr"])

    print(f"Epoch {epoch} Loss {avg_loss:.4f}")

    # immer einen Last Checkpoint mit kompletter History speichern
    save_checkpoint(
        LAST_PATH, model, optimizer, epoch, best_loss if best_loss != float('inf') else None,
        train_history, extra={"learning_rate": optimizer.param_groups[0]["lr"]}
    )

    # besten Checkpoint bei Verbesserung speichern
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint(
            CHECKPOINT_PATH, model, optimizer, epoch, best_loss,
            train_history, extra={"learning_rate": optimizer.param_groups[0]["lr"]}
        )
        print(f"bestes Modell gespeichert {CHECKPOINT_PATH}")

print(f"CSV Log {HISTORY_CSV}")
print(f"Last Checkpoint {LAST_PATH}")
print(f"Best Checkpoint {CHECKPOINT_PATH}")