import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt  # Für Loss-Plot

from kitti_dataset import KittiDataset
from unet import UNet
from utils.dice_score import dice_loss

# Konstanten
NUM_CLASSES = 19
IMAGE_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
MASK_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic"
CHECKPOINT_PATH = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoints3/unet_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
from torchvision import transforms
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
])

# Dataset & Loader
dataset = KittiDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    transform_img=transform_img,
    transform_mask=transform_mask
)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Modell
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)

# Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)
ce_loss = nn.CrossEntropyLoss(ignore_index=255)

# Training Loop
best_loss = float('inf')
epochs = 50
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
loss_history = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = ce_loss(preds, masks) + dice_loss(preds, masks, multiclass=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # Bestes Modell speichern
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"✅ Bestes Modell gespeichert: {CHECKPOINT_PATH}")

# Nach Training: Loss Plot
plt.figure()
plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss über Epochen")
plt.grid(True)
plt.savefig("training_loss_plot.png")
plt.close()
print("📈 Trainingskurve gespeichert unter: training_loss_plot.png")