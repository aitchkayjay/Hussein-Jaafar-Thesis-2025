import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from KittiCarlaDataset import KittiCarlaDataset
from unet import UNet

from PIL import Image

# === Constants ===
NUM_CLASSES = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mit-Carla-Masks/unet_best.pth"
IMAGE_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
MASK_DIR = "/mnt/data1/Hussein-thesis-repo/Kitti-2-Carla-masks"
SAVE_DIR = "/mnt/data1/Hussein-thesis-repo/Code-Results/Kitti_Mit_Carla_eval_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# === CARLA Color Palette ===
carla_palette = {
    0:  (0, 0, 0),           # None (void)
    1:  (70, 70, 70),        # Building
    2:  (100, 40, 40),       # Fence
    3:  (55, 90, 80),        # Other
    4:  (220, 20, 60),       # Pedestrian
    5:  (153, 153, 153),     # Pole
    6:  (157, 234, 50),      # Road Line
    7:  (128, 64, 128),      # Road
    8:  (244, 35, 232),      # Sidewalk
    9:  (107, 142, 35),      # Vegetation
    10: (0, 0, 142),         # Vehicle
    11: (102, 102, 156),     # Wall
    12: (220, 220, 0),       # Traffic Sign
    13: (70, 130, 180),      # Sky
    14: (81, 0, 81),         # Ground
    15: (150, 100, 100),     # Bridge
    16: (230, 150, 140),     # Rail Track
    17: (180, 165, 180),     # Guard Rail
    18: (250, 170, 30),      # Traffic Light
    19: (110, 190, 160),     # Static
    20: (170, 120, 50),      # Dynamic
    21: (45, 60, 150),       # Water
    22: (152, 251, 152),     # Terrain
    23: (255, 0, 0),         # MyNewTag or custom
}

# === Utility to apply palette ===
def apply_palette(mask_np, palette):
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in palette.items():
        color_mask[mask_np == class_id] = color
    return color_mask

# === Transforms ===
transform_img = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_mask = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)
])

# === Dataset & Loader ===
dataset = KittiCarlaDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    transform_img=transform_img,
    transform_mask=transform_mask
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# === Load Model ===
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded.")

# === Evaluation ===
with torch.no_grad():
    for i, (image, mask) in enumerate(loader):
        image = image.to(DEVICE)
        output = model(image)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        # Unnormalize image
        image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        image_np = np.clip(image_np, 0, 1)

        # Apply CARLA color map
        pred_color = apply_palette(pred, carla_palette)
        mask_color = apply_palette(mask_np, carla_palette)

        # Save side-by-side figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image_np)
        axs[0].set_title("RGB Image")
        axs[1].imshow(mask_color)
        axs[1].set_title("Ground Truth (CARLA colors)")
        axs[2].imshow(pred_color)
        axs[2].set_title("Prediction (CARLA colors)")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"sample_{i:03d}.png"))
        plt.close()

        if i >= 2:
            break

print(f"✅ Evaluation completed. Colorized outputs saved in: {SAVE_DIR}")