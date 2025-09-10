import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt

from kitti_dataset import KittiDataset
from unet import UNet
from kitti_labels import trainId_to_label, NUM_CLASSES

# ======= CONFIG =======
CHECKPOINT_PATH = "/home/husjaa/Pytorch-UNet/checkpoints2/unet_best.pth"
IMAGE_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
MASK_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
SAVE_DIR = "./evaluation_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======= TRANSFORMS =======
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)
])

# ======= COLOR MAPPING =======
def decode_segmap(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for train_id, (_, color) in trainId_to_label.items():
        if train_id == 255:
            continue
        color_mask[mask == train_id] = color
    return color_mask

def create_legend():
    legend_elements = []
    for train_id, (label, color) in trainId_to_label.items():
        if train_id == 255:
            continue
        patch = mpatches.Patch(color=np.array(color) / 255.0, label=label)
        legend_elements.append(patch)
    return legend_elements

# ======= DATASET =======
dataset = KittiDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    transform_img=transform_img,
    transform_mask=transform_mask
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======= MODEL =======
model = UNet(n_channels=3, n_classes=NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ======= INFERENCE & SAVE FIGURES =======
with torch.no_grad():
    for i, (images, masks) in enumerate(dataloader):
        if i >= 3:
            break  # nur 3 Bilder speichern

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        preds = model(images)
        preds = torch.argmax(preds, dim=1)

        # Convert to numpy
        image_np = images[0].permute(1, 2, 0).cpu().numpy()
        gt_mask_np = masks[0].cpu().numpy()
        pred_mask_np = preds[0].cpu().numpy()

        gt_color = decode_segmap(gt_mask_np)
        pred_color = decode_segmap(pred_mask_np)

        # Plot & Save
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image_np)
        axs[0].set_title("Input")
        axs[0].axis("off")

        axs[1].imshow(gt_color)
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(pred_color)
        axs[2].set_title("Prediction")
        axs[2].axis("off")

        fig.legend(handles=create_legend(), loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()

        save_path = os.path.join(SAVE_DIR, f"eval_{i+1}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"✅ Gespeichert: {save_path}")
        