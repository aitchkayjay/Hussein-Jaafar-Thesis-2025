import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from unet import UNet

# === Konfiguration ===
NUM_CLASSES = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "/mnt/data1/synthetic_data/synthetic_semseg/images"
MASK_DIR = "/mnt/data1/Hussein-thesis-repo/Code-Results/Carla-ID-masks"
CHECKPOINT_PATH = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mit-synthData/unet_carla_best.pth"
SAVE_DIR = "/mnt/data1/Hussein-thesis-repo/Code-Results/Eval_Carla_ID_colored"
os.makedirs(SAVE_DIR, exist_ok=True)

# === CARLA-Palette (ID → RGB) ===
carla_id_to_color = {
    0: (0, 0, 0),
    1: (70, 70, 70),
    2: (190, 153, 153),
    3: (250, 170, 160),
    4: (220, 20, 60),
    5: (153, 153, 153),
    6: (157, 234, 50),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (107, 142, 35),
    10: (0, 0, 142),
    11: (102, 102, 156),
    12: (220, 220, 0),
    13: (70, 130, 180),
    14: (81, 0, 81),
    15: (150, 100, 100),
    16: (230, 150, 140),
    17: (180, 165, 180),
    18: (250, 170, 30),
    19: (0, 0, 230),
    20: (119, 11, 32),
    21: (0, 60, 100),
    22: (0, 0, 70),
    23: (0, 80, 100)
}

def colorize_mask(mask_np):
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in carla_id_to_color.items():
        color_mask[mask_np == class_id] = color
    return color_mask

# === Dataset (für Visualisierung)
class CarlaDatasetEval(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(os.listdir(image_dir))
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, np.array(mask), self.filenames[idx]

# === Transforms
transform_img = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_mask = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)
])

# === Load data
dataset = CarlaDatasetEval(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    transform_img=transform_img,
    transform_mask=transform_mask
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# === Load Model
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
print("✅ Modell geladen")

# === Evaluation
with torch.no_grad():
    for i, (image, mask_gt, fname) in enumerate(tqdm(loader, desc="Evaluating")):
        image = image.to(DEVICE)
        pred = model(image).argmax(dim=1).squeeze().cpu().numpy()

        image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        image_np = np.clip(image_np * np.array([0.229, 0.224, 0.225]) + 
                           np.array([0.485, 0.456, 0.406]), 0, 1)

        pred_col = colorize_mask(pred)
        gt_col = colorize_mask(mask_gt.squeeze().numpy())

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image_np)
        axs[0].set_title("Input Image")
        axs[1].imshow(gt_col)
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred_col)
        axs[2].set_title("Prediction")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{fname[0].split('.')[0]}_result.png"))
        plt.close()

print(f"✅ Fertig! Ergebnisse gespeichert in: {SAVE_DIR}")