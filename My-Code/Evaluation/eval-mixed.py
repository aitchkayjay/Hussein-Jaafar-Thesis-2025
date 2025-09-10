import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms
from tqdm import tqdm

from unet import UNet

# =========================
# Config
# =========================
NUM_CLASSES = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)

# --- Pfade anpassen ---
KITTI_IMAGE_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
KITTI_MASK_DIR  = "/mnt/data1/Hussein-thesis-repo/Kitti-2-Carla-masks"
CARLA_IMAGE_DIR = "/mnt/data1/synthetic_data/synthetic_semseg/images"
CARLA_MASK_DIR  = "/mnt/data1/Hussein-thesis-repo/Code-Results/Carla-ID-masks"

CHECKPOINT_PATH = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mixed/unet_kitti_plus_carla_best.pth"
SAVE_DIR        = "/mnt/data1/Hussein-thesis-repo/Code-Results/Eval_KITTI_plus_CARLA"
os.makedirs(SAVE_DIR, exist_ok=True)

# Anzahl zu visualisierender Beispiele je Domain (None = alle im Val-Split)
MAX_PER_DOMAIN = 20

# =========================
# CARLA Palette (ID -> RGB)
# =========================
CARLA_ID2COLOR = {
    0:(0,0,0), 1:(70,70,70), 2:(190,153,153), 3:(250,170,160), 4:(220,20,60),
    5:(153,153,153), 6:(157,234,50), 7:(128,64,128), 8:(244,35,232),
    9:(107,142,35), 10:(0,0,142), 11:(102,102,156), 12:(220,220,0),
    13:(70,130,180), 14:(81,0,81), 15:(150,100,100), 16:(230,150,140),
    17:(180,165,180), 18:(250,170,30), 19:(0,0,230), 20:(119,11,32),
    21:(0,60,100), 22:(0,0,70), 23:(0,80,100)
}

def colorize(mask_np):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in CARLA_ID2COLOR.items():
        rgb[mask_np == cid] = col
    return rgb

def denorm(img_t):
    # undo ImageNet-Norm für Anzeige
    mean = np.array([0.485, 0.456, 0.406])[None, None, :]
    std  = np.array([0.229, 0.224, 0.225])[None, None, :]
    img = img_t.permute(1,2,0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)
    return img

def overlay(rgb_img_0_1, color_mask_rgb, alpha=0.5):
    cm = color_mask_rgb.astype(np.float32)/255.0
    return np.clip((1-alpha)*rgb_img_0_1 + alpha*cm, 0, 1)

# =========================
# Dataset
# =========================
class FolderSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None, domain_name=""):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.domain = domain_name

        imgs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
        # Masken als .png erwartet:
        self.filenames = [f for f in imgs if os.path.exists(os.path.join(mask_dir, os.path.splitext(f)[0]+".png"))]

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(fname)[0]+".png")

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)  # ID-Maske (uint8)

        if transform_img: image = transform_img(image)
        if transform_mask: mask = transform_mask(mask)

        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()
        return image, mask, fname, self.domain

# =========================
# Transforms
# =========================
transform_img = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
transform_mask = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)
])

# =========================
# Build Val-Set (concat + split)
# =========================
kitti_ds = FolderSegDataset(KITTI_IMAGE_DIR, KITTI_MASK_DIR, transform_img, transform_mask, "KITTI")
carla_ds = FolderSegDataset(CARLA_IMAGE_DIR, CARLA_MASK_DIR, transform_img, transform_mask, "CARLA")
full_ds = ConcatDataset([kitti_ds, carla_ds])

# kleiner Val-Split (fix): 10%
val_len = max(1, int(0.1 * len(full_ds)))
train_len = len(full_ds) - val_len
_, val_ds = random_split(full_ds, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))

val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

# =========================
# Load Model
# =========================
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print("✅ Modell geladen:", CHECKPOINT_PATH)

# =========================
# Run & Save
# =========================
os.makedirs(os.path.join(SAVE_DIR, "KITTI"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "CARLA"), exist_ok=True)

count_domain = {"KITTI":0, "CARLA":0}

with torch.no_grad():
    for image, mask, fname, domain in tqdm(val_loader, desc="Evaluating"):
        # limit per domain if desired
        d = domain[0]
        if MAX_PER_DOMAIN is not None and count_domain[d] >= MAX_PER_DOMAIN:
            continue

        image = image.to(DEVICE)
        logits = model(image)
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()

        # visuals
        img_np = denorm(image.squeeze())
        gt_np  = mask.squeeze().cpu().numpy()

        pred_col = colorize(pred)
        gt_col   = colorize(gt_np)
        ov_pred  = overlay(img_np, pred_col)
        ov_gt    = overlay(img_np, gt_col)

        # figure: RGB | GT | Pred | Overlay(GT) | Overlay(Pred)
        fig, axs = plt.subplots(1, 5, figsize=(16, 4))
        axs[0].imshow(img_np);   axs[0].set_title("RGB")
        axs[1].imshow(gt_col);   axs[1].set_title("GT (CARLA)")
        axs[2].imshow(pred_col); axs[2].set_title("Pred (CARLA)")
        axs[3].imshow(ov_gt);    axs[3].set_title("Overlay GT")
        axs[4].imshow(ov_pred);  axs[4].set_title("Overlay Pred")
        for ax in axs: ax.axis("off")
        plt.tight_layout()

        base = os.path.splitext(fname[0])[0]
        out_dir = os.path.join(SAVE_DIR, d)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f"{base}_viz.png"))
        plt.close()

        count_domain[d] += 1

print(f"Fertig. Gespeichert in: {SAVE_DIR}")
print("Anzahl gespeicherter Beispiele:", count_domain)