import os
import csv
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms

from unet import UNet

# =========================
# Konfiguration
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 24

# Welche Kennzahl anzeigen: "train" oder "val"
LOSS_METRIC = "train"   # <<— HIER festlegen

# Pfade für KITTI Eval
EVAL_KITTI_IMAGE_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
EVAL_KITTI_MASK_DIR  = "/mnt/data1/Hussein-thesis-repo/Kitti-2-Carla-masks"

# Pfade für CARLA Eval
EVAL_CARLA_IMAGE_DIRS = {
    "Town01": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town01/generated2/images_rgb",
    "Town02": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town02/generated2/images_rgb",
    "Town03": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town03/generated2/images_rgb",
    "Town04": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town04/generated2/images_rgb",
    "Town05": "/mnt/data1/synthetic_data/kitti_carla_adjusted_res/Town05/generated2/images_rgb",
}
EVAL_CARLA_MASK_ROOT  = "/mnt/data1/Hussein-thesis-repo/Code-Results/Carla-ID-masks"  # enthält TownXX Unterordner

# Drei Modelle mit Namen und Pfaden
MODELS: Dict[str, str] = {
    "Modell 1": "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mit-Carla-Masks/unet_best.pth",         # KITTI mit CARLA-IDs
    "Modell 2": "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mit-synthData/unet_carla_best.pth",     # CARLA
    "Modell 3": "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mixed/unet_kitti_plus_carla_best.pth",  # Mixed
}

OUT_DIR = "/mnt/data1/Hussein-thesis-repo/Eval-Results"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Farbpalette
# =========================
carla_id_to_color = {
    0:(0,0,0), 1:(70,70,70), 2:(190,153,153), 3:(250,170,160),
    4:(220,20,60), 5:(153,153,153), 6:(157,234,50), 7:(128,64,128),
    8:(244,35,232), 9:(107,142,35), 10:(0,0,142), 11:(102,102,156),
    12:(220,220,0), 13:(70,130,180), 14:(81,0,81), 15:(150,100,100),
    16:(230,150,140), 17:(180,165,180), 18:(250,170,30), 19:(0,0,230),
    20:(119,11,32), 21:(0,60,100), 22:(0,0,70), 23:(0,80,100),
}
def build_lut_256(mapping: dict) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for k, rgb in mapping.items():
        lut[int(k)] = rgb
    return lut
PALETTE_LUT = build_lut_256(carla_id_to_color)
def colorize_ids_rgb(id_mask: np.ndarray, lut_256x3: np.ndarray) -> np.ndarray:
    id_mask = id_mask.astype(np.uint8, copy=False)
    return lut_256x3[id_mask]

# ========= Transforms =========
transform_img = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
transform_mask = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST),
])

def denorm_img(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3,1,1)
    x = t * std + mean
    x = torch.clamp(x, 0, 1)
    return x.permute(1,2,0).cpu().numpy()

# ========= Datenpaare =========
def collect_pairs_kitti(img_dir: str, mask_dir: str) -> List[Tuple[str,str]]:
    img_paths = sorted([p for p in Path(img_dir).glob("*") if p.suffix.lower() in [".png",".jpg",".jpeg"]])
    pairs = []
    for ip in img_paths:
        mp = Path(mask_dir) / (ip.stem + ".png")
        if mp.exists():
            pairs.append((str(ip), str(mp)))
    if len(pairs) == 0:
        raise RuntimeError("keine KITTI Paare gefunden")
    return pairs

def collect_pairs_carla(image_dirs_dict: Dict[str,str], mask_root: str) -> List[Tuple[str,str]]:
    pairs = []
    for town, img_dir in image_dirs_dict.items():
        mask_dir = Path(mask_root) / town
        if not Path(img_dir).is_dir() or not mask_dir.is_dir():
            continue
        for ip in sorted([p for p in Path(img_dir).glob("*") if p.suffix.lower() in [".png",".jpg",".jpeg"]]):
            mp = mask_dir / (ip.stem + ".png")
            if mp.exists():
                pairs.append((str(ip), str(mp)))
    if len(pairs) == 0:
        raise RuntimeError("keine CARLA Paare gefunden")
    return pairs

# ========= Loss-Ermittlung (strict nach gewünschter Metrik) =========
def _float_list(seq) -> List[float]:
    out = []
    for x in seq:
        try:
            fx = float(x)
        except Exception:
            continue
        if not np.isnan(fx):
            out.append(fx)
    return out

def _best_from_history(ckpt_dict: dict, prefer: str) -> Optional[float]:
    hist = ckpt_dict.get("history", {})
    if not isinstance(hist, dict):
        return None
    if prefer == "train" and "train_loss" in hist and hist["train_loss"]:
        vals = _float_list(hist["train_loss"])
        return float(np.min(vals)) if vals else None
    if prefer == "val" and "val_loss" in hist and hist["val_loss"]:
        vals = _float_list(hist["val_loss"])
        return float(np.min(vals)) if vals else None
    return None

def _best_from_ckpt_keys(ckpt_dict: dict, prefer: str) -> Optional[float]:
    if prefer == "train":
        k = ckpt_dict.get("train_loss", None)
        return float(k) if k is not None else None
    else:
        k = ckpt_dict.get("best_val_loss", None)
        return float(k) if k is not None else None

def _best_from_csv(ckpt_path: str, prefer: str) -> Optional[float]:
    csv_path = Path(ckpt_path).with_name("loss_history.csv")
    if not csv_path.exists():
        return None
    best = None
    col = "train_loss" if prefer == "train" else "val_loss"
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            val = row.get(col, "")
            try:
                if val not in ("", None):
                    v = float(val)
                    best = v if best is None or v < best else best
            except ValueError:
                continue
    return best

# ========= Modelle =========
def load_model_from_checkpoint(path: str, prefer_metric: str = "train") -> Tuple[nn.Module, Optional[float]]:
    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)

    # State laden
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict):
        if any(torch.is_tensor(v) for v in ckpt.values()):
            model.load_state_dict(ckpt)   # reines state_dict
        else:
            raise RuntimeError(f"Unerwartetes Checkpoint-Format ohne state dict: {path}")
    else:
        model.load_state_dict(ckpt)

    # Best-Loss NUR aus der gewünschten Quelle
    best = None
    if isinstance(ckpt, dict):
        best = _best_from_ckpt_keys(ckpt, prefer_metric)
        if best is None:
            best = _best_from_history(ckpt, prefer_metric)
    if best is None:
        best = _best_from_csv(path, prefer_metric)

    model.eval()
    return model, best

# ========= Inferenz =========
@torch.no_grad()
def predict_mask(model: nn.Module, img_t: torch.Tensor) -> np.ndarray:
    logits = model(img_t.unsqueeze(0).to(DEVICE))
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred

def assert_valid_ids(arr: np.ndarray):
    u = np.unique(arr)
    bad = u[(u > 23) | (u < 0)]
    if bad.size > 0:
        raise ValueError(f"unerwartete Klassenwerte gefunden {bad.tolist()}")

# ========= Rendering =========
def render_matrix_for_domain(sample_pair: Tuple[str,str],
                             models: Dict[str,str],
                             out_png: str,
                             fig_title: str):
    loaded = {}
    for name, path in models.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint fehlt für {name}: {path}")
        loaded[name] = load_model_from_checkpoint(path, prefer_metric=LOSS_METRIC)

    img_path, mask_path = sample_pair
    img_pil = Image.open(img_path).convert("RGB")
    msk_pil = Image.open(mask_path)

    img_t = transform_img(img_pil)
    msk_t = transform_mask(msk_pil)
    gt_ids = np.array(msk_t, dtype=np.uint8)
    assert_valid_ids(gt_ids)

    names  = list(loaded.keys())
    losses = [loaded[n][1] for n in names]

    n_models = len(loaded)
    fig, axes = plt.subplots(n_models, 3, figsize=(12, 3.2 * n_models))
    if n_models == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, name in enumerate(names):
        model, _ = loaded[name]
        pred_ids = predict_mask(model, img_t)
        assert_valid_ids(pred_ids)

        img_vis = denorm_img(img_t)
        gt_vis  = colorize_ids_rgb(gt_ids,  PALETTE_LUT)
        pr_vis  = colorize_ids_rgb(pred_ids, PALETTE_LUT)

        axes[r,0].imshow(img_vis); axes[r,0].set_title("Original");      axes[r,0].axis("off")
        axes[r,1].imshow(gt_vis);  axes[r,1].set_title("Ground Truth");  axes[r,1].axis("off")
        axes[r,2].imshow(pr_vis);  axes[r,2].set_title("Prediction");    axes[r,2].axis("off")

        axes[r,0].set_ylabel(name, fontsize=11, rotation=90, labelpad=12)

        # Nur Zahl, kein Debug Text
        if losses[r] is not None:
            axes[r,2].text(0.99, 1.02, f" {LOSS_METRIC} loss {losses[r]:.4f}",
                           transform=axes[r,2].transAxes, ha="right", va="bottom", fontsize=10)
        else:
            axes[r,2].text(0.99, 1.02, f" {LOSS_METRIC} loss n/a",
                           transform=axes[r,2].transAxes, ha="right", va="bottom", fontsize=10)

    fig.suptitle(fig_title, fontsize=14)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.93])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"gespeichert {out_png}")

# ========= Hauptteil =========
def main():
    kitti_pairs = collect_pairs_kitti(EVAL_KITTI_IMAGE_DIR, EVAL_KITTI_MASK_DIR)
    carla_pairs = collect_pairs_carla(EVAL_CARLA_IMAGE_DIRS, EVAL_CARLA_MASK_ROOT)

    random.shuffle(kitti_pairs); random.shuffle(carla_pairs)

    kitti_sample = kitti_pairs[0]
    carla_sample = carla_pairs[0]

    out_kitti = os.path.join(OUT_DIR, "eval_matrix_kitti_sample_1.png")
    out_carla = os.path.join(OUT_DIR, "eval_matrix_carla_sample_1.png")

    render_matrix_for_domain(kitti_sample, MODELS, out_kitti, "KITTI sample evaluation")
    render_matrix_for_domain(carla_sample, MODELS, out_carla, "CARLA sample evaluation")

if __name__ == "__main__":
    main()