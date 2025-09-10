import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# === Paths ===
input_dir = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic"
output_dir = "/mnt/data1/Hussein-thesis-repo/Kitti-2-Carla-masks"
os.makedirs(output_dir, exist_ok=True)

# === Final KITTI to CARLA ID Mapping ===
kitti_to_carla = {
    7: 7,     # road
    9: 7,
    8: 8,     # sidewalk
    11: 1,    # building
    12: 11,   # wall
    13: 2,    # fence
    17: 5,    # pole
    19: 18,   # traffic light
    20: 12,   # traffic sign
    21: 9,    # vegetation
    22: 22,   # terrain
    23: 13,   # sky
    24: 4,    # person
    25: 4,    # rider → person
    26: 10,   # car
    27: 10,   # truck → car
    28: 10,   # bus → car
    29: 10,   # caravan → car
    30: 10,   # trailer → car
    31: 10,   # train → car
    32: 10,   # motorcycle → car
    33: 10,   # bicycle → car
    # All others → void (0)
}
default_val = 0  # void

all_found_ids = set()

# === Remap grayscale masks ===
for fname in tqdm(sorted(os.listdir(input_dir))):
    if not fname.endswith(".png"):
        continue

    img_path = os.path.join(input_dir, fname)
    mask = np.array(Image.open(img_path))  # Already grayscale!

    unique_ids = np.unique(mask)
    all_found_ids.update(unique_ids)

    remapped = np.full_like(mask, default_val)
    for kitti_id in unique_ids:
        carla_id = kitti_to_carla.get(int(kitti_id), default_val)
        remapped[mask == kitti_id] = carla_id

    out_path = os.path.join(output_dir, fname)
    Image.fromarray(remapped.astype(np.uint8)).save(out_path)

# === Summary ===
print("✅ Remapping complete. Output saved to:", output_dir)
print("📌 Unique KITTI IDs found:", sorted(all_found_ids))