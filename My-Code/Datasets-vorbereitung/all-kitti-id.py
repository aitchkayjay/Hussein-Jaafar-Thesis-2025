from PIL import Image
import numpy as np

mask = np.array(Image.open("/mnt/data1/datasets/kitti/kitti_semantics/training/semantic/000000_10.png"))
print("✅ Shape:", mask.shape)
print("🔎 Unique values (class IDs):", np.unique(mask))