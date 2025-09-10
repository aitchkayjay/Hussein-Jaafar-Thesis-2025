import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class KittiCarlaDataset(Dataset):
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
        mask = Image.open(mask_path)  # grayscale CARLA mask (0–23)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)  # ensure class IDs as long

        return image, mask