# scripts/data_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, patch_dir):
        self.image_dir = os.path.join(patch_dir, "images")
        self.mask_dir = os.path.join(patch_dir, "masks")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

        assert len(self.image_files) == len(self.mask_files), \
            "Mismatch between image and mask count"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.load(img_path)
        mask = np.load(mask_path)

        # Convert to float32 tensor
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return image, mask
