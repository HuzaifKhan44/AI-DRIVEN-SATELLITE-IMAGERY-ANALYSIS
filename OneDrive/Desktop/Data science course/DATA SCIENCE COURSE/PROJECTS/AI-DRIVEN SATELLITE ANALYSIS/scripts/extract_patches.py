import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import rasterio
from scripts.config import IMAGE_DIR, MASK_DIR, PATCH_SAVE_DIR, PATCH_SIZE, STRIDE, NORMALIZE, REMOVE_EMPTY_PATCHES, THRESHOLD_MASK

def extract_patches(image, mask, patch_size, stride):
    h, w = image.shape[1], image.shape[2]
    image_patches = []
    mask_patches = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            img_patch = image[:, i:i+patch_size, j:j+patch_size]
            mask_patch = mask[:, i:i+patch_size, j:j+patch_size]

            if REMOVE_EMPTY_PATCHES:
                if mask_patch.sum() / (patch_size * patch_size) < THRESHOLD_MASK:
                    continue

            image_patches.append(img_patch)
            mask_patches.append(mask_patch)

    return image_patches, mask_patches


def main():
    # ✅ Hardcoded paths to your .tif files
    image_path = r"C:\Users\khuza\OneDrive\Desktop\Data science course\DATA SCIENCE COURSE\PROJECTS\AI-DRIVEN SATELLITE ANALYSIS\data\raw\bangalore_rgb.tif"
    mask_path  = r"C:\Users\khuza\OneDrive\Desktop\Data science course\DATA SCIENCE COURSE\PROJECTS\AI-DRIVEN SATELLITE ANALYSIS\data\raw\bangalore_mask.tif"

    out_img_dir = Path(PATCH_SAVE_DIR) / "images"
    out_mask_dir = Path(PATCH_SAVE_DIR) / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Load .tif files using rasterio
    with rasterio.open(image_path) as src_img:
        image = src_img.read()  # Shape: (C, H, W)

    with rasterio.open(mask_path) as src_mask:
        mask = src_mask.read(1)  # Shape: (H, W)
        mask = np.expand_dims(mask, axis=0)  # Convert to (1, H, W)

    if NORMALIZE:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    img_patches, mask_patches = extract_patches(image, mask, PATCH_SIZE, STRIDE)

    patch_id = 0
    for ip, mp in zip(img_patches, mask_patches):
        np.save(out_img_dir / f"patch_{patch_id}.npy", ip)
        np.save(out_mask_dir / f"patch_{patch_id}.npy", mp)
        patch_id += 1

    print(f"[INFO] Extracted and saved {patch_id} patches.")



if __name__ == "__main__":
    main()
