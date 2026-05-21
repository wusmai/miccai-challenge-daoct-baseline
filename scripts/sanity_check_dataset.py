import os
from glob import glob
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
args = parser.parse_args()

# -----------------------------
# Paths
# -----------------------------
dataset_root = Path(args.data_root)
devices = ["Topcon_Maestro2", "Topcon_Maestro2_unlabeled", "Heidelberg_Spectralis", "Zeiss_Cirrus"]
labels = ["Diseased", "Healthy"]

# -----------------------------
# Summary
# -----------------------------
for device in devices:
    print(f"\nDevice: {device}")
    for label in labels:
        folder = dataset_root/device/label
        images = sorted(glob(str(folder/"*-image.png")))
        masks = sorted(glob(str(folder/"*-mask.png")))

        print(f"  {label}: {len(images)} images, {len(masks)} masks")

        # Check Maestro2 has 1:1 pairing
        if device == "Topcon_Maestro2":
            missing_mask = [img for img in images if img.replace("-image.png", "-mask.png") not in masks]
            if missing_mask:
                print(f"    Missing masks for {len(missing_mask)} images: {missing_mask[:5]} ...")
            else:
                print("    All images have masks")
            for img_f, mask_f in zip(images, masks):
                img = np.array(Image.open(img_f))
                mask = np.array(Image.open(mask_f))
                #print(img_f, img.dtype, mask_f, mask.dtype)
                assert img.shape[:2] == mask.shape[:2], "Image and mask shapes mismatch"
                # Check dtype for MONAI
                assert img.dtype in [np.uint8], f"Image dtype not float: {img_f} ({img.dtype})"
                assert mask.dtype in [np.uint8], f"Mask dtype not integer: {mask_f} ({mask.dtype})"
                # print unique class labels
                unique_labels = np.unique(mask)
                #print(f"{mask_f} -> unique labels: {unique_labels}")
        else:
            # Confirm no masks exist
            if masks:
                print(f"    Warning: Found unexpected masks in {device}/{label}!")
        




print("\nSanity check finished.")
