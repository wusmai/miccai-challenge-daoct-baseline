import os
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import torch
import cv2

from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize

# -----------------------------
# Determinism
# -----------------------------
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------------------------
# Model definition
# -----------------------------
def build_model(num_classes=10):
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=num_classes,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2
    )
    return model


# -----------------------------
# Preprocess
# -----------------------------
preprocess = Compose([
    LoadImage(image_only=True, reverse_indexing=False),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((256, 256))
])


# -----------------------------
# Inference function
# -----------------------------
def run_inference(model, image_path, device):
    img = preprocess(image_path)
    img = torch.as_tensor(img).unsqueeze(0).to(device)  # [1,1,H,W]

    with torch.no_grad():
        logits = model(img)
        pred = torch.argmax(logits, dim=1)  # [1,H,W]

    return pred[0].cpu().numpy().astype(np.uint8)


# -----------------------------
# Main
# -----------------------------
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = build_model(num_classes=10).to(device)

    # load weights
    print(f"[INFO] Loading checkpoint: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # input images
    image_paths = sorted(glob(os.path.join(args.input_dir, "*-image.png")))

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Found {len(image_paths)} images")

    for img_path in image_paths:

        fname = os.path.basename(img_path)
        out_mask_path = os.path.join(
            args.output_dir,
            fname.replace("-image.png", "-mask.png")
        )
        # run inference
        pred_mask = run_inference(model, img_path, device)
        cv2.imwrite(out_mask_path, pred_mask.astype(np.uint8))
        
    print("[INFO] Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
