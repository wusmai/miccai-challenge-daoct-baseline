import os
from glob import glob
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity,
    ToTensor, Resize
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
args = parser.parse_args()


# -----------------------------
# 1. Dataset paths
# -----------------------------
dataset_root = Path(args.data_root)
classes = 10  # 10-class segmentation

data = []
for label in ["Diseased", "Healthy"]:
    images = sorted(glob(str(dataset_root/label/"*-image.png")))
    for img_path in images:
        mask_path = img_path.replace("-image.png", "-mask.png")
        data.append({"image": img_path, "label": mask_path})

# -----------------------------
# 2. Transforms
# -----------------------------
train_transforms = Compose([
    LoadImage(image_only=True, reverse_indexing=False),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((256, 256)),
    ToTensor()
])

mask_transforms = Compose([
    LoadImage(image_only=True, reverse_indexing=False),
    EnsureChannelFirst(),
    Resize((256, 256), mode="nearest"),
    ToTensor()
])

# -----------------------------
# 3. Dataset and loader
# -----------------------------
dataset = CacheDataset(
    data=[{"image": d["image"], "label": d["label"]} for d in data],
    transform=Compose([
        lambda x: {"image": train_transforms(x["image"]), "label": mask_transforms(x["label"])}
    ]),
    cache_rate=1.0
)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------------
# 4. Model, loss, optimizer
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=classes,
    channels=(16,32,64,128),
    strides=(2,2,2),
    num_res_units=2
).to(device)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dice_metric = DiceMetric(include_background=True, reduction="mean")

# -----------------------------
# 5. Training loop (few iterations)
# -----------------------------
epochs = 5  # just a quick test
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    epoch_loss = 0
    for batch in loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        #print(labels.min().item(), labels.max().item())  # should be in 0..9
        #labels = labels // 10
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Loss: {epoch_loss/len(loader):.4f}")

# -----------------------------
# 6. Quick evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    for batch in loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        #labels = labels // 10
        outputs = sliding_window_inference(inputs, (256,256), 1, model)
        metric = dice_metric(y_pred=outputs, y=labels)
        print("Dice per class:", metric.cpu().numpy())
        break  # just check first batch

# -----------------------------
# 7. Save model
# -----------------------------
os.makedirs("checkpoints", exist_ok=True)

model_path = "checkpoints/unet_maestro2.pth"

torch.save(model.state_dict(), model_path)

print(f"[INFO] Model saved to {model_path}")
