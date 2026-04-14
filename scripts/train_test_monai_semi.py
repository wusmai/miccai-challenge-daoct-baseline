import os
from glob import glob
from pathlib import Path
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, ToTensor, Resize
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# -----------------------------
# 1. Dataset paths
# -----------------------------
dataset_root = Path("dataset")
supervised_device = "Topcon_Maestro2"
unlabeled_devices = ["Heidelberg_Spectralis", "Zeiss_Cirrus"]
classes = 10  # 10-class segmentation

# -----------------------------
# 2. Prepare supervised data (Maestro2)
# -----------------------------
supervised_data = []
for label in ["Diseased", "Healthy"]:
    images = sorted(glob(str(dataset_root/supervised_device/label/"*-image.png")))
    for img_path in images:
        mask_path = img_path.replace("-image.png", "-mask.png")
        supervised_data.append({"image": img_path, "label": mask_path})

# -----------------------------
# 3. Prepare unlabeled data (optional)
# -----------------------------
unlabeled_data = []
for device in unlabeled_devices:
    for label in ["Diseased", "Healthy"]:
        images = sorted(glob(str(dataset_root/device/label/"*-image.png")))
        for img_path in images:
            unlabeled_data.append({"image": img_path})

print(f"Supervised samples: {len(supervised_data)}, Unlabeled samples: {len(unlabeled_data)}")

# -----------------------------
# 4. Transforms
# -----------------------------
image_transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((256, 256)),
    ToTensor()
])

mask_transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Resize((256, 256), mode="nearest"),
    ToTensor()
])

def supervised_transform(item):
    return {
        "image": image_transform(item["image"]),
        "label": mask_transform(item["label"])
    }

def unlabeled_transform(item):
    return {
        "image": image_transform(item["image"])
    }

# -----------------------------
# 5. Dataset and DataLoader
# -----------------------------
sup_dataset = CacheDataset(data=supervised_data, transform=supervised_transform, cache_rate=1.0)
sup_loader = DataLoader(sup_dataset, batch_size=2, shuffle=True)

# Optional: unlabeled loader for semi-supervised training
ul_dataset = CacheDataset(data=unlabeled_data, transform=unlabeled_transform, cache_rate=1.0)
ul_loader = DataLoader(ul_dataset, batch_size=2, shuffle=True)

# -----------------------------
# 6. Model, loss, optimizer
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
# 7. Training loop (supervised only)
# -----------------------------
epochs = 5  # quick test
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    epoch_loss = 0
    for batch in sup_loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        #labels = labels // 10
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Supervised loss: {epoch_loss/len(sup_loader):.4f}")

# -----------------------------
# 8. Quick evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    for batch in sup_loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        #labels = labels // 10
        outputs = sliding_window_inference(inputs, (256,256), 1, model)
        metric = dice_metric(y_pred=outputs, y=labels)
        print("Dice per class:", metric.cpu().numpy())
        break
# -----------------------------
# Test loop to check loading (unlabeled only)
# -----------------------------
for batch in ul_loader:
    imgs = batch["image"].to(device)
    print("Batch shape:", imgs.shape)
    # just pass through model to test
    with torch.no_grad():
        outputs = model(imgs)
        print("Output shape:", outputs.shape)
    break  # just first batch