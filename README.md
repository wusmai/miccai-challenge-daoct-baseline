# miccai-challenge-daoct-baseline
This repository provides a simple MONAI-based baseline pipeline for MICCAI 2026 DAOCT challenge.  


## Overview

The goal is to develop and validate segmentation pipelines for retinal OCT images under a domain adaptation setting.

This repo is intended for:
- sanity checking the dataset
- verifying data loading and preprocessing
- testing supervised and semi-supervised training pipelines

## Structure

scripts/
  - sanity_check_data.py        # Verify dataset consistency (shape, dtype, pairing)
  - train_test_monai.py         # Supervised training (Maestro2 with labels)
  - train_test_monai_semi.py    # Semi-supervised training (adds unlabeled data)


## Dataset Assumptions

- Images and masks follow the naming convention:
  xxx-image.png
  xxx-mask.png
- Labels are integer encoded per pixel
---

### Device-wise structure

- `Topcon_Maestro2`
  - Contains labeled data with image-mask pairs
  - Subdivided into:
    - `Diseased`
    - `Healthy`

- `Topcon_Maestro2_unlabeled`
  - Contains images only (no masks provided)
  - Used for semi-supervised learning (if enabled)

- `Heidelberg_Spectralis`
  - Unlabeled dataset (no masks provided)

- `Zeiss_Cirrus`
  - Unlabeled dataset (no masks provided)

---

### Training assumptions

- Supervised training uses only `Topcon_Maestro2` labeled data
- Semi-supervised training may additionally use `Topcon_Maestro2_unlabeled`
- Other devices are intended for domain generalization / adaptation studies

---

### Inference assumptions

- No device/domain label is provided at inference time
- Models must operate without access to metadata such as device type or disease label
- Only image input is allowed for prediction


## Usage

### Python venv setup
```bash
pyenv virtualenv 3.10.0 miccai_satelite_events
pyenv activate miccai_satelite_events
# pyenv only
pip install .
# poetry
poetry install 
```

### 0. Sanity Check
```bash
md5sum -c md5_v1.0.txt # checksum
tar -xvf data_synthetic_v1.0.tar # unzip
```

### 1. Sanity check dataset
```bash
python scripts/sanity_check_dataset.py --data_root release_dataset/

## Expected output:
# Device: Topcon_Maestro2
#   Diseased: 57 images, 57 masks
#     All images have masks
#   Healthy: 173 images, 173 masks
#     All images have masks

# Device: Topcon_Maestro2_unlabeled
#   Diseased: 166 images, 0 masks
#   Healthy: 496 images, 0 masks

# Device: Heidelberg_Spectralis
#   Diseased: 56 images, 0 masks
#   Healthy: 169 images, 0 masks

# Device: Zeiss_Cirrus
#   Diseased: 53 images, 0 masks
#   Healthy: 159 images, 0 masks

# Sanity check finished.



```

### 2. Supervised training
```bash
python scripts/train_test_monai.py --data_root release_dataset/Topcon_Maestro2

## Expected output:  
# Loading dataset: 100%|██████████████████████████████████████████████████████████████████████████████████| 230/230 [00:00<00:00, 1435550.48it/s]
# Epoch 1/5
# Loss: 0.5693
# Epoch 2/5
# Loss: 0.2140
# Epoch 3/5
# Loss: 0.1703
# Epoch 4/5
# Loss: 0.1531
# Epoch 5/5
# Loss: 0.1371
# Dice per class: [[0.9480319  0.8957055  0.8240506  0.8635193  0.8513873  0.8512035
#   0.8992927  0.9362319  0.8779139  0.93587047]
#  [0.9710476  0.87990314 0.8099973  0.8975318  0.90161663 0.84085107
#   0.8626172  0.93398136 0.9047619  0.9909789 ]]
# [INFO] Model saved to checkpoints/unet_maestro2.pth
```

### 3. Semi-supervised training
```bash
python scripts/train_test_monai_semi.py --data_root release_dataset/

## Expected output: 
# Supervised samples: 230, Unlabeled samples: 1099
# Loading dataset: 100%|██████████████████████████████████████████████████████████████████████████████████| 230/230 [00:00<00:00, 1479585.77it/s]
# Loading dataset: 100%|█████████████████████████████████████████████████████████████████████████████████| 1099/1099 [00:00<00:00, 181821.56it/s]
# Epoch 1/5
# Supervised loss: 0.5335
# Epoch 2/5
# Supervised loss: 0.2143
# Epoch 3/5
# Supervised loss: 0.1730
# Epoch 4/5
# Supervised loss: 0.1535
# Epoch 5/5
# Supervised loss: 0.1469
# Dice per class: [[0.99307853 0.8676269  0.90450203 0.92600685 0.9015189  0.91910005
#   0.94293016 0.9206212  0.8669355  0.99625546]
#  [0.796272   0.9221201  0.9156569  0.92600423 0.8668363  0.89321226
#   0.9310345  0.95404667 0.9241012  0.9457579 ]]
# [INFO] Model saved to checkpoints/unet_maestro2_semi.pth
# Batch shape: torch.Size([2, 1, 256, 256])
# Output shape: torch.Size([2, 10, 256, 256])
```

### 4. Inference

Inference is performed on a directory of raw input images. No labels or metadata are required.

The input directory may contain arbitrary or unseen data. The model must operate using image-only input without any domain or class information.
Predictions are saved as per-image segmentation masks in output_dir using the same filename convention as input images.
Participants may output segmentation maps at any resolution.
During evaluation, all predicted masks will be resized to the corresponding ground-truth resolution using **nearest-neighbor interpolation** prior to metric computation.
This ensures consistent comparison across submissions while allowing flexibility in model design and inference pipeline.

```bash
python scripts/infer_test_monai.py \
  --input_dir external_test/ \
  --output_dir pred/ \
  --model_path checkpoints/unet_maestro2.pth

## Expected output: 
# [INFO] Loading checkpoint: checkpoints/unet_maestro2.pth
# [INFO] Found N images
# [INFO] Inference complete.
```



## Notes

- This is a minimal baseline for debugging and experimentation
- Not optimized for performance
- Designed to match the expected dataset structure of the challenge
- This repository is for local prototyping only and does not reflect final challenge submission code


## Template Submission
```bash
# Build
docker build -t daoct-baseline .

# Run (mounts your local directory into the container)
docker run --rm -it -v $(pwd):/app daoct-baseline

# Then inside the container:
python scripts/train.py
python scripts/inference.py
```
The -v $(pwd):/app flag mounts the local repository into the Docker container at runtime, ensuring reproducibility without embedding data or code into the image.

Participants are permitted to use publicly available pretrained models and foundation model checkpoints.
To ensure secure, reproducible, and stable evaluation, submitted code will execute in an isolated offline environment without internet access.
All required pretrained checkpoints and dependencies must be included locally within the submission package.
External downloads during runtime are prohibited.
