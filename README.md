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
# Loading dataset: 100%|███████████████████████████████████████████████████████| 230/230 [00:00<00:00, 1437689.90it/s]
# Epoch 1/5
# Loss: 0.5358
# Epoch 2/5
# Loss: 0.2059
# Epoch 3/5
# Loss: 0.1657
# Epoch 4/5
# Loss: 0.1511
# Epoch 5/5
# Loss: 0.1432
# Dice per class: [[0.63997924 0.05056815 0.05111888 0.05738677 0.03837289 0.04033789
#   0.04673343 0.04827494 0.04145062 0.504956  ]
#  [0.19913991 0.05689719 0.05568659 0.06069215 0.04670432 0.04708266
#   0.0512058  0.03023654 0.03260184 0.819352  ]]
# [INFO] Model saved to checkpoints/unet_maestro2.pth

```

### 3. Semi-supervised training
```bash
python scripts/train_test_monai_semi.py --data_root release_dataset/

## Expected output: 
# Loading dataset: 100%|███████████████████████████████████████████████████████| 230/230 [00:00<00:00, 1504976.47it/s]
# Loading dataset: 100%|██████████████████████████████████████████████████████| 1099/1099 [00:00<00:00, 173297.50it/s]
# Epoch 1/5
# Supervised loss: 0.5232
# Epoch 2/5
# Supervised loss: 0.2127
# Epoch 3/5
# Supervised loss: 0.1838
# Epoch 4/5
# Supervised loss: 0.1580
# Epoch 5/5
# Supervised loss: 0.1437
# Dice per class: [[0.40216503 0.07770038 0.07832041 0.08531027 0.0499881  0.05039417
#   0.05583079 0.05100296 0.04673343 0.6597681 ]
#  [0.47707573 0.07916526 0.07648733 0.08208835 0.04209426 0.04294204
#   0.04961088 0.03702151 0.03957582 0.6254601 ]]
# [INFO] Model saved to checkpoints/unet_maestro2_semi.pth
# Batch shape: torch.Size([2, 1, 256, 256])
# Output shape: torch.Size([2, 10, 256, 256])
```

### 4. Inference

Inference is performed on a directory of raw input images. No labels or metadata are required.

The input directory may contain arbitrary or unseen data. The model must operate using image-only input without any domain or class information.
Predictions are saved as per-image segmentation masks in output_dir using the same filename convention as input images.

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
docker build -t bearceb/daoct-baseline:latest .

# Run (mounts your local directory into the container)
docker run --rm -it -v $(pwd):/app bearceb/daoct-baseline:latest

# Then inside the container:
python scripts/sanity_check_dataset.py
python scripts/train_test_monai.py
python scripts/train_test_monai_semi.py
```