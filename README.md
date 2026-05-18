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
- Only Maestro2 images have labels
- Other devices (e.g., Cirrus, Spectralis) are unlabeled


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
md5sum -c md5.txt # checksum
tar -xvf data_synthetic_v1.tar # unzip
```

### 1. Sanity check dataset
```bash
python scripts/sanity_check_dataset.py

## Expected output:
# Device: Topcon_Maestro2
#   Diseased: 223 images, 223 masks
#     All images have masks
#   Healthy: 669 images, 669 masks
#     All images have masks

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
python scripts/train_test_monai.py

# Loading dataset: 100%|██████████████████████████████████████████████████████████████████████████| 892/892 [00:00<00:00, 32475.88it/s]
# Epoch 1/5
# Loss: 0.2865
# Epoch 2/5
# Loss: 0.1420
# Epoch 3/5
# Loss: 0.1243
# Epoch 4/5
# Loss: 0.1144
# Epoch 5/5
# Loss: 0.1041

# Dice per class: [[0.576202   0.05536928 0.05652264 0.06089298 0.03746238 0.03734483
#   0.04235744 0.04332184 0.0342247  0.5785798 ]
#  [0.31262392 0.05934261 0.06404348 0.07479216 0.04080657 0.04142136
#   0.04786798 0.04416856 0.04089442 0.75176656]]
```

### 3. Semi-supervised training
```bash
python scripts/train_test_monai_semi.py

# Supervised samples: 892, Unlabeled samples: 437
# Loading dataset: 100%|██████████████████████████████████████████████████████████████████████████| 892/892 [00:00<00:00, 37451.77it/s]
# Loading dataset: 100%|██████████████████████████████████████████████████████████████████████████| 437/437 [00:00<00:00, 30624.57it/s]
# Epoch 1/5
# Supervised loss: 0.3621
# Epoch 2/5
# Supervised loss: 0.2410
# Epoch 3/5
# Supervised loss: 0.2270
# Epoch 4/5
# Supervised loss: 0.2150
# Epoch 5/5
# Supervised loss: 0.2110

# Dice per class: [[0.24741937 0.05195886 0.05123478 0.05712761 0.03254278 0.03295615
#   0.03737422 0.03345786 0.0313312  0.8174026 ]
#  [0.672823   0.0643008  0.06584325 0.07184678 0.04760628 0.04635495
#   0.05305843 0.04705356 0.03939988 0.4239145 ]]
# Batch shape: torch.Size([2, 1, 256, 256])
# Output shape: torch.Size([2, 10, 256, 256])
```

## Notes

- This is a minimal baseline for debugging and experimentation
- Not optimized for performance
- Designed to match the expected dataset structure of the challenge
- This repository is for local prototyping only and does not reflect final challenge submission code
