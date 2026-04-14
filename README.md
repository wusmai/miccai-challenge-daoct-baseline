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
├── sanity_check_data.py        # Verify dataset consistency (shape, dtype, pairing)
├── train_test_monai.py        # Supervised training (Maestro2 with labels)
├── train_test_monai_semi.py   # Semi-supervised training (adds unlabeled data)

## Dataset Assumptions

- Images and masks follow the naming convention:
  xxx-image.png
  xxx-mask.png
- Labels are integer encoded per pixel
- Only Maestro2 images have labels
- Other devices (e.g., Cirrus, Spectralis) are unlabeled

## Usage

### 1. Sanity check dataset
python scripts/sanity_check_data.py

### 2. Supervised training
python scripts/train_test_monai.py

### 3. Semi-supervised training
python scripts/train_test_monai_semi.py

## Notes

- This is a minimal baseline for debugging and experimentation
- Not optimized for performance
- Designed to match the expected dataset structure of the challenge
- This repository is for local prototyping only and does not reflect final challenge submission code
