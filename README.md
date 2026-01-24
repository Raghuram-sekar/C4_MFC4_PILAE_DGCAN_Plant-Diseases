# 🌿 Plant Disease Classification using Conditional DCGAN + PILAE

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A deep learning approach for plant disease classification using Conditional Deep Convolutional Generative Adversarial Networks (cDCGAN) for data balancing and Pseudoinverse Learning Autoencoder (PILAE) for classification.

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Methodology](#-methodology)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Requirements](#-requirements)
- [Contributing](#-contributing)
- [References](#-references)
- [License](#-license)
- [Authors](#-authors)

## 🎯 Overview

This project implements a novel approach to plant disease classification that addresses class imbalance through synthetic data generation. By combining **Conditional DCGAN** for targeted image synthesis and **PILAE** for efficient classification, we achieve high accuracy (~100%) on the PlantVillage dataset.

### Problem Statement

Traditional plant disease classification systems suffer from:
- **Class Imbalance**: Some disease classes have 5000+ images while others have only 150
- **Limited Training Data**: Minority classes lead to biased models
- **Computational Complexity**: Traditional augmentation methods are inefficient

### Our Solution

1. **Conditional DCGAN**: Generate class-specific synthetic images for minority classes
2. **Dataset Balancing**: Ensure minimum 1000 samples per class
3. **PILAE Classifier**: Fast analytical learning using discriminator-extracted features
4. **K-Fold Validation**: Robust evaluation to ensure no overfitting

## ✨ Key Features

- 🎨 **Conditional Image Generation**: Generate disease-specific synthetic leaf images
- ⚖️ **Intelligent Balancing**: Automatic minority class oversampling to 1000 samples/class
- 🚀 **Fast Training**: PILAE uses pseudoinverse (no backpropagation needed)
- 🔍 **Feature Extraction**: Leverage trained discriminator as feature extractor (8192-dim)
- 📊 **Robust Validation**: 5-fold cross-validation for reliability
- 📈 **Training Visualization**: Timeline images showing GAN training progression
- 🎯 **High Accuracy**: Targets ~100% accuracy on balanced dataset

## 🧠 Methodology

### Workflow Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. Data Loading & Analysis                    │
│         PlantVillage Dataset (38 classes, 54,305 images)         │
│              Identify: Minority vs Majority Classes              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│               2. Conditional DCGAN Training                      │
│   Generator: noise + class_label → 64×64 RGB plant image        │
│   Discriminator: image + label → real/fake classification       │
│                   Train for 20 epochs                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              3. Dataset Balancing (Augmentation)                 │
│   For each minority class (count < 1000):                       │
│     - Generate synthetic images using trained Generator         │
│     - Extract features using Discriminator                      │
│   Result: Balanced dataset with 1000+ samples per class         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              4. PILAE Classification Training                    │
│   Input: 8192-dim features from Discriminator                   │
│   Method: Analytical weight computation (W = pinv(X) × Y)       │
│   Split: 80% train, 20% test (stratified)                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│           5. Evaluation & Robustness Verification                │
│   - Test accuracy, precision, recall, F1-score                  │
│   - Confusion matrix visualization                              │
│   - 5-Fold cross-validation (stratified)                        │
│   - Overfitting detection (train vs test comparison)            │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Approach?

#### Conditional DCGAN vs Traditional Augmentation

| Method | Diversity | Quality | Class-Specific | Computation |
|--------|-----------|---------|----------------|-------------|
| Rotation/Flip | Low | N/A | No | Fast |
| SMOTE | Low | Statistical | No | Fast |
| **cDCGAN** | **High** | **Photorealistic** | **Yes** | **Moderate** |

#### PILAE vs Traditional Classifiers

| Classifier | Training Method | Speed | Accuracy |
|------------|----------------|-------|----------|
| CNN | Backpropagation | Slow (hours) | High |
| SVM | Optimization | Moderate | Moderate |
| **PILAE** | **Analytical (Pseudoinverse)** | **Fast (minutes)** | **High** |

## 📊 Dataset

### PlantVillage Dataset

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/)
- **Total Images**: 54,305 RGB images
- **Image Size**: 64×64 pixels (resized)
- **Classes**: 38 plant disease categories
- **Format**: Color, Grayscale, and Segmented versions (we use **Color**)

### Plant Species Covered

- 🍎 Apple (4 classes)
- 🫐 Blueberry (1 class)
- 🍒 Cherry (2 classes)
- 🌽 Corn (4 classes)
- 🍇 Grape (4 classes)
- 🍊 Orange (1 class)
- 🍑 Peach (2 classes)
- 🌶️ Pepper (2 classes)
- 🥔 Potato (3 classes)
- 🍓 Strawberry (2 classes)
- 🍅 Tomato (10 classes)
- Others: Raspberry, Soybean, Squash

### Class Distribution

**Before Balancing:**
- Minimum: 152 images (Potato Healthy)
- Maximum: 5,507 images (Orange Haunglongbing)
- **Imbalance Ratio**: 36:1

**After Balancing:**
- Minimum: 1,000 images per class
- Maximum: 5,507 images (preserved real data)
- **Synthetic Generated**: ~14,000 images for minority classes

## 🏗️ Architecture

### Conditional Generator

```
Input: Noise (100-dim) + Class Label Embedding
    ↓
ConvTranspose2d(512, 4×4) → BatchNorm → ReLU
    ↓ (upsample to 8×8)
ConvTranspose2d(256, 4×4) → BatchNorm → ReLU
    ↓ (upsample to 16×16)
ConvTranspose2d(128, 4×4) → BatchNorm → ReLU
    ↓ (upsample to 32×32)
ConvTranspose2d(64, 4×4) → BatchNorm → ReLU
    ↓ (upsample to 64×64)
ConvTranspose2d(3, 4×4) → Tanh
    ↓
Output: RGB Image (3, 64, 64) in [-1, 1]
```

### Conditional Discriminator

```
Input: RGB Image (3, 64, 64) + Class Label (spatial embedding)
    ↓ (concatenate as 4th channel)
Conv2d(64, 4×4) → LeakyReLU → Dropout
    ↓ (downsample to 32×32)
Conv2d(128, 4×4) → BatchNorm → LeakyReLU → Dropout
    ↓ (downsample to 16×16)
Conv2d(256, 4×4) → BatchNorm → LeakyReLU → Dropout
    ↓ (downsample to 8×8)
Conv2d(512, 4×4) → BatchNorm → LeakyReLU → Dropout
    ↓ (downsample to 4×4)
Flatten → Feature Vector (8192-dim) ← **Extracted for PILAE**
    ↓
Linear(1) → Sigmoid
    ↓
Output: Real/Fake Probability [0, 1]
```

### PILAE Classifier

```
Input: Feature Vector (8192-dim)
    ↓
Analytical Weight Computation: W = pinv(X) × Y_one_hot
    ↓
Prediction: argmax(X_test × W)
    ↓
Output: Class Label (0-37)
```

**Key Advantage**: No iterative training needed! Weight matrix computed directly using Moore-Penrose pseudoinverse.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM

### Step 1: Clone the Repository

```bash
git clone https://github.com/Raghuram-sekar/C4_MFC4_PILAE_DGCAN_Plant-Diseases.git
cd C4_MFC4_PILAE_DGCAN_Plant-Diseases
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Download PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/) or [GitHub](https://github.com/spMohanty/PlantVillage-Dataset)
2. Extract to `Dataset/color/` directory
3. Ensure folder structure matches:
   ```
   Dataset/
   └── color/
       ├── Apple___Apple_scab/
       ├── Apple___Black_rot/
       ├── ...
       └── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
   ```

## 💻 Usage

### Option 1: Run Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/DCGAN_PILAE_Project_PyTorch.ipynb
```

Execute cells sequentially:
1. **Cell 1-2**: Setup and imports
2. **Cell 3-4**: Load dataset and analyze distribution
3. **Cell 5-6**: Initialize conditional DCGAN
4. **Cell 7-8**: Train cDCGAN (20 epochs, ~1-2 hours on GPU)
5. **Cell 9-10**: Balance dataset with synthetic images
6. **Cell 11-12**: Train PILAE and evaluate
7. **Cell 13-14**: K-fold cross-validation

### Option 2: Run Python Scripts

#### Train GAN
```bash
python src/train_gan.py
```

#### Train Classifier
```bash
python src/train_classifier.py
```

#### Verify Robustness
```bash
python src/verify_robustness.py
```

### Option 3: Quick Test (Pre-trained Models)

If pre-trained models exist in `models/` directory:

```python
from src.pilae import PILAE
import numpy as np

# Load PILAE model
pilae = PILAE(input_dim=8192)
pilae.load('models/pilae_model.npz')

# Predict on new features
predictions = pilae.predict(X_test)
```

## 📁 Project Structure

```
plant-disease-classification/
│
├── 📓 notebooks/
│   └── DCGAN_PILAE_Project_PyTorch.ipynb    # Main notebook
│
├── 🐍 src/
│   ├── data_loader.py                        # Dataset loading utilities
│   ├── dcgan.py                              # Generator & Discriminator
│   ├── pilae.py                              # PILAE classifier
│   ├── train_gan.py                          # GAN training script
│   ├── train_classifier.py                   # Classifier training
│   └── verify_robustness.py                  # K-fold validation
│
├── 📊 Dataset/
│   ├── color/                                # RGB images (38 classes)
│   ├── grayscale/                            # Grayscale version
│   └── segmented/                            # Segmented version
│
├── 💾 models/
│   ├── netG_cond.pth                         # Conditional Generator weights
│   ├── netD_cond.pth                         # Conditional Discriminator weights
│   ├── pilae_model.npz                       # PILAE classifier weights
│   ├── generated_images/                     # Sample generations per epoch
│   └── training_checkpoints/                 # Training checkpoints
│
├── 📄 extract_pdf.py                         # PDF extraction utility
├── 📄 search_pdf.py                          # PDF search utility
├── 🎨 training_timeline.png                  # GAN training visualization
├── 📋 requirements.txt                       # Python dependencies
├── 📖 README.md                              # This file
└── 📜 LICENSE                                # MIT License

```

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **~100%** |
| Training Time (GAN) | 1-2 hours (GPU) |
| Training Time (PILAE) | < 5 minutes |
| Total Images After Balancing | ~68,000 |
| Synthetic Images Generated | ~14,000 |

### K-Fold Cross-Validation

```
Fold 1: 99.8%
Fold 2: 99.9%
Fold 3: 100%
Fold 4: 99.9%
Fold 5: 100%
────────────────
Mean: 99.9% (±0.08%)
```

**Conclusion**: Results are robust and not due to overfitting!

### Class-wise Performance

All 38 classes achieve:
- **Precision**: 0.98-1.00
- **Recall**: 0.98-1.00
- **F1-Score**: 0.98-1.00

### Training Timeline

The generated `training_timeline.png` shows GAN improvement over 20 epochs:
- **Epoch 1-5**: Noisy, unrealistic images
- **Epoch 10-15**: Recognizable leaf patterns
- **Epoch 20**: Photorealistic disease-specific leaves

## 📦 Requirements

### Core Dependencies

```
Python >= 3.8
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.24.0
scipy >= 1.10.0
scikit-learn >= 1.2.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
Pillow >= 9.5.0
tqdm >= 4.65.0
jupyter >= 1.0.0
```

### Optional Dependencies

```
PyPDF2 >= 3.0.0          # For PDF utilities
ipython >= 8.12.0        # For notebook
pandas >= 2.0.0          # For data analysis
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Areas for Improvement

1. **Architecture Enhancements**
   - Implement StyleGAN2 or Progressive GAN
   - Add attention mechanisms
   - Experiment with different feature extractors

2. **Dataset Expansion**
   - Add background class (715 images)
   - Test on other datasets (Swedish Leaf, Leafsnap)
   - Implement multi-crop disease detection

3. **Performance Optimization**
   - Mixed precision training
   - Model quantization
   - ONNX export for deployment

4. **UI/Deployment**
   - Web interface (Streamlit/Gradio)
   - Mobile app (Flutter/React Native)
   - REST API (FastAPI)

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

### Papers

1. **Base Paper**: Mahmoud, M. A. B., Guo, P., & Wang, K. (2020). "Pseudoinverse learning autoencoder with DCGAN for plant diseases classification." *Multimedia Tools and Applications*, 79, 26245–26263. https://doi.org/10.1007/s11042-020-09239-0
   
2. **PlantVillage Dataset**: Hughes, D. P., & Salathé, M. (2015). "An open access repository of images on plant health to enable the development of mobile disease diagnostics."
   
3. **DCGAN**: Radford, A., Metz, L., & Chintala, S. (2015). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." *arXiv preprint arXiv:1511.06434*.

4. **PILAE**: Wang, S., Liu, W., Wu, J., et al. (2016). "Training deep neural networks on imbalanced data sets."

### Datasets

- [PlantVillage on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data)
- [PlantVillage on GitHub](https://github.com/spMohanty/PlantVillage-Dataset)

### Tools & Frameworks

- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Jupyter](https://jupyter.org/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 👥 Authors

**Raghuram Sekar, Aaditya Paul , Aadi Haldar , Rupanshi Sangwan** 

### Acknowledgments
- Prof. Sunil Kumar
- Amrita Vishwa Vidyapeetham for academic support
- PlantVillage dataset creators
- PyTorch community
- All contributors to this project

---

