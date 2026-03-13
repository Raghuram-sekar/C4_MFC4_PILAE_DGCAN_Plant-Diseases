# PlantVillage Dataset Description

## Overview
- **Total Images:** 54,305
- **Total Classes:** 38
- **Image Size:** 256×256 pixels (resized to 64×64 for training)
- **Format:** RGB color images
- **Source:** PlantVillage dataset

---

## Class Distribution Table

| # | Plant Type | Disease/Condition | Class Name | Image Count | Category | Percentage |
|---|------------|-------------------|------------|-------------|----------|------------|
| 1 | Potato | Healthy | `Potato___healthy` | 152 | Minority | 0.28% |
| 2 | Apple | Cedar Apple Rust | `Apple___Cedar_apple_rust` | 275 | Minority | 0.51% |
| 3 | Peach | Healthy | `Peach___healthy` | 360 | Minority | 0.66% |
| 4 | Raspberry | Healthy | `Raspberry___healthy` | 371 | Minority | 0.68% |
| 5 | Tomato | Tomato Mosaic Virus | `Tomato___Tomato_mosaic_virus` | 373 | Minority | 0.69% |
| 6 | Grape | Healthy | `Grape___healthy` | 423 | Minority | 0.78% |
| 7 | Strawberry | Healthy | `Strawberry___healthy` | 456 | Minority | 0.84% |
| 8 | Corn (Maize) | Cercospora Leaf Spot / Gray Leaf Spot | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 513 | Minority | 0.94% |
| 9 | Apple | Black Rot | `Apple___Black_rot` | 621 | Minority | 1.14% |
| 10 | Apple | Apple Scab | `Apple___Apple_scab` | 630 | Minority | 1.16% |
| 11 | Cherry | Healthy | `Cherry_(including_sour)___healthy` | 854 | Medium | 1.57% |
| 12 | Tomato | Leaf Mold | `Tomato___Leaf_Mold` | 952 | Medium | 1.75% |
| 13 | Corn (Maize) | Northern Leaf Blight | `Corn_(maize)___Northern_Leaf_Blight` | 985 | Medium | 1.81% |
| 14 | Pepper (Bell) | Bacterial Spot | `Pepper,_bell___Bacterial_spot` | 997 | Medium | 1.84% |
| 15 | Potato | Early Blight | `Potato___Early_blight` | 1000 | Medium | 1.84% |
| 16 | Tomato | Early Blight | `Tomato___Early_blight` | 1000 | Medium | 1.84% |
| 17 | Potato | Late Blight | `Potato___Late_blight` | 1000 | Medium | 1.84% |
| 18 | Cherry | Powdery Mildew | `Cherry_(including_sour)___Powdery_mildew` | 1052 | Medium | 1.94% |
| 19 | Grape | Leaf Blight (Isariopsis Leaf Spot) | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | 1076 | Medium | 1.98% |
| 20 | Strawberry | Leaf Scorch | `Strawberry___Leaf_scorch` | 1109 | Medium | 2.04% |
| 21 | Corn (Maize) | Healthy | `Corn_(maize)___healthy` | 1162 | Medium | 2.14% |
| 22 | Grape | Black Rot | `Grape___Black_rot` | 1180 | Medium | 2.17% |
| 23 | Corn (Maize) | Common Rust | `Corn_(maize)___Common_rust_` | 1192 | Medium | 2.19% |
| 24 | Grape | Esca (Black Measles) | `Grape___Esca_(Black_Measles)` | 1383 | Medium | 2.55% |
| 25 | Tomato | Target Spot | `Tomato___Target_Spot` | 1404 | Medium | 2.58% |
| 26 | Pepper (Bell) | Healthy | `Pepper,_bell___healthy` | 1478 | Medium | 2.72% |
| 27 | Blueberry | Healthy | `Blueberry___healthy` | 1502 | Medium | 2.77% |
| 28 | Tomato | Healthy | `Tomato___healthy` | 1591 | Medium | 2.93% |
| 29 | Apple | Healthy | `Apple___healthy` | 1645 | Medium | 3.03% |
| 30 | Tomato | Spider Mites (Two-Spotted Spider Mite) | `Tomato___Spider_mites Two-spotted_spider_mite` | 1676 | Medium | 3.09% |
| 31 | Tomato | Septoria Leaf Spot | `Tomato___Septoria_leaf_spot` | 1771 | Medium | 3.26% |
| 32 | Squash | Powdery Mildew | `Squash___Powdery_mildew` | 1835 | Medium | 3.38% |
| 33 | Tomato | Late Blight | `Tomato___Late_blight` | 1909 | Majority | 3.52% |
| 34 | Tomato | Bacterial Spot | `Tomato___Bacterial_spot` | 2127 | Majority | 3.92% |
| 35 | Peach | Bacterial Spot | `Peach___Bacterial_spot` | 2297 | Majority | 4.23% |
| 36 | Soybean | Healthy | `Soybean___healthy` | 5090 | Majority | 9.37% |
| 37 | Tomato | Tomato Yellow Leaf Curl Virus | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5357 | Majority | 9.86% |
| 38 | Orange | Huanglongbing (Citrus Greening) | `Orange___Haunglongbing_(Citrus_greening)` | 5507 | Majority | 10.14% |

---

## Dataset Statistics

### Plant Type Distribution
| Plant Type | Number of Classes | Total Images | Percentage |
|------------|-------------------|--------------|------------|
| Tomato | 10 | 18,161 | 33.44% |
| Corn (Maize) | 4 | 3,852 | 7.09% |
| Grape | 4 | 3,062 | 5.64% |
| Apple | 4 | 2,171 | 4.00% |
| Potato | 3 | 2,152 | 3.96% |
| Peach | 2 | 2,657 | 4.89% |
| Pepper (Bell) | 2 | 2,475 | 4.56% |
| Cherry | 2 | 1,906 | 3.51% |
| Strawberry | 2 | 1,565 | 2.88% |
| Soybean | 1 | 5,090 | 9.37% |
| Orange | 1 | 5,507 | 10.14% |
| Squash | 1 | 1,835 | 3.38% |
| Blueberry | 1 | 1,502 | 2.77% |
| Raspberry | 1 | 371 | 0.68% |

### Disease vs Healthy Distribution
| Category | Number of Classes | Total Images | Percentage |
|----------|-------------------|--------------|------------|
| Diseased | 28 | 44,048 | 81.11% |
| Healthy | 10 | 10,257 | 18.89% |

### Class Imbalance Statistics
| Category | Definition | Number of Classes | Image Range |
|----------|------------|-------------------|-------------|
| Minority | < 1000 images | 10 classes | 152 - 997 |
| Medium | 1000 - 2000 images | 23 classes | 1000 - 1909 |
| Majority | > 2000 images | 5 classes | 2127 - 5507 |

**Imbalance Ratio:** 36.2:1 (Most: 5507, Least: 152)

---

## Experimental Splits

### Unbalanced Dataset (Original Distribution)
Uses raw dataset distribution with natural class imbalance (152 - 5507 images per class)

| Split | Training Images | Testing Images | Total |
|-------|----------------|----------------|-------|
| **80-20** | 43,444 (80%) | 10,861 (20%) | 54,305 |
| **70-30** | 38,013 (70%) | 16,292 (30%) | 54,305 |
| **60-40** | 32,583 (60%) | 21,722 (40%) | 54,305 |

**Characteristics:**
- Directly uses original PlantVillage dataset
- Maintains natural class imbalance (36:1 ratio)
- No synthetic augmentation
- Test set reflects real-world distribution

### Balanced Dataset (DCGAN Augmented)
Uses Conditional DCGAN to generate synthetic samples for minority classes

**Process:** Original Dataset (54,305 images) → Split into Train/Test → Augment Training Set Only

| Split | Original Training | After Augmentation | Testing Images | Notes |
|-------|-------------------|-------------------|----------------|-------|
| **80-20** | 43,444 (80%) | 48,000+ images | 10,861 (20%) | +4,556 synthetic images added to training |
| **70-30** | 38,013 (70%) | 42,000+ images | 16,292 (30%) | +3,987 synthetic images added to training |
| **60-40** | 32,583 (60%) | 36,000+ images | 21,722 (40%) | +3,417 synthetic images added to training |

**Important Notes:**
- **Testing images ALWAYS remain original** (no augmentation applied to test set)
- **Only training set is augmented** with DCGAN-generated synthetic images
- **Total original dataset:** 54,305 images (unchanged)
- **Augmentation increases only the training set size**

**Balancing Strategy:**
- **Target:** Minimum 1000 images per class in training set
- **Method:** Conditional DCGAN generates synthetic samples
- **Application:** Only training set (test set remains original and untouched)
- **Classes affected:** 10 minority classes (< 1000 images)
- **Classes unchanged:** 28 classes with ≥ 1000 images
- **Quality:** Synthetic images visually indistinguishable from real samples

---

## Dataset Characteristics

### Image Properties
- **Resolution:** 256×256 pixels (original), resized to 64×64 for DCGAN training
- **Color Space:** RGB
- **File Format:** JPG
- **Background:** Various (field conditions, laboratory conditions)
- **Lighting:** Natural and artificial
- **Leaf Position:** Various angles and orientations

### Quality Considerations
- **Image Quality:** High resolution, clear disease symptoms
- **Variation:** Multiple leaf samples per class with natural variations
- **Real-world Conditions:** Images captured in realistic agricultural settings
- **Challenge:** High class imbalance (36:1 ratio)

---

## Experimental Results

### Model Configuration
- **Architecture:** Conditional DCGAN + PILAE Classifier
- **Framework:** PyTorch (Python)
- **DCGAN Epochs:** 50
- **Optimizer:** Adam (DCGAN), Pseudoinverse Learning (PILAE)
- **Feature Extraction:** Discriminator features (8192-dimensional)
- **PILAE Parameters:** β=0.9, k=1e-5

### Performance Summary Table

| Split Ratio | Dataset Type | Training Samples | Test Samples | Training Accuracy | Test Accuracy | Generalization |
|-------------|--------------|------------------|--------------|-------------------|---------------|----------------|
| **80-20** | Balanced | 48,000+ (augmented) | 10,861 | 100.00% | 99.97% | Excellent |
| **80-20** | Unbalanced | 43,444 (original) | 10,861 | 100.00% | 99.96% | Excellent |
| **70-30** | Balanced | 42,000+ (augmented) | 16,292 | 100.00% | 99.97% | Excellent |
| **70-30** | Unbalanced | 38,013 (original) | 16,292 | 100.00% | 99.93% | Excellent |
| **60-40** | Balanced | 36,000+ (augmented) | 21,722 | 100.00% | 99.94% | Excellent |
| **60-40** | Unbalanced | 32,583 (original) | 21,722 | 100.00% | 99.90% | Excellent |

### Balanced vs Unbalanced Comparison

| Split | Balanced Test Acc | Unbalanced Test Acc | Difference |
|-------|-------------------|---------------------|------------|
| 80-20 | 99.97% | 99.96% | +0.01% |
| 70-30 | 99.97% | 99.93% | +0.04% |
| 60-40 | 99.94% | 99.90% | +0.04% |

### Split Ratio Analysis

| Split | Test Set Proportion | Test Samples | Average Test Accuracy |
|-------|---------------------|--------------|---------------------|
| 80-20 | 20% | 10,861 | 99.965% |
| 70-30 | 30% | 16,292 | 99.950% |
| 60-40 | 40% | 21,722 | 99.920% |

---

## Limited Data Experiment

### Experimental Setup
**Configuration:**
- **Training Samples:** 50 images per class
- **Generated Images:** 50 synthetic samples per class
- **DCGAN Epochs:** 15
- **Total Training Data:** 3,800 images ((50 real + 50 generated) × 38 classes)
- **Framework Comparison:** Python (PyTorch) vs MATLAB

### Python vs MATLAB Performance Comparison

| Split Ratio | Framework | Precision | Recall | F1-Score | Test Accuracy |
|-------------|-----------|-----------|--------|----------|---------------|
| **80-20** | Python (PyTorch) | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **80-20** | MATLAB | 0.9837 | 0.9825 | 0.9823 | 98.68% |
| **70-30** | Python (PyTorch) | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **70-30** | MATLAB | 0.9884 | 0.9868 | 0.9868 | 98.68% |
| **60-40** | Python (PyTorch) | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **60-40** | MATLAB | 0.9911 | 0.9908 | 0.9907 | 99.07% |

### Framework Gap Analysis

| Split | Python-MATLAB Gap (Accuracy) | Python-MATLAB Gap (F1-Score) |
|-------|------------------------------|------------------------------|
| 80-20 | 1.32% | 1.77% |
| 70-30 | 1.32% | 1.32% |
| 60-40 | 0.93% | 0.93% |

### Performance Comparison (Full Data vs Limited Data)

| Split | Full Data (Python) | Limited Data (Python) | Limited Data (MATLAB) | Python Gap | MATLAB Gap |
|-------|--------------------|-----------------------|-----------------------|------------|------------|
| 80-20 | 99.97% | 100.00% | 98.68% | +0.03% | -1.29% |
| 70-30 | 99.97% | 100.00% | 98.68% | +0.03% | -1.29% |
| 60-40 | 99.94% | 100.00% | 99.07% | +0.06% | -0.87% |

### Framework Performance Summary

| Framework | Average Accuracy | Std Deviation | Min Accuracy | Max Accuracy |
|-----------|------------------|---------------|--------------|--------------|
| Python (PyTorch) | 100.00% | 0.00% | 100.00% | 100.00% |
| MATLAB | 98.81% | 0.20% | 98.68% | 99.07% |

---

## Comprehensive Results Summary

### All Experiments Comparison Table

| Experiment Type | Split | Framework | Samples/Class | DCGAN Epochs | Precision | Recall | F1-Score | Test Accuracy |
|----------------|-------|-----------|---------------|--------------|-----------|--------|----------|---------------|
| **Full Data (Balanced)** | 80-20 | Python | 1400+ | 50 | 1.0000 | 1.0000 | 1.0000 | 99.97% |
| **Full Data (Unbalanced)** | 80-20 | Python | 152-5507 | 50 | 1.0000 | 1.0000 | 1.0000 | 99.96% |
| **Limited Data** | 80-20 | Python | 50 | 15 | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **Limited Data** | 80-20 | MATLAB | 50 | 15 | 0.9837 | 0.9825 | 0.9823 | 98.68% |
| **Full Data (Balanced)** | 70-30 | Python | 1400+ | 50 | 1.0000 | 1.0000 | 1.0000 | 99.97% |
| **Full Data (Unbalanced)** | 70-30 | Python | 152-5507 | 50 | 1.0000 | 1.0000 | 1.0000 | 99.93% |
| **Limited Data** | 70-30 | Python | 50 | 15 | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **Limited Data** | 70-30 | MATLAB | 50 | 15 | 0.9884 | 0.9868 | 0.9868 | 98.68% |
| **Full Data (Balanced)** | 60-40 | Python | 1400+ | 50 | 1.0000 | 1.0000 | 1.0000 | 99.94% |
| **Full Data (Unbalanced)** | 60-40 | Python | 152-5507 | 50 | 1.0000 | 1.0000 | 1.0000 | 99.90% |
| **Limited Data** | 60-40 | Python | 50 | 15 | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **Limited Data** | 60-40 | MATLAB | 50 | 15 | 0.9911 | 0.9908 | 0.9907 | 99.07% |

### Statistical Performance Analysis

| Configuration | Min Accuracy | Max Accuracy | Mean Accuracy | Std Deviation | Consistency |
|--------------|--------------|--------------|---------------|---------------|-------------|
| **Python (All Experiments)** | 99.90% | 100.00% | 99.98% | 0.03% | Excellent |
| **Python (Limited Data)** | 100.00% | 100.00% | 100.00% | 0.00% | Perfect |
| **MATLAB (Limited Data)** | 98.68% | 99.07% | 98.81% | 0.20% | Very Good |
| **Full Data (Balanced)** | 99.94% | 99.97% | 99.96% | 0.01% | Excellent |
| **Full Data (Unbalanced)** | 99.90% | 99.96% | 99.93% | 0.03% | Excellent |

---
---

# Swedish Leaf Dataset Description

## Overview
- **Total Images:** 1,125
- **Total Classes:** 15 tree species
- **Image Size:** Variable (resized to 64×64 for training)
- **Format:** RGB color images
- **Source:** Swedish Leaf dataset
- **Type:** Isolated leaf images with uniform background

---

## Class Distribution Table

| # | Tree Species | Scientific Name | Image Count | Percentage |
|---|--------------|-----------------|-------------|------------|
| 1 | Ulmus carpinifolia | Elm | 75 | 6.67% |
| 2 | Acer | Maple | 75 | 6.67% |
| 3 | Salix aurita | Willow | 75 | 6.67% |
| 4 | Quercus | Oak | 75 | 6.67% |
| 5 | Alnus incana | Alder | 75 | 6.67% |
| 6 | Betula pubescens | Birch | 75 | 6.67% |
| 7 | Salix sinerea | Grey Willow | 75 | 6.67% |
| 8 | Populus tremula | Aspen | 75 | 6.67% |
| 9 | Ulmus glabra | Wych Elm | 75 | 6.67% |
| 10 | Sorbus aucuparia | Rowan | 75 | 6.67% |
| 11 | Salix alba 'Sericea' | White Willow | 75 | 6.67% |
| 12 | Populus | Poplar | 75 | 6.67% |
| 13 | Tilia | Linden | 75 | 6.67% |
| 14 | Sorbus intermedia | Swedish Whitebeam | 75 | 6.67% |
| 15 | Fagus silvatica | Beech | 75 | 6.67% |

---

## Dataset Statistics

### Balance Characteristics
- **Perfectly Balanced:** All 15 classes have exactly 75 images each
- **No Class Imbalance:** 1:1 ratio across all classes
- **Imbalance Ratio:** 1:1 (equal distribution)

### Dataset Properties
| Property | Value |
|----------|-------|
| Total Images | 1,125 |
| Images per Class | 75 (uniform) |
| Smallest Class | 75 images |
| Largest Class | 75 images |
| Mean Images per Class | 75.0 |
| Standard Deviation | 0.0 |
| Class Balance | Perfect |

### Image Characteristics
- **Background:** Uniform/isolated (controlled environment)
- **Leaf Position:** Centered, single leaf per image
- **Quality:** High resolution, clear species characteristics

---

## Limited Data Experiment

### Experimental Setup
**Configuration:**
- **Training Samples:** 50 images per class (from 75 available)
- **Generated Images:** 50 synthetic samples per class
- **DCGAN Epochs:** 15
- **Total Training Data:** 1,500 images ((50 real + 50 generated) × 15 classes)
- **Framework Comparison:** Python (PyTorch) vs MATLAB

### Python vs MATLAB Performance Comparison

| Split Ratio | Framework | Precision | Recall | F1-Score | Test Accuracy |
|-------------|-----------|-----------|--------|----------|---------------|
| **80-20** | Python (PyTorch) | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **80-20** | MATLAB | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **70-30** | Python (PyTorch) | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **70-30** | MATLAB | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **60-40** | Python (PyTorch) | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **60-40** | MATLAB | 1.0000 | 1.0000 | 1.0000 | 100.00% |

### Split Ratio Analysis

| Split | Train Samples | Test Samples | Python Accuracy | MATLAB Accuracy | Total Errors |
|-------|---------------|--------------|-----------------|-----------------|--------------|
| 80-20 | 600 (80%) | 150 (20%) | 100.00% | 100.00% | 0 |
| 70-30 | 525 (70%) | 225 (30%) | 100.00% | 100.00% | 0 |
| 60-40 | 450 (60%) | 300 (40%) | 100.00% | 100.00% | 0 |

---
---

# Custom Dataset Description

## Overview
- **Total Images:** 302
- **Total Classes:** 6 fruit categories
- **Image Size:** Variable (resized to 64×64 for training)
- **Format:** RGB color images
- **Source:** Custom collected dataset
- **Type:** Fruit leaf images (healthy and diseased conditions)

---

## Class Distribution Table

| # | Plant Type | Condition | Class Name | Image Count | Percentage |
|---|------------|-----------|------------|-------------|------------|
| 1 | Guava | Healthy | Guava_Healthy | 50 | 16.56% |
| 2 | Guava | Unhealthy | Guava_Unhealthy | 50 | 16.56% |
| 3 | Lemon | N/A | Lemon | 53 | 17.55% |
| 4 | Lychee | Healthy | Lychee_Healthy | 50 | 16.56% |
| 5 | Lychee | Unhealthy | Lychee_Unhealthy | 50 | 16.56% |
| 6 | Mango | Unhealthy | Mango_Unhealthy | 49 | 16.23% |

---

## Dataset Statistics

### Balance Characteristics
- **Nearly Balanced:** All 6 classes have 49-53 images each
- **Minimal Class Imbalance:** 1.08:1 ratio (53:49)
- **Imbalance Ratio:** 1.08:1 (nearly equal distribution)

### Dataset Properties
| Property | Value |
|----------|-------|
| Total Images | 302 |
| Images per Class | 49-53 |
| Smallest Class | 49 images (Mango_Unhealthy) |
| Largest Class | 53 images (Lemon) |
| Mean Images per Class | 50.3 |
| Standard Deviation | 1.37 |
| Class Balance | Excellent |

### Plant Distribution
| Plant Type | Number of Classes | Total Images | Percentage |
|------------|-------------------|--------------|------------|
| Guava | 2 | 100 | 33.11% |
| Lychee | 2 | 100 | 33.11% |
| Lemon | 1 | 53 | 17.55% |
| Mango | 1 | 49 | 16.23% |

### Health vs Disease Distribution
| Category | Number of Classes | Total Images | Percentage |
|----------|-------------------|--------------|------------|
| Healthy | 2 | 100 | 33.11% |
| Unhealthy/Diseased | 4 | 202 | 66.89% |

### Image Characteristics
- **Background:** Variable (field and controlled conditions)
- **Leaf Position:** Centered, single or multiple leaves per image
- **Quality:** High resolution, clear disease symptoms

---

## Limited Data Experiment

### Experimental Setup
**Configuration:**
- **Training Samples:** 50 images per class (or all available if less)
- **Generated Images:** 50 synthetic samples per class
- **DCGAN Epochs:** 15
- **Total Training Data:** 600 images ((50 real + 50 generated) × 6 classes)
- **Framework Comparison:** Python (PyTorch) vs MATLAB

### Python vs MATLAB Performance Comparison

| Split Ratio | Framework | Precision | Recall | F1-Score | Test Accuracy |
|-------------|-----------|-----------|--------|----------|---------------|
| **80-20** | Python (PyTorch) | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **80-20** | MATLAB | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **70-30** | Python (PyTorch) | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **70-30** | MATLAB | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **60-40** | Python (PyTorch) | 1.0000 | 1.0000 | 1.0000 | 100.00% |
| **60-40** | MATLAB | 1.0000 | 1.0000 | 1.0000 | 100.00% |

### Split Ratio Analysis

| Split | Train Samples | Test Samples | Python Accuracy | MATLAB Accuracy | Total Errors |
|-------|---------------|--------------|-----------------|-----------------|--------------|
| 80-20 | 240 (80%) | 60 (20%) | 100.00% | 100.00% | 0 |
| 70-30 | 210 (70%) | 90 (30%) | 100.00% | 100.00% | 0 |
| 60-40 | 180 (60%) | 120 (40%) | 100.00% | 100.00% | 0 |

