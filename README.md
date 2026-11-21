# Medical Image Triage & Lesion Analysis Workstation

A deep learning-based medical imaging project that combines image classification, lesion segmentation, visual explainability, and analysis reporting in an interactive Streamlit application.

> **Disclaimer:** This is an academic research and demonstration project. It is not a clinically validated diagnostic system and must not be used for medical decision-making.

## Overview

Medical image analysis can benefit from automated systems that help identify suspicious regions and provide visual insights into model predictions.

This project brings together two deep learning models:

- **ResNet18:** Classifies dermoscopic images as benign or malignant.
- **U-Net:** Generates segmentation masks to highlight suspected lesion regions.

The application also integrates Grad-CAM visualizations, bounding-box localization, analysis history, and PDF report generation.

## Key Features

- **Image Classification:** Benign and malignant classification using a fine-tuned ResNet18.
- **Lesion Segmentation:** U-Net-based predicted lesion masks.
- **Explainable AI:** Grad-CAM visualizations to inspect image regions influencing classification.
- **Lesion Localization:** Bounding-box generation from predicted segmentation masks.
- **Adjustable Threshold:** Allows users to modify the classification decision threshold.
- **Analysis History:** Save and revisit previous analyses.
- **PDF Reports:** Generate downloadable analysis reports.
- **Dataset Analytics:** Explore dataset distribution and characteristics.
- **Optimization Dashboard:** Compare optimization algorithms and training configurations through experiments.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Classification | ResNet18 |
| Segmentation | U-Net |
| Computer Vision | OpenCV |
| Interface | Streamlit |
| Visualization | Plotly, Matplotlib |
| Data Processing | NumPy, Pandas |
| Reporting | PDF generation |

## System Architecture

The application follows this workflow:

1. Upload a dermoscopic image.
2. Preprocess the image for model input.
3. Generate a classification score using ResNet18.
4. Generate a segmentation mask using U-Net.
5. Produce Grad-CAM visualizations for classification explainability.
6. Derive a bounding box from the predicted segmentation mask.
7. Display results and optionally save the analysis.

## Deep Learning Models

### 1. ResNet18 — Classification

- Pretrained on ImageNet.
- Modified final fully connected layer for binary classification.
- Input resolution: 224 × 224.
- Loss function: BCEWithLogitsLoss with positive-class weighting.
- Optimizer: Adam.
- Learning rate: 0.0001.
- Weight decay: 0.0001.
- Training epochs: 10.
- Best checkpoint selected using validation loss.

### 2. U-Net — Segmentation

- Encoder-decoder architecture with skip connections.
- Input resolution: 256 × 256.
- Channel configuration: 32, 64, 128, 256, 512.
- Loss: Combination of Binary Cross-Entropy and Dice loss.
- Optimizer: Adam.
- Learning rate: 0.0001.
- Batch size: 8.
- Training epochs: 10.

## Dataset

The project uses the **ISIC 2016 training cohort** for model development.

| Dataset Detail | Value |
|---|---:|
| Dermoscopic images | 900 |
| Segmentation masks | 900 |
| Benign images | 727 |
| Malignant images | 173 |
| Training split | 630 |
| Validation split | 135 |
| Internal test split | 135 |

The data was divided using a stratified image-level split with `random_state=42`.

**Important:** The 135-image internal test split is derived from the training cohort. It is not the official ISIC 2016 challenge test set. Verified patient identifiers were unavailable, so patient-level splitting could not be performed.

The dataset is not included in this GitHub repository. Users must obtain it separately from the official ISIC challenge resources.

## Model Evaluation

### Classification Results

Evaluation on the internal held-out test split:

| Metric | Result |
|---|---:|
| Accuracy | 68.15% |
| AUROC | 0.7862 |
| Sensitivity (Malignant Recall) | 73.08% |
| Specificity | 66.97% |
| Malignant Precision | 35% |

The malignant class has lower precision, which indicates that false-positive predictions remain an important limitation.

The model's sigmoid output is a classification score and should not be interpreted as a calibrated clinical probability.

### Segmentation

The U-Net model generates predicted lesion masks that can be visually compared with reference masks.

Quantitative segmentation metrics are not reported here because verified final test-set Dice and IoU values are not available.

## Optimization Experiments

The project also includes a separate experimental module to study neural network optimization.

Experiments use a synthetic `make_moons` dataset and a simple multilayer perceptron.

Topics explored include:

- Batch Gradient Descent
- Momentum
- Nesterov Accelerated Gradient
- Mini-batch Stochastic Gradient Descent
- Learning-rate comparison
- Weight initialization strategies
- L2 regularization

These experiments are separate from the medical image model training.

## Repository Structure

```text
Medical-Imaging-Triage/
│
├── app/
│   ├── app.py
│   ├── components/
│   └── pages/
│
├── documentation/
│   ├── ARCHITECTURE.md
│   ├── CLINICAL_GOVERNANCE.md
│   ├── DATASET_SPECIFICATION.md
│   ├── EXPERIMENTS.md
│   └── MODEL_CARDS.md
│
├── experiments/
│   └── optimizers/
│
├── models/
│   ├── classification/
│   │   └── best_resnet18.pth
│   └── segmentation/
│       └── best_unet.pth
│
├── results/
│   └── experiments/
│       └── benchmarks.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### Prerequisites

- Python installed on your system.
- Git.
- VS Code or another Python IDE.

### 1. Clone the repository

```bash
git clone https://github.com/sharmaaviraj/Medical-Imaging-Triage.git
cd Medical-Imaging-Triage
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app/app.py
```

The application should open in your default browser.

## Future Scope

- Improve classification performance and reduce false-positive predictions.
- Evaluate segmentation using verified quantitative metrics.
- Explore transformer-based vision architectures.
- Perform patient-level evaluation when verified patient identifiers become available.
- Improve model calibration and robustness.
- Conduct external validation on independent datasets.

## Limitations

- The project uses a relatively small dataset.
- The classification model shows signs of overfitting.
- Malignant-class precision remains limited.
- Patient-level separation was not possible with the available metadata.
- No clinical validation or prospective evaluation has been conducted.
- Grad-CAM provides visual explanations, not proof of model correctness.

## Author

**Aviraj Sharma**  
B.Tech Computer Science Engineering — AI/ML Specialization  
UPES, Dehradun

GitHub: [@sharmaaviraj](https://github.com/sharmaaviraj)

---

*Developed as an academic deep learning project exploring medical image classification, segmentation, and explainable AI.*