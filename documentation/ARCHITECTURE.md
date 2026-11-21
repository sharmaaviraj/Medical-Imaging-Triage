# System Architecture & Technical Specifications

**Project**: Medical Imaging Triage and Lesion Analysis Workstation  
**Domain**: Computer-Aided Dermoscopic Lesion Analysis  
**Framework**: PyTorch & Streamlit  

---

## 1. Architectural Overview

The workstation is designed as an interactive clinical decision-support prototype that performs dual-task inference on dermoscopic images:
1. **Binary Lesion Classification**: Assesses whether a suspicious lesion exhibits features characteristic of malignancy vs. benign pathology using an adapted **ResNet-18** architecture.
2. **Lesion Boundary Segmentation**: Delineates the exact spatial boundaries of the lesion using a custom 4-stage **U-Net** architecture.

```
                              ┌────────────────────────────────────────┐
                              │       Dermoscopic Input Image          │
                              └──────────────────┬─────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │                                                 │
                        ▼                                                 ▼
             ┌─────────────────────┐                           ┌─────────────────────┐
             │ Classification Head │                           │ Segmentation Engine │
             │     (ResNet-18)     │                           │       (U-Net)       │
             └──────────┬──────────┘                           └──────────┬──────────┘
                        │                                                 │
                        ▼                                                 ▼
             ┌─────────────────────┐                           ┌─────────────────────┐
             │ Malignancy Score    │                           │ Binary Lesion Mask  │
             │ P(malignant) ∈[0,1] │                           │ (256x256 -> Origin) │
             └─────────────────────┘                           └──────────┬──────────┘
                                                                          │
                                                               ┌──────────┴──────────┐
                                                               ▼                     ▼
                                                    ┌────────────────────┐ ┌───────────────────┐
                                                    │ Alpha Mask Overlay │ │ Lesion Bounding   │
                                                    │    (Green Alpha)   │ │ Box (OpenCV Cont) │
                                                    └────────────────────┘ └───────────────────┘
```

---

## 2. Directory Layout & Module Responsibilities

```
Medical-Imaging-Triage/
├── app/
│   ├── app.py                         # Entry point & Live Clinical Triage Workstation
│   ├── model_architectures.py         # Pure PyTorch neural definitions & factory loaders
│   ├── components/                    # Modular shared libraries
│   │   ├── __init__.py                # Package export interface
│   │   ├── config.py                  # Path resolution & Google Colab path translation
│   │   ├── device.py                  # Dynamic CUDA/CPU detection & telemetry
│   │   ├── gradcam.py                 # Native PyTorch Grad-CAM explainability engine
│   │   ├── history_manager.py         # Permanent analysis archival & metadata serialization
│   │   ├── pdf_generator.py           # Publication-grade 2-page A4 clinical report compiler
│   │   ├── styles.py                  # Clinical Blue CSS theme & Plotly styling engine
│   │   └── sidebar.py                 # Shared telemetry & navigation sidebar
│   └── pages/                         # Multi-page Streamlit views
│       ├── 1_📊_Dataset_Analytics.py       # ISBI 2016 split metrics & class balance
│       ├── 2_🧠_Model_Architecture.py      # Layer configurations & parameter counts
│       ├── 3_⚖️_Clinical_Governance.py     # Scope, limitations & ethical guidelines
│       ├── 4_🧪_Optimization_Experiments.py# First-order optimization laboratory
│       └── 5_📁_Saved_Analysis_History.py  # Analysis history archive & PDF report retrieval
├── data/
│   ├── splits/                        # Benchmark partition metadata (ISBI 2016 Part 1)
│   │   ├── train.csv                  # 630 entries (509 benign, 121 malignant)
│   │   ├── validation.csv             # 135 entries (109 benign, 26 malignant)
│   │   └── test.csv                   # 135 entries (internal held-out split, not the 379 official test set)
│   ├── raw/                           # Reserved for local raw datasets
│   └── processed/                     # Reserved for preprocessed arrays
├── experiments/
│   └── optimizers/                    # Foundational deep learning experiment engine
│       ├── models.py                  # SimpleMLP (641 params) & synthetic data generators
│       └── experiment_runner.py       # Deterministic benchmark runner & JSON serializer
├── models/
│   ├── classification/
│   │   └── best_resnet18.pth          # ResNet-18 state_dict (11,177,025 params)
│   └── segmentation/
│       └── best_unet.pth              # U-Net state_dict (7,765,985 params)
├── results/
│   ├── analysis_history/              # Permanent analysis session archive
│   │   └── ANL_<TIMESTAMP>_<ID>/      # Saved images, metadata.json & report.pdf
│   └── experiments/
│       └── benchmarks.json            # Serialized benchmark suite records
├── documentation/                     # Technical documentation & project cards
│   ├── ARCHITECTURE.md                # System topology & module layout
│   ├── CLINICAL_GOVERNANCE.md         # Ethical AI & dataset constraints
│   ├── DATASET_SPECIFICATION.md       # ISBI 2016 partition metadata
│   ├── MODEL_CARDS.md                 # Checkpoint parameters & verification status
│   └── EXPERIMENTS.md                 # Comprehensive academic evaluation & viva guide
└── requirements.txt                   # Pinned Python package environment
```

---

## 3. Data Flow & Processing Pipelines

### Classification Pipeline (ResNet-18)
1. **Input**: Original dermoscopic image (RGB).
2. **Preprocessing**:
   - Resized to $224 \times 224$ pixels.
   - Converted to PyTorch tensor with values in $[0, 1]$.
   - Normalized using standard ImageNet channel statistics:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
3. **Inference**:
   - Forward pass produces a scalar logit $z$.
   - Sigmoid activation maps logit to a continuous confidence score:
     $$P(\text{malignant}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
4. **Decision**:
   - Default mathematical binarization at threshold $= 0.5$.

### Segmentation Pipeline (U-Net)
1. **Input**: Original dermoscopic image (RGB).
2. **Preprocessing**:
   - Resized to $256 \times 256$ pixels.
   - Converted to PyTorch tensor with values in $[0, 1]$ (unnormalized).
3. **Inference**:
   - Forward pass yields single-channel spatial logits $(1, 1, 256, 256)$.
   - Sigmoid activation with threshold $= 0.5$ generates binary mask $(256, 256)$.
4. **Spatial Post-Processing**:
   - Nearest-neighbor interpolation scales the binary mask back to the original image dimensions $(W, H)$ without introducing intermediate grayscale blending artifacts.
   - Alpha-blended green overlay: $0.5 \times I + 0.5 \times [0, 255, 0]$.
   - OpenCV external contour detection (`cv2.RETR_EXTERNAL`, `cv2.CHAIN_APPROX_SIMPLE`) extracts the largest contour and bounds it with a bounding box rectangle $(x, y, w, h)$.
