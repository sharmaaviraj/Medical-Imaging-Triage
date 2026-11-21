# Dataset Specification & Partition Metadata

**Benchmark**: ISBI 2016 Challenge (Part 1: Lesion Classification & Segmentation)  
**Official Sources**: [ISIC Challenge Archive](https://challenge.isic-archive.com/data/) | [2016 Task 1 Benchmark](https://challenge.isic-archive.com/landing/2016/37/)  
**Licensing**: **Creative Commons Zero 1.0 Universal (CC-0 / Public Domain Dedication)**  
*(The ISIC 2016 Task 1 training images and segmentation masks are published under CC-0, distinct from later challenge iterations which adopted CC BY-NC 4.0).*  
**Total Cohort Size**: 900 dermoscopic records  
**Splits Location**: `data/splits/`  

---

## 1. Verified Cohort Breakdown

Empirical verification of `train.csv`, `validation.csv`, and `test.csv` establishes a 70% / 15% / 15% partition scheme:

| Partition | Total Images | Benign Count | Benign % | Malignant Count | Malignant % | Imbalance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** (`train.csv`) | 630 | 509 | 80.79% | 121 | 19.21% | 4.21 : 1 |
| **Validation** (`validation.csv`) | 135 | 109 | 80.74% | 26 | 19.26% | 4.19 : 1 |
| **Internal Test Split** (`test.csv`) | 135 | 109 | 80.74% | 26 | 19.26% | 4.19 : 1 |
| **Total Cohort** | **900** | **727** | **80.78%** | **173** | **19.22%** | **4.20 : 1** |

> [!IMPORTANT]
> **Internal Held-Out Test Partition vs. Official Challenge Test Set**:  
> The 135-image test split (`test.csv`) is an **internal held-out partition** derived from the 900-image ISIC 2016 training cohort (15% held out). It is **not** the official ISIC 2016 external challenge test set (which consists of 379 independent images). All reported test performance metrics (e.g., accuracy 68.15%, AUROC 0.7862) refer strictly to this internal held-out split.

### Stratification Characteristics
All three partitions are precisely stratified at an ~80.8% benign to ~19.2% malignant ratio (random seed 42). This class imbalance reflects clinical prevalence patterns in dermatology screening where malignant lesions (e.g. melanoma) form the minority class relative to benign nevi.

### Partitioning Boundary Conditions
* **Image-Level Partitioning**: Splitting was performed strictly at the image level because verified patient identifiers are not available in the ISIC 2016 metadata. No patient-level separation is claimed.
* **Near-Duplicate Detection**: Cryptographic perceptual hashing was not performed in code.

---

## 2. CSV Metadata Schema

Each split CSV contains the following four columns:

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `image_id` | String | Unique ISIC sample identifier | `ISIC_0011295` |
| `image_path` | String | Recorded source dermoscopy file path | `/content/medical_imaging/ISBI2016_ISIC_Part1_Training_Data/ISIC_0011295.jpg` |
| `mask_path` | String | Recorded ground-truth segmentation mask path | `/content/medical_imaging/ISBI2016_ISIC_Part1_Training_GroundTruth/ISIC_0011295_Segmentation.png` |
| `diagnosis` | String | Binary diagnostic label (`benign` or `malignant`) | `malignant` |

---

## 3. Handling Google Colab Paths Locally

The recorded paths in the CSV files point to the Google Colab training environment (`/content/...`). To ensure complete reproducibility without modifying the original dataset CSV files:

1. **Resolution Utility**: The function `resolve_image_path(csv_path)` in `app/components/config.py` translates recorded Colab paths by matching filename basenames (`ISIC_xxxxxxx.jpg`) against local folders:
   - `data/raw/`
   - `data/raw/ISBI2016_ISIC_Part1_Training_Data/`
   - `data/samples/`
   - `data/processed/`
2. **Local Image Placement**: Users wishing to run batch evaluations on local images can place ISIC images directly in `data/raw/` or `data/samples/`. The utility will resolve them automatically.
3. **Decoupled Analytics**: The Dataset Analytics page (`1_📊_Dataset_Analytics.py`) reads the metadata directly from the CSV files, functioning with 100% fidelity even when raw image files are unmounted.
