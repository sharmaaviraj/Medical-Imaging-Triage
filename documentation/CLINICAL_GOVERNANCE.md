# Clinical Governance, Methodological Scope & Responsible Use

This document establishes the clinical boundaries, established dataset characteristics, and governance principles for the Medical Imaging Triage workstation.

---

## 1. Distinguishing Established Dataset Scope from General Clinical AI

To maintain scientific integrity, the project strictly distinguishes empirical properties of the training data from broader clinical AI considerations:

### A. Established ISBI 2016 Dataset Characteristics
1. **Prevalence & Class Imbalance**:
   - The ISBI 2016 Part 1 benchmark exhibits a fixed ~4:1 ratio of benign to malignant samples (80.7% benign vs. 19.3% malignant).
   - In unadjusted models, class imbalance can skew raw posterior probabilities toward the majority class (benign), making a naive 0.5 decision cutoff suboptimal for sensitive triage.
2. **Imaging Modality**:
   - The dataset consists exclusively of dermoscopic images (specialized magnified optical imaging with immersion or polarized light).
   - The models cannot be assumed to generalize to clinical photography (e.g. unmagnified smartphone photographs) without retraining and domain adaptation.
3. **Annotation Standard**:
   - Ground truth masks represent expert consensus segmentations provided by the ISIC challenge organizers. Diffuse lesion borders and regression regions introduce inherent inter-annotator variability.
4. **Natural Artifacts**:
   - Dermoscopic fields often feature non-pathological elements (skin markings, hair, immersion gel air bubbles, and peripheral lens vignetting).

### B. General Machine Learning & Medical Software Considerations
1. **Regulatory Non-Device Classification**:
   - This workstation is developed strictly as an academic research and technical portfolio demonstration.
   - It is **not** cleared, certified, or registered as Software as a Medical Device (SaMD) by the FDA, EMA, or other regulatory authorities.
   - It is prohibited from being deployed for clinical diagnosis, patient triage, or direct therapy selection.
2. **Assistive Decision-Support Framing**:
   - Computer-aided lesion analysis systems are intended to operate in an assistive, human-in-the-loop capacity to highlight regions of interest and compute standardized spatial measurements.
   - Final clinical assessments require comprehensive in-person evaluation by a licensed dermatologist incorporating dermoscopy (ABCD rules), clinical history, and histopathological confirmation.
3. **Operational Threshold Sensitivity**:
   - In oncology screening scenarios, decision thresholds are typically tuned to optimize sensitivity (minimizing false negatives) according to clinical risk tolerance, rather than relying on an arbitrary default 0.5 mathematical boundary.
