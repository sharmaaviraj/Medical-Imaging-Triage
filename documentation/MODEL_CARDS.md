# Model Cards & Verification Summary

This document details the architectural specifications, verification status, and visual explainability methodology for the neural network checkpoints in the repository.

---

## Model 1: ResNet-18 Binary Lesion Classifier

### 1. Verified Architecture & Checkpoint Specifications
* **Architecture**: 18-layer Deep Residual Network (torchvision backbone with custom classification head).
* **Head Modification**: Standard ImageNet 1,000-class linear layer replaced with `nn.Linear(in_features=512, out_features=1)`.
* **Parameter Count**: **11,177,025 parameters** (empirically calculated from loaded state dict).
* **Checkpoint File**: `models/classification/best_resnet18.pth` (~44.8 MB).
* **State Dict Structure**: 122 tensor keys (convolution weights, biases, batch normalization running means and variances).
* **Loading Security**: Verified loading using `torch.load(..., weights_only=True)`.

### 2. Verified Input & Preprocessing Pipeline
* **Input Resolution**: $224 \times 224$ pixels, 3 color channels (RGB).
* **Normalization**: Channel-wise z-score normalization using ImageNet statistics:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard Deviation: `[0.229, 0.224, 0.225]`
* **Output Activation**: Single logit transformed via standard logistic sigmoid $\sigma(z) \in [0, 1]$.
* **Operational Threshold**: Configurable research slider defaulting strictly to $0.50$ (baseline).

### 3. Training & Evaluation Details (Verification Status)
* **Training Dataset**: ISBI 2016 Challenge Part 1 (`train.csv`, 630 dermoscopy images).
* **Target Classes**: Binary (`benign` = 0, `malignant` = 1).
* **Training Protocol**: 10 epochs using Adam optimizer ($\eta = 1 \times 10^{-4}$, weight decay $\lambda = 1 \times 10^{-4}$, batch size $= 16$) with `BCEWithLogitsLoss` and positive class weighting. Best checkpoint selected at minimum validation loss.
* **Reported Test Performance Metrics** ($N_{\text{test}} = 135$ samples from the internal held-out partition of the 900-image training cohort; **not** the official 379-image ISIC 2016 external challenge test set, default threshold $\tau = 0.50$):
  - **Accuracy**: **68.15%** (92 / 135)
  - **Malignant Sensitivity (Recall)**: **73.08%** (19 / 26)
  - **Malignant Precision**: **35.00%** (19 / 54)
  - **Malignant F1-Score**: $\approx \mathbf{0.473}$
  - **Specificity (Benign Recall)**: **66.97%** (73 / 109)
  - **AUROC**: **0.7862**
  *(See `documentation/EXPERIMENTS.md` for full metric derivations, confusion matrix analysis, and clinical trade-off evaluation).*

### 4. Visual Explainability (Grad-CAM)
* **Objective**: Visual feature attribution showing spatial regions that positively influenced the malignant-class score.
* **Target Layer**: `model.layer4[-1]` (the final residual `BasicBlock` of the fourth stage before global average pooling).
* **Feature Dimension**: 512 channels at $7 \times 7$ spatial resolution.
* **Mathematical Calculation**:
  1. Neuron importance weights computed via Global Average Pooling of gradients:
     $$\alpha_k = \frac{1}{Z} \sum_{i=1}^{7} \sum_{j=1}^{7} \frac{\partial y_{\text{malignant}}}{\partial A_{i, j}^k}$$
  2. Weighted feature combination with ReLU rectification:
     $$L_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_{k=1}^{512} \alpha_k A^k \right)$$
  3. Bilinear interpolation scales the $7 \times 7$ map to original image resolution.
  4. Robust min-max scaling normalizes intensities to $[0, 1]$ with $\epsilon = 10^{-8}$ zero-spread protection.
* **Clinical Limitations & Caveats**:
  - Grad-CAM reflects mathematical gradient magnitude within the convolutional feature maps; it **does not** constitute pathological evidence, histological validation, or biological reasoning.
  - Saliency heatmaps may highlight non-pathological imaging artifacts (e.g., hair, marker ink, dark vignetting) if those features correlate with training set biases.
  - Saliency maps should only be interpreted as an auxiliary debugging and interpretability check by a qualified clinical researcher.

---

## Model 2: Custom 4-Stage U-Net Lesion Segmenter

### 1. Verified Architecture & Checkpoint Specifications
* **Architecture**: Custom 4-stage convolutional U-Net with contracting (encoder) and expanding (decoder) paths connected via residual skip concatenation.
* **Building Block**: `UNetDoubleConv` (Two successive Conv2d $3 \times 3$ with padding=1, BatchNorm2d, and ReLU activations).
* **Encoder Stages**:
  - `enc1`: $3 \to 32$ channels
  - `enc2`: $32 \to 64$ channels
  - `enc3`: $64 \to 128$ channels
  - `enc4`: $128 \to 256$ channels
  - Downsampling: $2 \times 2$ `MaxPool2d(2)` between stages.
* **Bottleneck**: `UNetDoubleConv` ($256 \to 512$ channels).
* **Decoder Stages**:
  - Transposed Convolutions: 4 stages of `ConvTranspose2d(..., kernel_size=2, stride=2)`.
  - Skip Concatenation: Channel concatenation of decoder feature maps with corresponding encoder feature maps ($512 \to 256 \to 128 \to 64$).
  - `dec4`: $512 \to 256$ channels
  - `dec3`: $256 \to 128$ channels
  - `dec2`: $128 \to 64$ channels
  - `dec1`: $64 \to 32$ channels
* **Segmentation Head**: $1 \times 1$ Conv2d mapping $32 \to 1$ spatial logit channel.
* **Parameter Count**: **7,765,985 parameters** (empirically calculated from loaded state dict).
* **Checkpoint File**: `models/segmentation/best_unet.pth` (~31.1 MB).
* **State Dict Structure**: 136 tensor keys.
* **Loading Security**: Verified loading using `torch.load(..., weights_only=True)`.

### 2. Verified Input & Post-Processing Pipeline
* **Input Resolution**: $256 \times 256$ pixels, 3 color channels (RGB).
* **Normalization**: Pixel values scaled to range $[0, 1]$ (no ImageNet z-score normalization).
* **Output Processing**: Spatial logits $(1, 1, 256, 256)$ transformed via sigmoid and thresholded at $0.5$ to yield a binary mask.
* **Spatial Alignment**: Resized to original image dimensions via nearest-neighbor interpolation (`cv2.INTER_NEAREST`).
* **Bounding Box**: Derived from the largest external contour of the binary mask via OpenCV (`cv2.findContours`).

### 3. Training & Evaluation Details (Verification Status)
* **Training Dataset**: ISBI 2016 Challenge Part 1 (`train.csv`, 630 dermoscopy images and expert masks).
* **Training Protocol**: 10 epochs using Adam optimizer ($\eta = 1 \times 10^{-4}$, batch size $= 8$) with compound `BCEWithLogitsLoss` + Soft Dice Loss. Checkpoint saved at minimum validation loss.
* **Status of Test Segmentation Metrics**: *Exact Numerical Test Logs Not Retained* (exact numerical test set Mean Intersection-over-Union [mIoU] and Dice Similarity Coefficient [DSC] evaluation logs were not recorded in the repository checkpoint metadata; in accordance with strict academic guidelines, no synthetic metrics have been fabricated). *(See `documentation/EXPERIMENTS.md` for mathematical definitions of DSC and IoU).*
