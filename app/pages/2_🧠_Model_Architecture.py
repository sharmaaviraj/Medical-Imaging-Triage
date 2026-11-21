"""
Page 2: Neural Model Architectures & Specifications
Documents layer structures, parameter counts, and tensor dimensions.
Styled with the Blue Clinical theme (#102A43, #176B9A, #EAF3FA).
"""

import streamlit as st
import pandas as pd

# Import shared workstation components
from components.styles import apply_custom_css
from components.sidebar import render_sidebar

st.set_page_config(
    page_title="Model Architecture | Medical Imaging Triage",
    layout="wide"
)

apply_custom_css()
render_sidebar()

st.markdown("""
<div class="workstation-header">
    <h1>Model Architectures & Technical Specifications</h1>
    <p>Architectural inspection of the triage classifier and lesion segmentation network.</p>
</div>
""", unsafe_allow_html=True)

# Comparison Table
specs_data = [
    {
        "Property": "Task",
        "ResNet-18 (Classification)": "Binary Lesion Triage (Benign vs Malignant)",
        "U-Net (Segmentation)": "Pixel-Level Lesion Boundary Delineation"
    },
    {
        "Property": "Total Parameters",
        "ResNet-18 (Classification)": "11,177,025 parameters",
        "U-Net (Segmentation)": "7,765,985 parameters"
    },
    {
        "Property": "Input Dimensions",
        "ResNet-18 (Classification)": "224 × 224 × 3 (RGB)",
        "U-Net (Segmentation)": "256 × 256 × 3 (RGB)"
    },
    {
        "Property": "Normalization",
        "ResNet-18 (Classification)": "ImageNet Z-score (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225])",
        "U-Net (Segmentation)": "Pixel values scaled to [0, 1]"
    },
    {
        "Property": "Output Representation",
        "ResNet-18 (Classification)": "1 scalar logit -> Sigmoid probability",
        "U-Net (Segmentation)": "1 × 256 × 256 logit map -> Sigmoid binary mask"
    },
    {
        "Property": "Checkpoint Size",
        "ResNet-18 (Classification)": "~44.8 MB (`best_resnet18.pth`)",
        "U-Net (Segmentation)": "~31.1 MB (`best_unet.pth`)"
    }
]

st.dataframe(pd.DataFrame(specs_data), use_container_width=True, hide_index=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="clinical-card">
        <h3>ResNet-18 Binary Classifier</h3>
        <ul>
            <li><strong>Backbone:</strong> Standard 18-layer Deep Residual Network (He et al., 2016).</li>
            <li><strong>Residual Blocks:</strong> 4 residual layer stages with identity skip connections.</li>
            <li><strong>Classification Head:</strong> Final linear layer modified to <code>nn.Linear(512, 1)</code>.</li>
            <li><strong>Target Explainability Layer:</strong> <code>layer4[-1]</code> (final convolutional BasicBlock before pooling).</li>
        </ul>
        <p><strong>Inference Function:</strong></p>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"P(\text{malignant}) = \sigma(z) = \frac{1}{1 + e^{-z}}")
    st.caption("Outputs a scalar logit passed through a standard logistic sigmoid activation.")

with col2:
    st.markdown("""
    <div class="clinical-card">
        <h3>Custom 4-Stage U-Net</h3>
        <ul>
            <li><strong>Contracting Path (Encoder):</strong> 4 stages of <code>UNetDoubleConv</code> (Conv3x3 + BatchNorm + ReLU) with <code>MaxPool2d(2)</code>.
            Channels: $3 \\to 32 \\to 64 \\to 128 \\to 256$.</li>
            <li><strong>Bottleneck:</strong> <code>UNetDoubleConv</code> ($256 \\to 512$ channels).</li>
            <li><strong>Expanding Path (Decoder):</strong> 4 stages of <code>ConvTranspose2d</code> (stride 2) + skip feature concatenation + <code>UNetDoubleConv</code>.
            Channels: $512 \\to 256 \\to 128 \\to 64 \\to 32$.</li>
            <li><strong>Head:</strong> $1 \\times 1$ convolution mapping $32 \\to 1$ channel raw spatial logits.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>Reproducibility Note:</strong> Parameter counts and tensor dimensions have been empirically
    verified from the active checkpoints (<code>best_resnet18.pth</code> and <code>best_unet.pth</code>)
    loaded under evaluation mode.
</div>
""", unsafe_allow_html=True)
