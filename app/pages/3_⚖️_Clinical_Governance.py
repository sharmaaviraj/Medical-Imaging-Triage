"""
Page 3: Clinical Governance, Dataset Limitations & Responsible AI
Distinguishes verified dataset characteristics from broader clinical AI principles.
"""

import streamlit as st

# Import shared workstation components
from components.styles import apply_custom_css
from components.sidebar import render_sidebar

st.set_page_config(
    page_title="Clinical Governance | Medical Imaging Triage",
    layout="wide"
)

apply_custom_css()
render_sidebar()

st.markdown("""
<div class="workstation-header">
    <h1>Clinical Governance & Methodological Scope</h1>
    <p>Established dataset characteristics, boundary conditions, and responsible use principles.</p>
</div>
""", unsafe_allow_html=True)

# Section 1: Established Dataset-Specific Limitations
st.subheader("1. Established ISBI 2016 Dataset Characteristics & Limitations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Class Distribution & Prevalence**
    - The dataset displays a verified ~4:1 class imbalance (80.8% benign vs. 19.2% malignant across all splits).
    - Models trained on such distributions without cost-sensitive weighting or threshold tuning may exhibit higher specificity than sensitivity at a default 0.5 decision cutoff.

    **Image Acquisition Scope**
    - The dataset consists exclusively of dermoscopic images (specialized optical magnification with contact or non-contact polarization).
    - The models are not trained on, nor intended for, macroscopic (standard clinical smartphone/digital camera) photographs.
    """)

with col2:
    st.markdown("""
    **Annotation & Boundary Delineation**
    - Ground truth masks in the challenge were expert-annotated. Lesions with diffuse boundaries or subtle pigment regression naturally introduce inter-observer variability.
    
    **Imaging Artifacts**
    - Dermoscopic captures frequently contain natural artifacts (e.g., skin markings, hair, immersion gel bubbles, and vignetting around lens borders) that can influence unconstrained convolutional features.
    """)

st.divider()

# Section 2: General Medical AI & Governance Considerations
st.subheader("2. General Medical AI Principles & Software Scope")

st.markdown("""
**Prototype Status & Intended Use**
- This workstation is developed strictly as an academic research and technical demonstration system.
- It is **not** a certified Software as a Medical Device (SaMD) and has not undergone clearance by regulatory authorities (e.g., FDA, EMA, or national health authorities).
- The workstation must not be used as an independent clinical diagnostic instrument or to determine patient treatment pathways.

**Human-in-the-Loop Decision Support**
- Computer-aided detection (CAD) systems in dermatopathology are designed to support qualified clinical specialists by highlighting regions of interest (ROI) and providing consistent measurements.
- Algorithmic predictions must always be interpreted in conjunction with patient clinical history, physical examination, dermoscopy (ABCD criteria), and histopathological biopsy where indicated.

**Operational Threshold Considerations**
- The default 0.5 decision threshold is a standard mathematical midpoint for binary classification, not a clinically validated triage threshold.
- In clinical screening environments, triage thresholds are typically calibrated based on clinical utility functions—often prioritizing high sensitivity to avoid missed malignancies.
""")

st.markdown("""
<div class="disclaimer-box">
    <strong>Academic Disclaimer:</strong> This application and its associated models are provided solely
    for educational research and technical evaluation. All analysis results should be treated as preliminary
    experimental outputs.
</div>
""", unsafe_allow_html=True)
