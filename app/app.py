"""
Medical Imaging Triage and Lesion Analysis Workstation
Primary Workstation Interface with Result Persistence, Adjustable Triage Threshold,
and Native PyTorch Grad-CAM Explainability.
"""

import time
import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import streamlit as st
import torch
from torchvision import transforms

# Shared workstation configuration, device telemetry, and UI components
import components.config as cfg
from components.device import get_device
from components.styles import apply_custom_css
from components.sidebar import render_sidebar
from components.gradcam import GradCAMExplainer
from components.history_manager import save_analysis_record, HISTORY_DIR
from components.pdf_generator import generate_pdf_report

from model_architectures import (
    load_classification_model,
    load_segmentation_model
)


# --------------------------------------------------
# PAGE CONFIGURATION & STYLING
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Imaging Triage Workstation",
    layout="wide"
)

apply_custom_css()
render_sidebar()

st.markdown("""
<div class="workstation-header">
    <h1>Medical Imaging Triage and Lesion Analysis</h1>
    <p>Academic deep learning workstation for skin lesion classification, U-Net segmentation, and Grad-CAM explainability.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PROJECT PATHS & DEVICE INITIALIZATION
# --------------------------------------------------

ROOT = cfg.ROOT_DIR
CLASSIFIER_PATH = cfg.CLASSIFIER_PATH
SEGMENTATION_PATH = cfg.SEGMENTATION_PATH
DEVICE = get_device()


# --------------------------------------------------
# MODEL LOADING WITH RESOURCE CACHING
# --------------------------------------------------

@st.cache_resource
def load_models():
    classifier = load_classification_model(CLASSIFIER_PATH, DEVICE)
    segmenter = load_segmentation_model(SEGMENTATION_PATH, DEVICE)
    return classifier, segmenter


# --------------------------------------------------
# PREPROCESSING TRANSFORMS (PRESERVED STRICTLY)
# --------------------------------------------------

classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

segmentation_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# --------------------------------------------------
# CONTOUR BOUNDING BOX EXTRACTION
# --------------------------------------------------

def get_bounding_box(mask):
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, w, h


# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------

if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None

if "uploaded_file_id" not in st.session_state:
    st.session_state["uploaded_file_id"] = None

if "cached_image" not in st.session_state:
    st.session_state["cached_image"] = None

if "saved_analysis_id" not in st.session_state:
    st.session_state["saved_analysis_id"] = None

if "saved_pdf_bytes" not in st.session_state:
    st.session_state["saved_pdf_bytes"] = None

if "independent_pdf_bytes" not in st.session_state:
    st.session_state["independent_pdf_bytes"] = None


# --------------------------------------------------
# IMAGE UPLOAD SECTION
# --------------------------------------------------

st.subheader("Dermoscopic Image Acquisition")

uploaded_file = st.file_uploader(
    "Select a dermoscopic image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

# Detect file changes to manage state invalidation
if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state["uploaded_file_id"] != current_file_id:
        st.session_state["uploaded_file_id"] = current_file_id
        st.session_state["cached_image"] = Image.open(uploaded_file).convert("RGB")
        st.session_state["analysis_results"] = None
        st.session_state["saved_analysis_id"] = None
        st.session_state["saved_pdf_bytes"] = None
        st.session_state["independent_pdf_bytes"] = None
else:
    st.session_state["uploaded_file_id"] = None
    st.session_state["cached_image"] = None
    st.session_state["analysis_results"] = None
    st.session_state["saved_analysis_id"] = None
    st.session_state["saved_pdf_bytes"] = None
    st.session_state["independent_pdf_bytes"] = None


# --------------------------------------------------
# EXECUTION & RESULT RENDERING
# --------------------------------------------------

if st.session_state["cached_image"] is not None:
    original_image = st.session_state["cached_image"]

    col_preview, col_action = st.columns([1, 2])
    with col_preview:
        st.image(original_image, caption="Acquired Input Image", use_container_width=True)

    with col_action:
        st.markdown("**Analysis Workflow**")
        st.caption(
            "Execute the dual-engine triage pipeline: ResNet-18 classification, "
            "U-Net boundary segmentation, and Grad-CAM feature attribution."
        )

        analyze_button = st.button("Run Full Analysis", type="primary")

        if analyze_button:
            if not CLASSIFIER_PATH.exists():
                st.error(f"Classification checkpoint missing: {CLASSIFIER_PATH}")
            elif not SEGMENTATION_PATH.exists():
                st.error(f"Segmentation checkpoint missing: {SEGMENTATION_PATH}")
            else:
                try:
                    with st.spinner("Executing neural inference and computing explainability..."):
                        classifier, segmenter = load_models()

                        # 1. Classification Forward Pass
                        clf_input = classification_transform(original_image).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            classification_logit = classifier(clf_input)
                            malignant_score = float(torch.sigmoid(classification_logit).item())

                        # 2. Native Grad-CAM Feature Attribution
                        explainer = GradCAMExplainer(classifier)
                        heatmap_2d, heatmap_rgb, gradcam_overlay = explainer.generate(
                            input_tensor=clf_input,
                            original_image=original_image,
                            alpha=0.45
                        )

                        # 3. U-Net Segmentation Forward Pass
                        seg_input = segmentation_transform(original_image).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            seg_logits = segmenter(seg_input)
                            predicted_mask_256 = (
                                torch.sigmoid(seg_logits)[0, 0] > 0.5
                            ).cpu().numpy().astype(np.uint8)

                        # Scale mask to original dimensions
                        orig_array = np.array(original_image)
                        h_orig, w_orig = orig_array.shape[:2]

                        predicted_mask = cv2.resize(
                            predicted_mask_256,
                            (w_orig, h_orig),
                            interpolation=cv2.INTER_NEAREST
                        )

                        # Generate green alpha overlay (identical to baseline)
                        overlay = orig_array.copy()
                        overlay[predicted_mask == 1] = (
                            0.5 * overlay[predicted_mask == 1]
                            + 0.5 * np.array([0, 255, 0])
                        ).astype(np.uint8)

                        # Generate bounding box (identical to baseline)
                        bounding_box = get_bounding_box(predicted_mask)
                        boxed_image = orig_array.copy()
                        if bounding_box is not None:
                            bx, by, bw, bh = bounding_box
                            cv2.rectangle(
                                boxed_image,
                                (bx, by),
                                (bx + bw, by + bh),
                                (255, 0, 0),
                                3
                            )

                        # Store all outputs in session state to persist across reruns
                        st.session_state["analysis_results"] = {
                            "malignant_score": malignant_score,
                            "predicted_mask": predicted_mask,
                            "bounding_box": bounding_box,
                            "overlay": overlay,
                            "boxed_image": boxed_image,
                            "gradcam_heatmap": heatmap_2d,
                            "gradcam_heatmap_rgb": heatmap_rgb,
                            "gradcam_overlay": gradcam_overlay,
                        }

                except Exception as error:
                    st.error("An error occurred during workstation analysis.")
                    st.exception(error)


    # --------------------------------------------------
    # DISPLAY PERSISTENT RESULTS
    # --------------------------------------------------

    if st.session_state["analysis_results"] is not None:
        results = st.session_state["analysis_results"]
        malignant_score = results["malignant_score"]

        st.divider()
        st.subheader("Triage Decision & Sensitivity Control")

        # Threshold Control Slider
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            threshold = st.slider(
                "Operational Decision Cutoff",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.01,
                help="Adjusting this setting recalibrates the decision label without re-running model inference."
            )

        with col_ctrl2:
            st.caption(
                "Default threshold is strictly 0.50. In screening environments, lower cutoffs prioritize sensitivity "
                "to minimize false negatives, while higher cutoffs prioritize specificity. This is an experimental "
                "research setting, not a clinically certified diagnostic cutoff."
            )

        # Dynamic classification label re-evaluation
        is_malignant = malignant_score >= threshold

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric(
                "Raw Malignancy Score",
                f"{malignant_score * 100:.2f}%"
            )
            st.caption("Sigmoid probability output from ResNet-18.")

        with col_res2:
            st.metric(
                "Active Decision Cutoff",
                f"{threshold * 100:.1f}%"
            )
            st.caption("Operational classification threshold.")

        with col_res3:
            if is_malignant:
                st.error("Triage Result: Malignant")
                st.caption(f"Score ({malignant_score*100:.1f}%) ≥ Cutoff ({threshold*100:.1f}%)")
            else:
                st.success("Triage Result: Benign")
                st.caption(f"Score ({malignant_score*100:.1f}%) < Cutoff ({threshold*100:.1f}%)")


        # --------------------------------------------------
        # GRAD-CAM EXPLAINABILITY SECTION
        # --------------------------------------------------

        st.divider()
        st.subheader("Model Attention & Feature Attribution (Grad-CAM)")
        st.caption(
            "Visual attribution showing input regions with the highest gradient activations "
            "for the malignant class from ResNet-18 layer4[-1]."
        )

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.image(original_image, caption="Original Input", use_container_width=True)
        with col_g2:
            st.image(results["gradcam_heatmap_rgb"], caption="Grad-CAM Saliency Map", use_container_width=True)
        with col_g3:
            st.image(results["gradcam_overlay"], caption="Feature Attribution Overlay", use_container_width=True)

        st.markdown("""
        <div class="disclaimer-box">
            <strong>Explainability Notice:</strong> The Grad-CAM heatmap highlights image regions that most strongly
            influenced the model's malignant-class score. It reflects convolutional feature activations, not pathological
            proof or verified diagnostic markers.
        </div>
        """, unsafe_allow_html=True)


        # --------------------------------------------------
        # U-NET SEGMENTATION SECTION (PRESERVED FUNCTIONALITY)
        # --------------------------------------------------

        st.divider()
        st.subheader("Lesion Boundary Segmentation (U-Net)")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.image(
                results["predicted_mask"] * 255,
                caption="Predicted Binary Mask",
                use_container_width=True
            )
        with col_s2:
            st.image(
                results["overlay"],
                caption="Segmentation Overlay (Green)",
                use_container_width=True
            )
        with col_s3:
            st.image(
                results["boxed_image"],
                caption="Derived Bounding Box (Red)",
                use_container_width=True
            )

        if results["bounding_box"] is not None:
            bx, by, bw, bh = results["bounding_box"]
            st.caption(
                f"**Bounding box coordinates:** x={bx}, y={by}, width={bw}, height={bh}. "
                "Extracted from the largest contour of the U-Net segmentation mask."
            )
        else:
            st.warning("No lesion region was detected in the predicted mask.")

        st.caption(
            "The bounding box is derived from the U-Net segmentation mask. "
            "It is not an independent object detection model."
        )


        # --------------------------------------------------
        # PERMANENT ARCHIVAL & PDF REPORT GENERATION
        # --------------------------------------------------

        st.divider()
        st.subheader("Permanent Archival & PDF Report")
        st.caption(
            "Archive this complete analysis session (images, masks, heatmaps, and metadata) "
            "to results/analysis_history/ and download a publication-ready clinical PDF report."
        )

        col_save1, col_save2 = st.columns([1, 1])

        with col_save1:
            save_clicked = st.button("Save Analysis to History", type="primary", use_container_width=True)
            if save_clicked:
                try:
                    with st.spinner("Archiving analysis artifacts and compiling PDF report..."):
                        src_name = uploaded_file.name if uploaded_file is not None else "dermoscopic_image.png"
                        saved_meta = save_analysis_record(
                            original_image=original_image,
                            analysis_results=results,
                            threshold=threshold,
                            device_name=str(DEVICE),
                            source_filename=src_name
                        )
                        anl_id = saved_meta["analysis_id"]
                        anl_dir = HISTORY_DIR / anl_id
                        pdf_data = generate_pdf_report(anl_dir, saved_meta)

                        st.session_state["saved_analysis_id"] = anl_id
                        st.session_state["saved_pdf_bytes"] = pdf_data
                        st.success(f"Analysis saved successfully! Archived ID: `{anl_id}`")
                except Exception as save_err:
                    st.error("Failed to archive analysis record.")
                    st.exception(save_err)

        with col_save2:
            # If the report has been generated/saved, show the download button for it
            if st.session_state.get("saved_pdf_bytes") is not None:
                st.download_button(
                    label="Download Archived PDF Report",
                    data=st.session_state["saved_pdf_bytes"],
                    file_name=f"{st.session_state['saved_analysis_id']}_report.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
            else:
                # Provide independent on-demand PDF generation without persisting to history
                gen_pdf_btn = st.button("Generate & Download PDF (Without Saving)", use_container_width=True)
                if gen_pdf_btn:
                    try:
                        with st.spinner("Compiling on-demand PDF report..."):
                            import tempfile
                            import shutil
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                src_name = uploaded_file.name if uploaded_file is not None else "dermoscopic_image.png"
                                tmp_meta = save_analysis_record(
                                    original_image=original_image,
                                    analysis_results=results,
                                    threshold=threshold,
                                    device_name=str(DEVICE),
                                    source_filename=src_name,
                                    custom_id=f"TMP_{int(time.time())}"
                                )
                                tmp_saved_dir = HISTORY_DIR / tmp_meta["analysis_id"]
                                pdf_data = generate_pdf_report(tmp_saved_dir, tmp_meta)
                                shutil.rmtree(tmp_saved_dir, ignore_errors=True)
                                st.session_state["independent_pdf_bytes"] = pdf_data
                    except Exception as gen_err:
                        st.error("Failed to compile PDF report.")
                        st.exception(gen_err)

                if st.session_state.get("independent_pdf_bytes") is not None:
                    st.download_button(
                        label="Download Generated PDF Report",
                        data=st.session_state["independent_pdf_bytes"],
                        file_name=f"Triage_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )


# --------------------------------------------------
# RESPONSIBLE USE NOTICE (PRESERVED)
# --------------------------------------------------

st.divider()

st.warning(
    "Research and educational prototype only. "
    "This system is not a medical diagnostic device. "
    "Do not use its predictions to make clinical decisions."
)
