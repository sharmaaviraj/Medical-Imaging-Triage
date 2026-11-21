"""
PDF Report Generator for Medical Imaging Triage Workstation
Generates publication-quality, multi-page clinical/academic PDF reports
using matplotlib's native vector/raster PDF engine.
Adheres strictly to the project's clinical blue theme:
- Deep Navy: #102A43
- Primary Blue: #176B9A
- Accent Blue: #3B82B8
- Light Blue: #EAF3FA
- Dark Slate: #243B53
- Muted Grey: #627D98
- Border: #D9E5EF
- Danger Red: #B91C1C
- Success Green: #047857
"""

import io
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches


# Theme Palette Constants
C_NAVY = "#102A43"
C_PRIMARY = "#176B9A"
C_ACCENT = "#3B82B8"
C_LIGHT_BG = "#F5F8FC"
C_BOX_BG = "#EAF3FA"
C_TEXT = "#243B53"
C_GREY = "#627D98"
C_BORDER = "#D9E5EF"
C_DANGER = "#B91C1C"
C_SUCCESS = "#047857"


def generate_pdf_report(
    analysis_dir: Path,
    metadata: Dict[str, Any]
) -> bytes:
    """
    Compiles a comprehensive 2-page A4 clinical research report from saved artifacts
    in analysis_dir and returns the PDF file bytes.
    Also writes the file directly to analysis_dir / 'report.pdf'.
    """
    pdf_buffer = io.BytesIO()

    # Load artifacts safely
    def load_img(fname: str) -> Optional[np.ndarray]:
        p = analysis_dir / fname
        if p.exists():
            try:
                with Image.open(p) as img:
                    return np.array(img.convert("RGB"))
            except Exception:
                return None
        return None

    img_orig = load_img("original_image.png")
    img_mask = load_img("segmentation_mask.png")
    img_overlay = load_img("segmentation_overlay.png")
    img_bbox = load_img("bounding_box.png")
    img_cam_heat = load_img("gradcam_heatmap.png")
    img_cam_over = load_img("gradcam_overlay.png")

    clf_meta = metadata.get("classification", {})
    seg_meta = metadata.get("segmentation", {})
    sys_meta = metadata.get("system_telemetry", {})
    dims = metadata.get("image_dimensions", {})
    analysis_id = metadata.get("analysis_id", "N/A")
    timestamp = metadata.get("timestamp_formatted", metadata.get("timestamp_iso", "N/A"))
    src_file = metadata.get("source_filename", "N/A")

    score = clf_meta.get("malignant_score", 0.0)
    threshold = clf_meta.get("threshold", 0.50)
    is_malignant = clf_meta.get("is_malignant", score >= threshold)
    label = clf_meta.get("predicted_label", "Malignant" if is_malignant else "Benign")

    with PdfPages(pdf_buffer) as pdf:
        # =====================================================================
        # PAGE 1: EXECUTIVE SUMMARY, TRIAGE VERDICT & PRIMARY MORPHOLOGY
        # =====================================================================
        fig1 = plt.figure(figsize=(8.27, 11.69), dpi=150)  # Standard A4 size
        fig1.patch.set_facecolor("#FFFFFF")

        # Top Header Banner
        ax_top = fig1.add_axes([0.06, 0.90, 0.88, 0.07])
        ax_top.axis("off")
        ax_top.text(0.0, 0.70, "MEDICAL IMAGING TRIAGE & LESION ANALYSIS WORKSTATION",
                    fontsize=14, fontweight="bold", color=C_NAVY, family="sans-serif")
        ax_top.text(0.0, 0.30, "Academic Deep Learning Clinical Decision-Support & Morphological Delineation Report",
                    fontsize=8.5, color=C_GREY, family="sans-serif")
        # Header Accent Line
        rect_line = patches.Rectangle((0.0, 0.05), 1.0, 0.05, facecolor=C_PRIMARY, edgecolor="none", transform=ax_top.transAxes)
        ax_top.add_patch(rect_line)

        # Telemetry & Metadata Box
        ax_meta = fig1.add_axes([0.06, 0.76, 0.88, 0.12])
        ax_meta.axis("off")
        box_meta = patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.03",
                                          facecolor=C_BOX_BG, edgecolor=C_BORDER, linewidth=1,
                                          transform=ax_meta.transAxes)
        ax_meta.add_patch(box_meta)

        ax_meta.text(0.03, 0.75, f"Analysis ID: {analysis_id}", fontsize=9, fontweight="bold", color=C_NAVY, family="sans-serif")
        ax_meta.text(0.03, 0.48, f"Acquisition Timestamp: {timestamp}", fontsize=8.5, color=C_TEXT, family="sans-serif")
        ax_meta.text(0.03, 0.22, f"Source File: {src_file} ({dims.get('width', 'N/A')} × {dims.get('height', 'N/A')} px)",
                     fontsize=8.5, color=C_TEXT, family="sans-serif")

        ax_meta.text(0.53, 0.75, f"Classifier: {clf_meta.get('model_architecture', 'ResNet-18')}", fontsize=8.5, color=C_NAVY, family="sans-serif")
        ax_meta.text(0.53, 0.48, f"Segmenter: {seg_meta.get('model_architecture', 'Custom U-Net')}", fontsize=8.5, color=C_NAVY, family="sans-serif")
        ax_meta.text(0.53, 0.22, f"Execution Device: {sys_meta.get('device', 'CPU')} | PyTorch 2.15", fontsize=8.5, color=C_GREY, family="sans-serif")

        # Triage Verdict Card
        verdict_color = C_DANGER if is_malignant else C_SUCCESS
        verdict_bg = "#FDF2F2" if is_malignant else "#F0FDF4"
        verdict_border = "#FCA5A5" if is_malignant else "#86EFAC"

        ax_triage = fig1.add_axes([0.06, 0.63, 0.88, 0.11])
        ax_triage.axis("off")
        box_triage = patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.04",
                                            facecolor=verdict_bg, edgecolor=verdict_border, linewidth=1.5,
                                            transform=ax_triage.transAxes)
        ax_triage.add_patch(box_triage)

        ax_triage.text(0.04, 0.65, f"TRIAGE OUTCOME: {label.upper()}",
                       fontsize=13, fontweight="bold", color=verdict_color, family="sans-serif")
        ax_triage.text(0.04, 0.35, f"Raw Malignancy Score: {score * 100:.2f}%  |  Active Cutoff Threshold: {threshold * 100:.1f}%",
                       fontsize=10, fontweight="bold", color=C_NAVY, family="sans-serif")
        ax_triage.text(0.04, 0.12,
                       "Notice: Score represents the sigmoid logit output of ResNet-18, not a calibrated clinical probability.",
                       fontsize=7.5, fontstyle="italic", color=C_GREY, family="sans-serif")

        # Primary Morphology Visualizations (3 Columns)
        ax_title1 = fig1.add_axes([0.06, 0.57, 0.88, 0.03])
        ax_title1.axis("off")
        ax_title1.text(0.0, 0.2, "PRIMARY DERMOSCOPIC MORPHOLOGY & SEGMENTATION",
                       fontsize=10.5, fontweight="bold", color=C_NAVY, family="sans-serif")

        # 1. Original Image
        ax_im1 = fig1.add_axes([0.06, 0.25, 0.27, 0.30])
        ax_im1.axis("off")
        if img_orig is not None:
            ax_im1.imshow(img_orig)
        ax_im1.set_title("1. Acquired Input", fontsize=8.5, fontweight="bold", color=C_NAVY, pad=6)

        # 2. U-Net Mask
        ax_im2 = fig1.add_axes([0.365, 0.25, 0.27, 0.30])
        ax_im2.axis("off")
        if img_mask is not None:
            ax_im2.imshow(img_mask, cmap="gray")
        else:
            ax_im2.text(0.5, 0.5, "Mask Not Available", ha="center", va="center", color=C_GREY)
        ax_im2.set_title("2. U-Net Binary Mask", fontsize=8.5, fontweight="bold", color=C_NAVY, pad=6)

        # 3. Green Overlay
        ax_im3 = fig1.add_axes([0.67, 0.25, 0.27, 0.30])
        ax_im3.axis("off")
        if img_overlay is not None:
            ax_im3.imshow(img_overlay)
        else:
            ax_im3.text(0.5, 0.5, "Overlay Not Available", ha="center", va="center", color=C_GREY)
        ax_im3.set_title("3. Lesion Overlay (Green)", fontsize=8.5, fontweight="bold", color=C_NAVY, pad=6)

        # Section Explanatory Footer Card
        ax_note1 = fig1.add_axes([0.06, 0.09, 0.88, 0.12])
        ax_note1.axis("off")
        box_note1 = patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.015,rounding_size=0.02",
                                           facecolor="#FFFFFF", edgecolor=C_BORDER, linewidth=1,
                                           transform=ax_note1.transAxes)
        ax_note1.add_patch(box_note1)
        ax_note1.text(0.02, 0.75, "Segmentation Pipeline Summary:", fontsize=8.5, fontweight="bold", color=C_NAVY)
        ax_note1.text(0.02, 0.48, "• Deep convolutional U-Net (7,765,985 parameters) processes normalized 256×256 RGB input tensor.",
                      fontsize=7.8, color=C_TEXT)
        ax_note1.text(0.02, 0.26, "• Binary mask generated by thresholding spatial sigmoid logits at 0.50, upsampled to original resolution.",
                      fontsize=7.8, color=C_TEXT)
        ax_note1.text(0.02, 0.06, "• Alpha blending (50% original image + 50% green [0, 255, 0]) highlights verified lesion boundaries.",
                      fontsize=7.8, color=C_TEXT)

        # Page 1 Footer
        ax_foot1 = fig1.add_axes([0.06, 0.03, 0.88, 0.03])
        ax_foot1.axis("off")
        ax_foot1.text(0.0, 0.5, "Medical Imaging Triage Workstation • Academic Research Report", fontsize=7.5, color=C_GREY)
        ax_foot1.text(1.0, 0.5, "Page 1 of 2", fontsize=7.5, color=C_GREY, ha="right")

        pdf.savefig(fig1)
        plt.close(fig1)

        # =====================================================================
        # PAGE 2: EXPLAINABILITY (GRAD-CAM), BOUNDING BOX & CLINICAL GOVERNANCE
        # =====================================================================
        fig2 = plt.figure(figsize=(8.27, 11.69), dpi=150)
        fig2.patch.set_facecolor("#FFFFFF")

        # Page 2 Top Header
        ax_top2 = fig2.add_axes([0.06, 0.90, 0.88, 0.07])
        ax_top2.axis("off")
        ax_top2.text(0.0, 0.70, "FEATURE ATTRIBUTION & DERIVED MORPHOLOGY",
                     fontsize=14, fontweight="bold", color=C_NAVY, family="sans-serif")
        ax_top2.text(0.0, 0.30, f"Analysis ID: {analysis_id}  |  Grad-CAM Saliency & Contour Analysis",
                     fontsize=8.5, color=C_GREY, family="sans-serif")
        rect_line2 = patches.Rectangle((0.0, 0.05), 1.0, 0.05, facecolor=C_PRIMARY, edgecolor="none", transform=ax_top2.transAxes)
        ax_top2.add_patch(rect_line2)

        # Secondary Visualizations (3 Columns: Grad-CAM Heatmap, Grad-CAM Overlay, Bounding Box)
        # 1. Grad-CAM Heatmap
        ax_cam1 = fig2.add_axes([0.06, 0.58, 0.27, 0.28])
        ax_cam1.axis("off")
        if img_cam_heat is not None:
            ax_cam1.imshow(img_cam_heat)
        else:
            ax_cam1.text(0.5, 0.5, "Grad-CAM Heatmap\nNot Available", ha="center", va="center", color=C_GREY, fontsize=8)
        ax_cam1.set_title("4. Grad-CAM Heatmap", fontsize=8.5, fontweight="bold", color=C_NAVY, pad=6)

        # 2. Grad-CAM Overlay
        ax_cam2 = fig2.add_axes([0.365, 0.58, 0.27, 0.28])
        ax_cam2.axis("off")
        if img_cam_over is not None:
            ax_cam2.imshow(img_cam_over)
        else:
            ax_cam2.text(0.5, 0.5, "Grad-CAM Overlay\nNot Available", ha="center", va="center", color=C_GREY, fontsize=8)
        ax_cam2.set_title("5. Attribution Overlay", fontsize=8.5, fontweight="bold", color=C_NAVY, pad=6)

        # 3. Bounding Box
        ax_cam3 = fig2.add_axes([0.67, 0.58, 0.27, 0.28])
        ax_cam3.axis("off")
        if img_bbox is not None:
            ax_cam3.imshow(img_bbox)
        else:
            ax_cam3.text(0.5, 0.5, "Bounding Box\nNot Available", ha="center", va="center", color=C_GREY, fontsize=8)
        ax_cam3.set_title("6. Derived Bounding Box", fontsize=8.5, fontweight="bold", color=C_NAVY, pad=6)

        # Geometric & Architectural Notes
        ax_geom = fig2.add_axes([0.06, 0.38, 0.88, 0.16])
        ax_geom.axis("off")
        box_geom = patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.015,rounding_size=0.02",
                                          facecolor=C_BOX_BG, edgecolor=C_BORDER, linewidth=1,
                                          transform=ax_geom.transAxes)
        ax_geom.add_patch(box_geom)

        bbox_c = seg_meta.get("bounding_box_coordinates")
        bbox_str = f"x={bbox_c['x']}, y={bbox_c['y']}, width={bbox_c['width']}, height={bbox_c['height']}" if bbox_c else "No lesion contour detected"

        ax_geom.text(0.02, 0.82, "Technical Explainability & Contour Specifications:", fontsize=8.5, fontweight="bold", color=C_NAVY)
        ax_geom.text(0.02, 0.62, "• Grad-CAM Target Layer: model.layer4[-1] (ResNet-18 stage 4, 512 channels, 7×7 feature map).", fontsize=7.8, color=C_TEXT)
        ax_geom.text(0.02, 0.44, f"• Derived Bounding Box: {bbox_str}.", fontsize=7.8, color=C_TEXT)
        ax_geom.text(0.02, 0.26, "• Architectural Clarification: The bounding box is an OpenCV contour post-processing artifact (cv2.boundingRect)", fontsize=7.8, fontstyle="italic", color=C_TEXT)
        ax_geom.text(0.02, 0.08, "  extracted from the U-Net mask. It is NOT generated by an independent object detector (e.g. YOLO/RetinaNet).", fontsize=7.8, fontstyle="italic", color=C_TEXT)

        # Academic Governance & Clinical Disclaimer Box
        ax_disc = fig2.add_axes([0.06, 0.10, 0.88, 0.24])
        ax_disc.axis("off")
        box_disc = patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.03",
                                          facecolor="#FFFBEB", edgecolor="#FDE68A", linewidth=1.5,
                                          transform=ax_disc.transAxes)
        ax_disc.add_patch(box_disc)

        ax_disc.text(0.03, 0.86, "ACADEMIC RESEARCH & CLINICAL GOVERNANCE DISCLAIMER",
                     fontsize=9, fontweight="bold", color="#92400E", family="sans-serif")
        disclaimer_text = (
            "1. PROTOTYPE STATUS: This software is developed strictly as an academic research and technical portfolio prototype.\n"
            "   It is NOT cleared, certified, or registered as Software as a Medical Device (SaMD) by the FDA, EMA, or other regulatory bodies.\n"
            "2. NO CLINICAL DIAGNOSIS: Predictions, segmentations, and heatmaps in this report are experimental deep learning outputs.\n"
            "   Do not use this document to formulate clinical diagnoses, guide surgical biopsies, or direct patient therapy.\n"
            "3. UNCALIBRATED SCORES: Malignancy scores represent uncalibrated sigmoid outputs from ResNet-18, not validated clinical probabilities.\n"
            "4. EXPLAINABILITY LIMITS: Grad-CAM heatmaps highlight image regions that mathematically influenced model activations.\n"
            "   They do NOT provide biological proof, histological evidence, or validated dermatological etiology."
        )
        ax_disc.text(0.03, 0.14, disclaimer_text, fontsize=7.3, color="#78350F", family="monospace", linespacing=1.45)

        # Page 2 Footer
        ax_foot2 = fig2.add_axes([0.06, 0.03, 0.88, 0.03])
        ax_foot2.axis("off")
        ax_foot2.text(0.0, 0.5, "Medical Imaging Triage Workstation • Academic Research Report", fontsize=7.5, color=C_GREY)
        ax_foot2.text(1.0, 0.5, "Page 2 of 2", fontsize=7.5, color=C_GREY, ha="right")

        pdf.savefig(fig2)
        plt.close(fig2)

    pdf_bytes = pdf_buffer.getvalue()

    # Write file directly to target archive
    report_path = analysis_dir / "report.pdf"
    with open(report_path, "wb") as f:
        f.write(pdf_bytes)

    return pdf_bytes
