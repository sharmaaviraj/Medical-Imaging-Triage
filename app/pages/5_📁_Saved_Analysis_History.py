"""
Page 5: Saved Analysis History & Clinical Archive
Review past dermoscopic triage analyses, inspect segmentation masks,
examine Grad-CAM feature attribution maps, and download archived PDF reports.
"""

from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd

# Import shared workstation components
from components.styles import apply_custom_css
from components.sidebar import render_sidebar
from components.history_manager import list_saved_analyses, load_analysis_record, HISTORY_DIR
from components.pdf_generator import generate_pdf_report


st.set_page_config(
    page_title="Saved Analysis History | Medical Imaging Triage",
    layout="wide"
)

apply_custom_css()
render_sidebar()

st.markdown("""
<div class="workstation-header">
    <h1>Saved Analysis History & Clinical Archives</h1>
    <p>Permanent repository of past dermoscopy triage evaluations, segmentation masks, Grad-CAM saliency maps, and academic PDF reports.</p>
</div>
""", unsafe_allow_html=True)

# Fetch all saved records
records = list_saved_analyses()

# --------------------------------------------------
# EMPTY STATE HANDLING
# --------------------------------------------------
if not records:
    st.markdown("""
    <div class="clinical-card" style="text-align: center; padding: 2.5rem 1.5rem; margin-top: 1rem;">
        <h3 style="color: #176B9A; margin-bottom: 0.5rem;">No Saved Analyses Found</h3>
        <p style="color: #627D98; max-width: 640px; margin: 0 auto 1.25rem auto;">
            The permanent analysis archive at <code>results/analysis_history/</code> is currently empty.
            When you run an analysis on the primary workstation and click <strong>Save Analysis to History</strong>,
            the original image, U-Net mask, overlay, Grad-CAM attribution, and a publication-quality PDF report
            will be archived here.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 **How to create an archived record:** Return to the main Workstation page, upload an image, run the dual-engine analysis, and click **Save Analysis to History**.")
    st.stop()


# --------------------------------------------------
# ARCHIVE OVERVIEW METRICS
# --------------------------------------------------
total_records = len(records)
malignant_records = sum(1 for r in records if r.get("classification", {}).get("is_malignant", False))
benign_records = total_records - malignant_records
latest_timestamp = records[0].get("timestamp_formatted", records[0].get("timestamp_iso", "N/A"))

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Archived Records", f"{total_records} analyses")
col_m2.metric("Flagged Malignant", f"{malignant_records} ({(malignant_records/max(total_records, 1))*100:.1f}%)")
col_m3.metric("Cleared Benign", f"{benign_records} ({(benign_records/max(total_records, 1))*100:.1f}%)")
col_m4.metric("Latest Archive Date", latest_timestamp.split()[0] if " " in latest_timestamp else latest_timestamp)

st.divider()


# --------------------------------------------------
# SUMMARY TABLE OF ARCHIVED ANALYSES
# --------------------------------------------------
st.subheader("Archived Triage Records")

table_rows = []
for r in records:
    c_meta = r.get("classification", {})
    dims = r.get("image_dimensions", {})
    score = c_meta.get("malignant_score", 0.0)
    thresh = c_meta.get("threshold", 0.50)
    table_rows.append({
        "Analysis ID": r.get("analysis_id", "Unknown"),
        "Timestamp": r.get("timestamp_formatted", r.get("timestamp_iso", "")),
        "Source File": r.get("source_filename", "image.png"),
        "Triage Verdict": c_meta.get("predicted_label", "Unknown"),
        "Malignant Score": f"{score * 100:.2f}%",
        "Cutoff": f"{thresh * 100:.1f}%",
        "Resolution": f"{dims.get('width', '?')} × {dims.get('height', '?')}",
        "Grad-CAM": "Available" if r.get("explainability", {}).get("available") else "Not Available",
        "Device": r.get("system_telemetry", {}).get("device", "CPU")
    })

df_summary = pd.DataFrame(table_rows)
st.dataframe(df_summary, use_container_width=True, hide_index=True)

st.divider()


# --------------------------------------------------
# DETAILED RECORD INSPECTOR
# --------------------------------------------------
st.subheader("Inspect Individual Archived Analysis")

id_options = [r["analysis_id"] for r in records]


def format_record_label(anl_id: str) -> str:
    match = next((r for r in records if r["analysis_id"] == anl_id), None)
    if match:
        lbl = match.get("classification", {}).get("predicted_label", "Unknown")
        sc = match.get("classification", {}).get("malignant_score", 0.0)
        ts = match.get("timestamp_formatted", match.get("timestamp_iso", ""))
        fn = match.get("source_filename", "")
        return f"{anl_id}  |  {lbl.upper()} ({sc*100:.1f}%)  |  {fn}  |  {ts}"
    return anl_id


selected_id = st.selectbox(
    "Select an archived analysis to review:",
    options=id_options,
    format_func=format_record_label
)

active_record = load_analysis_record(selected_id)

if active_record is None:
    st.error(f"Could not load archive for ID: `{selected_id}`. The directory may be corrupted or deleted.")
    st.stop()

record_dir = Path(active_record["_dir_path"])
clf = active_record.get("classification", {})
seg = active_record.get("segmentation", {})
expl = active_record.get("explainability", {})
is_mal = clf.get("is_malignant", False)
score_val = clf.get("malignant_score", 0.0)
thresh_val = clf.get("threshold", 0.50)

# Card with Key Metadata and Download Button
col_info, col_download = st.columns([2, 1])

with col_info:
    st.markdown(f"""
    <div class="clinical-card">
        <h4 style="margin-top: 0; color: #102A43;">Analysis Overview: <code>{selected_id}</code></h4>
        <p style="margin-bottom: 0.4rem;">
            <strong>Timestamp:</strong> {active_record.get('timestamp_formatted', active_record.get('timestamp_iso', 'N/A'))} &nbsp;|&nbsp;
            <strong>Source Image:</strong> <code>{active_record.get('source_filename', 'N/A')}</code> &nbsp;|&nbsp;
            <strong>Inference Hardware:</strong> <code>{active_record.get('system_telemetry', {}).get('device', 'CPU')}</code>
        </p>
        <p style="margin-bottom: 0;">
            <strong>Malignancy Score:</strong> <code>{score_val*100:.2f}%</code> &nbsp;|&nbsp;
            <strong>Operational Cutoff:</strong> <code>{thresh_val*100:.1f}%</code> &nbsp;|&nbsp;
            <strong>Triage Classification:</strong> <span style="font-weight: 700; color: {'#B91C1C' if is_mal else '#047857'};">{clf.get('predicted_label', 'Unknown').upper()}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_download:
    st.markdown("**Report Retrieval**")
    pdf_path = record_dir / "report.pdf"
    pdf_bytes = None

    if pdf_path.exists():
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
        except Exception:
            pdf_bytes = None

    # Re-generate if missing
    if pdf_bytes is None:
        try:
            pdf_bytes = generate_pdf_report(record_dir, active_record)
        except Exception as e:
            st.error(f"Could not compile PDF: {e}")

    if pdf_bytes is not None:
        st.download_button(
            label="Download Archived PDF Report",
            data=pdf_bytes,
            file_name=f"{selected_id}_report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
        st.caption("Publication-ready 2-page A4 PDF report.")


# --------------------------------------------------
# VISUALIZATION GALLERY
# --------------------------------------------------
st.markdown("#### Primary Lesion Morphology & Segmentation")

col_img1, col_img2, col_img3 = st.columns(3)

with col_img1:
    orig_path = record_dir / "original_image.png"
    if orig_path.exists():
        st.image(str(orig_path), caption="1. Acquired Original Input", use_container_width=True)
    else:
        st.warning("Original image missing from archive.")

with col_img2:
    mask_path = record_dir / "segmentation_mask.png"
    if mask_path.exists():
        st.image(str(mask_path), caption="2. U-Net Predicted Binary Mask", use_container_width=True)
    else:
        st.warning("Segmentation mask not recorded.")

with col_img3:
    overlay_path = record_dir / "segmentation_overlay.png"
    if overlay_path.exists():
        st.image(str(overlay_path), caption="3. Lesion Overlay (Green Alpha)", use_container_width=True)
    else:
        st.warning("Overlay image not recorded.")

st.markdown("#### Feature Attribution & Spatial Bounding")

col_img4, col_img5, col_img6 = st.columns(3)

with col_img4:
    heat_path = record_dir / "gradcam_heatmap.png"
    if heat_path.exists():
        st.image(str(heat_path), caption="4. Grad-CAM Saliency Heatmap", use_container_width=True)
    else:
        st.caption("Grad-CAM heatmap not recorded for this analysis.")

with col_img5:
    g_over_path = record_dir / "gradcam_overlay.png"
    if g_over_path.exists():
        st.image(str(g_over_path), caption="5. Attribution Overlay", use_container_width=True)
    else:
        st.caption("Grad-CAM overlay not recorded for this analysis.")

with col_img6:
    bbox_path = record_dir / "bounding_box.png"
    if bbox_path.exists():
        st.image(str(bbox_path), caption="6. Derived Bounding Box (Red)", use_container_width=True)
        coords = seg.get("bounding_box_coordinates")
        if coords:
            st.caption(f"Coordinates: x={coords['x']}, y={coords['y']}, w={coords['width']}, h={coords['height']}")
    else:
        st.caption("Bounding box not detected or recorded.")

# Metadata JSON inspection
with st.expander("Inspect Raw Serialized Metadata (`metadata.json`)", expanded=False):
    st.json(active_record)
