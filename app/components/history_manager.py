"""
History Manager for Medical Imaging Triage Workstation
Handles persistence, directory management, image serialization,
and metadata validation for historical analysis records.
"""

import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np

import components.config as cfg

# Root directory for historical analysis archives
HISTORY_DIR = cfg.ROOT_DIR / "results" / "analysis_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def generate_analysis_id() -> str:
    """
    Generates a collision-resistant unique analysis identifier.
    Format: ANL_YYYYMMDD_HHMMSS_<short_uuid>
    """
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6].upper()
    return f"ANL_{timestamp_str}_{short_uuid}"


def save_analysis_record(
    original_image: Image.Image,
    analysis_results: Dict[str, Any],
    threshold: float,
    device_name: str,
    source_filename: str = "uploaded_image.png",
    custom_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Persists an analysis session to results/analysis_history/<ANALYSIS_ID>/
    Saves:
      - original_image.png
      - segmentation_mask.png
      - segmentation_overlay.png
      - gradcam_heatmap.png (if available)
      - gradcam_overlay.png (if available)
      - bounding_box.png (if available)
      - metadata.json
    Returns the metadata dictionary.
    """
    analysis_id = custom_id or generate_analysis_id()
    target_dir = HISTORY_DIR / analysis_id

    # Defensive path collision avoidance
    counter = 1
    while target_dir.exists():
        target_dir = HISTORY_DIR / f"{analysis_id}_{counter}"
        counter += 1
    target_dir.mkdir(parents=True, exist_ok=True)
    analysis_id = target_dir.name  # Sync analysis_id with final unique directory name

    # 1. Save Original Image
    orig_path = target_dir / "original_image.png"
    if isinstance(original_image, Image.Image):
        original_image.save(orig_path, format="PNG")
        width, height = original_image.size
    else:
        orig_pil = Image.fromarray(np.uint8(original_image))
        orig_pil.save(orig_path, format="PNG")
        width, height = orig_pil.size

    # 2. Save U-Net Segmentation Mask
    mask_saved = False
    if "predicted_mask" in analysis_results and analysis_results["predicted_mask"] is not None:
        mask_arr = analysis_results["predicted_mask"]
        if mask_arr.dtype != np.uint8:
            mask_arr = (mask_arr > 0).astype(np.uint8) * 255
        elif mask_arr.max() == 1:
            mask_arr = mask_arr * 255
        Image.fromarray(mask_arr).save(target_dir / "segmentation_mask.png", format="PNG")
        mask_saved = True

    # 3. Save Segmentation Overlay
    overlay_saved = False
    if "overlay" in analysis_results and analysis_results["overlay"] is not None:
        overlay_arr = np.uint8(analysis_results["overlay"])
        Image.fromarray(overlay_arr).save(target_dir / "segmentation_overlay.png", format="PNG")
        overlay_saved = True

    # 4. Save Bounding Box Image
    bbox_saved = False
    if "boxed_image" in analysis_results and analysis_results["boxed_image"] is not None:
        boxed_arr = np.uint8(analysis_results["boxed_image"])
        Image.fromarray(boxed_arr).save(target_dir / "bounding_box.png", format="PNG")
        bbox_saved = True

    # 5. Save Grad-CAM Heatmap & Overlay (Optional)
    gradcam_saved = False
    if "gradcam_heatmap_rgb" in analysis_results and analysis_results["gradcam_heatmap_rgb"] is not None:
        cam_arr = np.uint8(analysis_results["gradcam_heatmap_rgb"])
        Image.fromarray(cam_arr).save(target_dir / "gradcam_heatmap.png", format="PNG")
        gradcam_saved = True

    if "gradcam_overlay" in analysis_results and analysis_results["gradcam_overlay"] is not None:
        cam_ov_arr = np.uint8(analysis_results["gradcam_overlay"])
        Image.fromarray(cam_ov_arr).save(target_dir / "gradcam_overlay.png", format="PNG")

    # 6. Build Metadata
    score = float(analysis_results.get("malignant_score", 0.0))
    is_malignant = score >= threshold
    label = "Malignant" if is_malignant else "Benign"

    bbox_coords = None
    if analysis_results.get("bounding_box") is not None:
        bx, by, bw, bh = analysis_results["bounding_box"]
        bbox_coords = {"x": int(bx), "y": int(by), "width": int(bw), "height": int(bh)}

    metadata = {
        "analysis_id": analysis_id,
        "timestamp_iso": datetime.datetime.now().isoformat(),
        "timestamp_formatted": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_filename": Path(source_filename).name,
        "image_dimensions": {"width": width, "height": height},
        "classification": {
            "model_architecture": "ResNet-18 (Custom Head Linear(512, 1))",
            "checkpoint_name": "best_resnet18.pth",
            "malignant_score": round(score, 4),
            "threshold": round(threshold, 2),
            "predicted_label": label,
            "is_malignant": bool(is_malignant)
        },
        "segmentation": {
            "model_architecture": "Custom 4-Stage U-Net",
            "checkpoint_name": "best_unet.pth",
            "mask_saved": mask_saved,
            "overlay_saved": overlay_saved,
            "bounding_box_coordinates": bbox_coords
        },
        "explainability": {
            "method": "Grad-CAM (Layer: layer4[-1])",
            "available": gradcam_saved
        },
        "system_telemetry": {
            "device": str(device_name),
            "framework": "PyTorch 2.15+cpu",
            "workstation_version": "Phase 4.1"
        },
        "artifacts": {
            "directory": str(target_dir.name),
            "original_image": "original_image.png",
            "segmentation_mask": "segmentation_mask.png" if mask_saved else None,
            "segmentation_overlay": "segmentation_overlay.png" if overlay_saved else None,
            "bounding_box": "bounding_box.png" if bbox_saved else None,
            "gradcam_heatmap": "gradcam_heatmap.png" if gradcam_saved else None,
            "gradcam_overlay": "gradcam_overlay.png" if gradcam_saved else None,
            "report_pdf": "report.pdf"
        }
    }

    with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def list_saved_analyses() -> List[Dict[str, Any]]:
    """
    Scans results/analysis_history/ and returns a sorted list of all valid
    analysis metadata dictionaries (newest first).
    Gracefully skips incomplete or corrupted directories.
    """
    if not HISTORY_DIR.exists():
        return []

    records = []
    for entry in HISTORY_DIR.iterdir():
        if entry.is_dir():
            meta_file = entry / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["_dir_path"] = str(entry)
                        records.append(data)
                except Exception:
                    # Gracefully skip corrupted JSON files
                    continue

    # Sort descending by ISO timestamp
    records.sort(key=lambda r: r.get("timestamp_iso", ""), reverse=True)
    return records


def load_analysis_record(analysis_id: str) -> Optional[Dict[str, Any]]:
    """
    Loads the metadata and paths for a specific analysis ID.
    Returns None if not found or corrupted.
    """
    target_dir = HISTORY_DIR / analysis_id
    if not target_dir.exists() or not target_dir.is_dir():
        return None

    meta_file = target_dir / "metadata.json"
    if not meta_file.exists():
        return None

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_dir_path"] = str(target_dir)
            return data
    except Exception:
        return None
