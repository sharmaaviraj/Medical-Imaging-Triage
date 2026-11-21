"""
Components and utilities package for Medical Imaging Triage Workstation.
"""

from .config import (
    ROOT_DIR,
    APP_DIR,
    DATA_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    CLASSIFIER_PATH,
    SEGMENTATION_PATH,
    resolve_image_path,
)
from .device import get_device, get_device_info
from .styles import apply_custom_css, get_plotly_layout, CHART_COLORS
from .sidebar import render_sidebar
from .gradcam import GradCAMExplainer
from .history_manager import save_analysis_record, list_saved_analyses, load_analysis_record, HISTORY_DIR
from .pdf_generator import generate_pdf_report

__all__ = [
    "ROOT_DIR",
    "APP_DIR",
    "DATA_DIR",
    "SPLITS_DIR",
    "MODELS_DIR",
    "CLASSIFIER_PATH",
    "SEGMENTATION_PATH",
    "resolve_image_path",
    "get_device",
    "get_device_info",
    "apply_custom_css",
    "get_plotly_layout",
    "CHART_COLORS",
    "render_sidebar",
    "GradCAMExplainer",
    "save_analysis_record",
    "list_saved_analyses",
    "load_analysis_record",
    "HISTORY_DIR",
    "generate_pdf_report",
]
