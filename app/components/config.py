"""
Configuration, path resolution, and runtime environment settings.
"""

import sys
from pathlib import Path
from typing import Optional

# Determine workspace directories
# config.py is at: ROOT / app / components / config.py
COMPONENTS_DIR = Path(__file__).resolve().parent
APP_DIR = COMPONENTS_DIR.parent
ROOT_DIR = APP_DIR.parent

# Guarantee that both APP_DIR and ROOT_DIR are present in sys.path
for directory in (APP_DIR, ROOT_DIR):
    dir_str = str(directory)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)

# Canonical project paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = ROOT_DIR / "models"

# Checkpoint paths
CLASSIFIER_PATH = MODELS_DIR / "classification" / "best_resnet18.pth"
SEGMENTATION_PATH = MODELS_DIR / "segmentation" / "best_unet.pth"

# Dataset split CSV paths
TRAIN_CSV = SPLITS_DIR / "train.csv"
VALIDATION_CSV = SPLITS_DIR / "validation.csv"
TEST_CSV = SPLITS_DIR / "test.csv"


def resolve_image_path(csv_path_str: str) -> Optional[Path]:
    """
    Resolves image or mask paths recorded in dataset CSV files to local files.

    The original split CSV files contain Google Colab paths, e.g.:
      '/content/medical_imaging/ISBI2016_ISIC_Part1_Training_Data/ISIC_0011295.jpg'
      '/content/medical_imaging/ISBI2016_ISIC_Part1_Training_GroundTruth/ISIC_0011295_Segmentation.png'

    This utility checks:
    1. If the path exists as given (e.g. on Linux or if mounted directly).
    2. Candidate local folders under DATA_DIR based on filename and subfolder.
    3. Returns the resolved local Path if found, otherwise returns None without
       modifying the original CSV or throwing an unhandled exception.
    """
    if not csv_path_str or not isinstance(csv_path_str, str):
        return None

    direct_path = Path(csv_path_str)
    if direct_path.is_file():
        return direct_path

    # Extract basename (e.g. ISIC_0011295.jpg) and parent folder name
    filename = direct_path.name
    parent_folder = direct_path.parent.name

    candidate_locations = [
        RAW_DATA_DIR / parent_folder / filename,
        RAW_DATA_DIR / filename,
        PROCESSED_DATA_DIR / parent_folder / filename,
        PROCESSED_DATA_DIR / filename,
        SAMPLES_DIR / filename,
        DATA_DIR / filename,
    ]

    for candidate in candidate_locations:
        if candidate.is_file():
            return candidate

    return None
