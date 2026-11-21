"""
Hardware detection and device management.
Ensures reliable fallback to CPU on systems without CUDA support (such as
Intel HD Graphics systems with CPU-only PyTorch builds).
"""

import platform
import torch


def get_device() -> torch.device:
    """
    Returns torch.device('cuda') if CUDA is available and functional;
    otherwise safely falls back to torch.device('cpu').
    """
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


def get_device_info() -> dict:
    """
    Returns diagnostic telemetry for the current computing environment.
    """
    device = get_device()
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass

    if cuda_available and device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        compute_target = f"NVIDIA GPU: {name}"
    else:
        processor = platform.processor() or "Host CPU"
        compute_target = f"CPU Execution ({processor})"

    return {
        "device": device,
        "device_type": device.type.upper(),
        "compute_target": compute_target,
        "cuda_available": cuda_available,
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
    }
