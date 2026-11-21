"""
Shared sidebar component providing system telemetry, model status,
and workstation navigation across all pages.
Designed with a compact, professional clinical-research presentation
aligned with the Blue theme (#102A43, #176B9A, #627D98).
"""

import streamlit as st
from .config import CLASSIFIER_PATH, SEGMENTATION_PATH
from .device import get_device_info


def render_sidebar():
    """
    Renders standardized, compact sidebar telemetry across all pages.
    """
    with st.sidebar:
        st.markdown("**System Telemetry**")

        dev_info = get_device_info()
        st.caption(f"Compute: {dev_info['compute_target']}")
        st.caption(f"PyTorch: v{dev_info['pytorch_version']}")

        st.divider()

        st.markdown("**Model Weights**")

        col1, col2 = st.columns(2)
        with col1:
            if CLASSIFIER_PATH.exists():
                size_mb = CLASSIFIER_PATH.stat().st_size / (1024 * 1024)
                st.caption(f"ResNet-18\n\nOnline ({size_mb:.1f} MB)")
            else:
                st.caption("ResNet-18\n\nOffline")

        with col2:
            if SEGMENTATION_PATH.exists():
                size_mb = SEGMENTATION_PATH.stat().st_size / (1024 * 1024)
                st.caption(f"U-Net\n\nOnline ({size_mb:.1f} MB)")
            else:
                st.caption("U-Net\n\nOffline")

        st.divider()

        st.markdown("**Workstation Navigation**")
        st.caption(
            "• Live Triage: Primary workstation\n"
            "• Dataset Analytics: ISBI 2016 splits\n"
            "• Model Architecture: Neural topology\n"
            "• Clinical Governance: Scope & ethics\n"
            "• Optimization Experiments: Deep Learning laboratory"
        )

        st.divider()

        st.caption(
            "Academic Research Prototype\n"
            "ISBI 2016 Part 1 Benchmark"
        )
