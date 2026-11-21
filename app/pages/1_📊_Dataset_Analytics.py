"""
Page 1: Dataset & Split Analytics (ISBI 2016 Part 1 Benchmark)
Explores class distribution, split stratifications, and metadata verification.
Styled with the Blue Clinical theme (#102A43, #176B9A, #EAF3FA).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Import shared workstation components
import components.config as cfg
from components.styles import apply_custom_css, get_plotly_layout, CHART_COLORS
from components.sidebar import render_sidebar

st.set_page_config(
    page_title="Dataset Analytics | Medical Imaging Triage",
    layout="wide"
)

apply_custom_css()
render_sidebar()

st.markdown("""
<div class="workstation-header">
    <h1>ISBI 2016 Dataset & Split Analytics</h1>
    <p>Empirical distribution analysis across training, validation, and test partitions.</p>
</div>
""", unsafe_allow_html=True)

# Dataset Summary Overview
col1, col2, col3, col4 = st.columns(4)

splits_data = {}
for name, csv_path in [
    ("Training", cfg.TRAIN_CSV),
    ("Validation", cfg.VALIDATION_CSV),
    ("Test", cfg.TEST_CSV)
]:
    if csv_path.exists():
        splits_data[name] = pd.read_csv(csv_path)

if splits_data:
    total_samples = sum(len(df) for df in splits_data.values())
    train_count = len(splits_data.get("Training", []))
    val_count = len(splits_data.get("Validation", []))
    test_count = len(splits_data.get("Test", []))

    col1.metric("Total Cohort", f"{total_samples} samples")
    col2.metric("Training Split", f"{train_count} (70.0%)")
    col3.metric("Validation Split", f"{val_count} (15.0%)")
    col4.metric("Test Split", f"{test_count} (15.0%)")

    st.divider()

    st.subheader("Split Diagnosis Breakdown")
    summary_rows = []
    splits_names = []
    benign_counts = []
    malignant_counts = []

    for split_name, df in splits_data.items():
        counts = df["diagnosis"].value_counts()
        benign_n = int(counts.get("benign", 0))
        malignant_n = int(counts.get("malignant", 0))
        total_n = len(df)
        splits_names.append(split_name)
        benign_counts.append(benign_n)
        malignant_counts.append(malignant_n)

        summary_rows.append({
            "Split": split_name,
            "Total": total_n,
            "Benign (n)": benign_n,
            "Benign (%)": f"{(benign_n / total_n) * 100:.1f}%",
            "Malignant (n)": malignant_n,
            "Malignant (%)": f"{(malignant_n / total_n) * 100:.1f}%",
            "Imbalance Ratio": f"{benign_n / max(malignant_n, 1):.2f} : 1"
        })

    summary_df = pd.DataFrame(summary_rows)

    col_tbl, col_chart = st.columns([1, 1])

    with col_tbl:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="disclaimer-box">
            <strong>Observation on Class Stratification:</strong> All three splits exhibit consistent
            stratification (~80.8% benign to ~19.2% malignant). This ~4:1 class imbalance is characteristic of
            dermatological screening cohorts and directly influences model calibration and decision thresholds.
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            name="Benign",
            x=splits_names,
            y=benign_counts,
            marker_color=CHART_COLORS["primary_blue"]
        ))
        fig_dist.add_trace(go.Bar(
            name="Malignant",
            x=splits_names,
            y=malignant_counts,
            marker_color=CHART_COLORS["deep_navy"]
        ))
        layout = get_plotly_layout(
            title="Class Distribution Across Cohort Partitions",
            xaxis_title="Partition",
            yaxis_title="Sample Count",
            height=300
        )
        layout["barmode"] = "group"
        fig_dist.update_layout(layout)
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()
    st.subheader("Inspect Split Records")
    selected_split = st.selectbox("Select partition to browse:", list(splits_data.keys()))
    df_to_view = splits_data[selected_split]

    resolved_count = sum(1 for p in df_to_view["image_path"] if cfg.resolve_image_path(p) is not None)
    st.caption(
        f"Displaying {len(df_to_view)} records from `{selected_split.lower()}.csv`. "
        f"Locally mounted images: {resolved_count}/{len(df_to_view)}"
    )

    st.dataframe(df_to_view.head(25), use_container_width=True)

else:
    st.warning("Split CSV files were not found in `data/splits/`.")
