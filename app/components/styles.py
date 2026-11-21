"""
Shared clinical UI styling and theme management for the workstation.
Provides a modern, professional blue clinical/research aesthetic:
- Deep navy: #102A43
- Primary blue: #176B9A
- Accent blue: #3B82B8
- Light blue: #EAF3FA
- Main background: #F5F8FC
- Card background: #FFFFFF
- Primary text: #243B53
- Secondary text: #627D98
- Borders: #D9E5EF
"""

from typing import Dict, Any
import streamlit as st

# Coordinated chart color palette for Plotly charts
CHART_COLORS = {
    "deep_navy": "#102A43",
    "primary_blue": "#176B9A",
    "accent_blue": "#3B82B8",
    "muted_teal": "#0F766E",
    "light_blue": "#82B1D6",
    "neutral_grey": "#627D98",
    "border_blue": "#D9E5EF",
    "danger_red": "#B91C1C",   # Preserved for critical medical alerts & unregularized loss
    "success_green": "#047857", # Preserved for benign diagnosis
}


def apply_custom_css():
    """
    Injects global CSS rules implementing the consistent Blue Clinical/Research theme.
    """
    custom_css = """
    <style>
    /* Global Application Canvas */
    .stApp {
        background-color: #F5F8FC !important;
        color: #243B53;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1240px;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #102A43 !important;
        font-weight: 600;
        letter-spacing: -0.015em;
    }

    p, span, label, div {
        color: #243B53;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #D9E5EF !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown strong {
        color: #102A43 !important;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #627D98 !important;
    }

    /* Sidebar Navigation Links */
    div[data-testid="stSidebarNav"] a {
        color: #243B53 !important;
        border-radius: 4px;
        margin: 2px 0;
        padding: 6px 12px;
        transition: background-color 0.15s ease, color 0.15s ease;
    }

    div[data-testid="stSidebarNav"] a:hover {
        background-color: #EAF3FA !important;
        color: #176B9A !important;
    }

    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #EAF3FA !important;
        color: #102A43 !important;
        font-weight: 600 !important;
        border-left: 3px solid #176B9A !important;
    }

    /* Clinical Workstation Header */
    .workstation-header {
        background-color: #FFFFFF;
        border: 1px solid #D9E5EF;
        border-left: 4px solid #176B9A;
        border-radius: 4px;
        padding: 1.15rem 1.4rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(16, 42, 67, 0.04);
    }
    .workstation-header h1 {
        color: #102A43 !important;
        font-size: 1.55rem;
        font-weight: 700;
        margin: 0;
    }
    .workstation-header p {
        color: #627D98 !important;
        font-size: 0.88rem;
        margin-top: 0.30rem;
        margin-bottom: 0;
    }

    /* Clinical Cards */
    .clinical-card {
        background-color: #FFFFFF;
        border: 1px solid #D9E5EF;
        border-radius: 4px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(16, 42, 67, 0.03);
    }

    /* Structured Clinical Notices */
    .disclaimer-box {
        border-left: 3px solid #176B9A;
        background-color: #EAF3FA;
        color: #243B53;
        padding: 0.75rem 1rem;
        font-size: 0.84rem;
        border-radius: 0 4px 4px 0;
        margin-top: 0.85rem;
        margin-bottom: 0.85rem;
        border-top: 1px solid #D9E5EF;
        border-right: 1px solid #D9E5EF;
        border-bottom: 1px solid #D9E5EF;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D9E5EF !important;
        border-radius: 4px !important;
        padding: 0.70rem 0.90rem !important;
        box-shadow: 0 1px 2px rgba(16, 42, 67, 0.03) !important;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: 0.78rem !important;
        color: #627D98 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        color: #102A43 !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background-color: #176B9A !important;
        border: 1px solid #176B9A !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.10rem !important;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }

    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: #102A43 !important;
        border-color: #102A43 !important;
        color: #FFFFFF !important;
    }

    button[kind="secondary"], .stButton > button {
        background-color: #FFFFFF !important;
        border: 1px solid #D9E5EF !important;
        color: #243B53 !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
    }

    button[kind="secondary"]:hover, .stButton > button:hover {
        background-color: #EAF3FA !important;
        border-color: #3B82B8 !important;
        color: #176B9A !important;
    }

    /* Tabs Styling */
    div[data-baseweb="tab-list"] {
        border-bottom: 2px solid #D9E5EF !important;
        gap: 8px !important;
    }

    button[data-baseweb="tab"] {
        color: #627D98 !important;
        font-weight: 500 !important;
        font-size: 0.90rem !important;
        padding: 8px 16px !important;
        border: none !important;
        border-radius: 4px 4px 0 0 !important;
        background-color: transparent !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #176B9A !important;
        background-color: #EAF3FA !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #102A43 !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #176B9A !important;
        background-color: #FFFFFF !important;
    }

    /* Sliders */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #176B9A !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(16, 42, 67, 0.25) !important;
    }

    /* Selectboxes and Inputs */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #D9E5EF !important;
        border-radius: 4px !important;
        color: #243B53 !important;
    }

    div[data-baseweb="select"] > div:focus-within {
        border-color: #176B9A !important;
    }

    /* Tables & DataFrames */
    div[data-testid="stDataFrame"] {
        border: 1px solid #D9E5EF !important;
        border-radius: 4px !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 2px rgba(16, 42, 67, 0.02) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D9E5EF !important;
        border-radius: 4px !important;
    }

    /* Dividers */
    hr {
        border-color: #D9E5EF !important;
        margin: 1.25rem 0 !important;
    }

    /* Captions */
    .stCaption, small {
        color: #627D98 !important;
        font-size: 0.82rem;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def get_plotly_layout(
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int = 360,
    show_legend: bool = True
) -> Dict[str, Any]:
    """
    Generates a unified Plotly layout adhering to the Blue Clinical workstation aesthetic.
    """
    return dict(
        title=dict(
            text=title,
            font=dict(family="-apple-system, sans-serif", size=13, color=CHART_COLORS["deep_navy"]),
            x=0.01,
            y=0.96
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=11, color=CHART_COLORS["primary_blue"])),
            gridcolor="#EAF3FA",
            linecolor=CHART_COLORS["border_blue"],
            zerolinecolor="#EAF3FA",
            tickfont=dict(size=10, color=CHART_COLORS["secondary_text" if "secondary_text" in CHART_COLORS else "neutral_grey"])
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=11, color=CHART_COLORS["primary_blue"])),
            gridcolor="#EAF3FA",
            linecolor=CHART_COLORS["border_blue"],
            zerolinecolor="#EAF3FA",
            tickfont=dict(size=10, color=CHART_COLORS["neutral_grey"])
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=45, r=20, t=40, b=40),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color=CHART_COLORS["deep_navy"]),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=CHART_COLORS["border_blue"],
            borderwidth=1
        )
    )
