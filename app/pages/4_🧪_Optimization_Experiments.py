"""
Page 4: Deep Learning Foundations & Optimization Laboratory
Empirical evaluation of neural network optimization, weight initialization,
learning rate dynamics, and L2 regularization on a 641-parameter MLP.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import shared components and paths
import components.config as cfg
from components.styles import apply_custom_css, get_plotly_layout, CHART_COLORS
from components.sidebar import render_sidebar

# Import experiment model definitions
sys.path.insert(0, str(cfg.ROOT_DIR / "experiments" / "optimizers"))
from models import SimpleMLP, generate_benchmark_data, generate_overfitting_data


# --------------------------------------------------
# PAGE CONFIGURATION & STYLING
# --------------------------------------------------

st.set_page_config(
    page_title="Optimization Experiments | Medical Imaging Triage",
    layout="wide"
)

apply_custom_css()
render_sidebar()

st.markdown("""
<div class="workstation-header">
    <h1>Optimization Experiments & Deep Learning Laboratory</h1>
    <p>Empirical study of optimizer convergence, learning rate dynamics, weight initialization, and regularization on a 641-parameter MLP.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>Educational Laboratory Notice:</strong> All benchmarks in this module are evaluated on controlled
    synthetic feature manifolds (make_moons) to isolate first-order optimization dynamics, gradient backpropagation,
    and generalization mechanics. These experiments are mathematically and functionally independent from the primary
    ISBI 2016 dermoscopy triage ResNet-18 and U-Net models.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD BENCHMARK JSON DATA
# --------------------------------------------------

BENCHMARKS_PATH = cfg.ROOT_DIR / "results" / "experiments" / "benchmarks.json"

@st.cache_data
def get_benchmarks_data() -> Optional[Dict[str, Any]]:
    if not BENCHMARKS_PATH.exists():
        return None
    with open(BENCHMARKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

bench_data = get_benchmarks_data()

if bench_data is None:
    st.error(f"Benchmark results file not found at: `{BENCHMARKS_PATH}`. Please execute `experiment_runner.py` first.")
    st.stop()


# --------------------------------------------------
# ARCHITECTURE SUMMARY CARD (641 PARAMETERS)
# --------------------------------------------------

meta = bench_data.get("metadata", {})
total_params = meta.get("total_parameters", 641)

with st.expander("Neural Architecture Inspection: SimpleMLP (641 Parameters)", expanded=False):
    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        st.markdown("**Network Topology**")
        st.markdown("""
        - **Input Layer**: 2 features ($x_1, x_2$)
        - **Hidden Layer 1**: $\\text{Linear}(2 \\to 32) + \\text{ReLU}$
        - **Hidden Layer 2**: $\\text{Linear}(32 \\to 16) + \\text{ReLU}$
        - **Output Layer**: $\\text{Linear}(16 \\to 1)$ raw logit
        - **Loss Function**: `BCEWithLogitsLoss`
        - **Trainable Parameters**: **641**
        """)
    with col_a2:
        st.markdown("**Parameter Breakdown by Layer**")
        param_rows = []
        for name, info in meta.get("parameter_breakdown", {}).items():
            param_rows.append({
                "Layer Parameter": name,
                "Tensor Shape": str(info["shape"]),
                "Parameter Count": info["count"],
                "Calculation": (
                    "2 × 32 = 64 weights" if name == "fc1.weight" else
                    "32 biases" if name == "fc1.bias" else
                    "32 × 16 = 512 weights" if name == "fc2.weight" else
                    "16 biases" if name == "fc2.bias" else
                    "16 × 1 = 16 weights" if name == "fc3.weight" else
                    "1 output bias"
                )
            })
        st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)


# --------------------------------------------------
# MAIN WORKSTATION TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Optimizer Comparison",
    "Learning Rate & Initialization",
    "Overfitting & Regularization",
    "Live Training Sandbox"
])


# ==================================================
# TAB 1: OPTIMIZER COMPARISON
# ==================================================

with tab1:
    st.subheader("1. First-Order Optimizer Convergence Comparison")
    st.caption("Empirical convergence comparison across Batch Gradient Descent, Mini-batch SGD, Polyak Momentum, and Nesterov Accelerated Gradient.")

    suite_opt = bench_data.get("suite_optimizers", {})

    # Comparative Metrics Summary Table
    summary_rows = []
    for key in ["bgd", "sgd", "momentum", "nag"]:
        if key in suite_opt:
            v = suite_opt[key]
            summary_rows.append({
                "Optimizer": v["name"],
                "Batch Size": v["batch_size"],
                "Updates / Epoch": v["updates_per_epoch"],
                "Total Updates (80 Ep)": v["updates_per_epoch"] * 80,
                "Final Train Loss": f"{v['final_train_loss']:.4f}",
                "Final Val Loss": f"{v['final_val_loss']:.4f}",
                "Final Val Accuracy": f"{v['final_val_acc'] * 100:.1f}%",
                "CPU Time (s)": f"{v['wall_clock_seconds']:.3f}"
            })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Plotly Interactive Loss Curves
    col_chart1, col_chart2 = st.columns(2)

    color_map = {
        "bgd": CHART_COLORS["neutral_grey"],      # Neutral Grey (#627D98)
        "sgd": CHART_COLORS["deep_navy"],         # Deep Navy (#102A43)
        "momentum": CHART_COLORS["primary_blue"], # Primary Blue (#176B9A)
        "nag": CHART_COLORS["muted_teal"]         # Muted Teal (#0F766E)
    }

    with col_chart1:
        fig_train = go.Figure()
        for key in ["bgd", "sgd", "momentum", "nag"]:
            if key in suite_opt:
                v = suite_opt[key]
                epochs_x = list(range(1, len(v["train_losses"]) + 1))
                fig_train.add_trace(go.Scatter(
                    x=epochs_x,
                    y=v["train_losses"],
                    mode="lines",
                    name=v["name"],
                    line=dict(color=color_map.get(key, CHART_COLORS["primary_blue"]), width=2)
                ))
        fig_train.update_layout(
            get_plotly_layout(
                title="Training Loss vs. Epochs (80 Epochs, lr=0.05)",
                xaxis_title="Training Epoch",
                yaxis_title="Binary Cross-Entropy Loss",
                height=360
            )
        )
        st.plotly_chart(fig_train, use_container_width=True)

    with col_chart2:
        fig_val = go.Figure()
        for key in ["bgd", "sgd", "momentum", "nag"]:
            if key in suite_opt:
                v = suite_opt[key]
                epochs_x = list(range(1, len(v["val_losses"]) + 1))
                fig_val.add_trace(go.Scatter(
                    x=epochs_x,
                    y=v["val_losses"],
                    mode="lines",
                    name=v["name"],
                    line=dict(color=color_map.get(key, CHART_COLORS["primary_blue"]), width=2)
                ))
        fig_val.update_layout(
            get_plotly_layout(
                title="Validation Loss vs. Epochs (80 Epochs, lr=0.05)",
                xaxis_title="Training Epoch",
                yaxis_title="Binary Cross-Entropy Loss",
                height=360
            )
        )
        st.plotly_chart(fig_val, use_container_width=True)

    # Update Frequency & Fair-Comparison Discussion
    st.markdown("""
    <div class="clinical-card">
        <h4>Fair Comparison & Update Frequency Analysis</h4>
        <p>
            When comparing optimizers across a fixed number of epochs (80 epochs), the update frequency creates an important distinction:
        </p>
        <ul>
            <li><strong>Batch Gradient Descent, Momentum, and NAG</strong> operate on the full dataset (N=700), performing <strong>1 parameter update per epoch</strong> (80 total updates). All three start from the exact same cloned initial weights vector. Under this identical step budget, <strong>Polyak Momentum</strong> accelerates convergence by accumulating velocity in consistent directions, while <strong>Nesterov Accelerated Gradient (NAG)</strong> achieves slightly lower loss by evaluating gradients at a look-ahead position to dampen overshoot.</li>
            <li><strong>Mini-batch SGD (B=32)</strong> performs <strong>22 parameter updates per epoch</strong> (1,760 total updates). Consequently, its rapid loss decline per epoch is primarily driven by a 22× higher parameter update frequency, coupled with gradient stochasticity that assists in navigating non-convex regions.</li>
            <li><em>Methodological Note:</em> The curves above represent the exact measured loss values at each training epoch. No synthetic interpolation or artificial step-normalization has been applied.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Mathematical Update Equations
    st.divider()
    st.subheader("Mathematical Update Formulations")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**1. Batch Gradient Descent (BGD)**")
        st.latex(r"g_t = \frac{1}{N} \sum_{i=1}^N \nabla_\theta \mathcal{L}_i(\theta_t)")
        st.latex(r"\theta_{t+1} = \theta_t - \eta g_t")
        st.caption("Exact empirical gradient over all N samples. Zero gradient variance, smooth monotonic trajectory.")

        st.markdown("**3. Polyak Classical Momentum**")
        st.latex(r"v_{t+1} = \gamma v_t + \eta g_t")
        st.latex(r"\theta_{t+1} = \theta_t - v_{t+1}")
        st.caption("Velocity accumulation along low-curvature ravines; dampens high-frequency perpendicular oscillations (γ=0.9).")

    with col_m2:
        st.markdown("**2. Mini-batch Stochastic Gradient Descent (SGD)**")
        st.latex(r"g_{t, \mathcal{B}} = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_\theta \mathcal{L}_i(\theta_t)")
        st.latex(r"\theta_{t+1} = \theta_t - \eta g_{t, \mathcal{B}}")
        st.caption("Stochastic gradient estimated from batch B=32. Gradient variance acts as regularizing noise.")

        st.markdown("**4. Nesterov Accelerated Gradient (NAG)**")
        st.latex(r"v_{t+1} = \gamma v_t + \eta \nabla_\theta \mathcal{L}(\theta_t - \gamma v_t)")
        st.latex(r"\theta_{t+1} = \theta_t - v_{t+1}")
        st.caption("Evaluates gradient at the anticipated look-ahead position (θ_t - γ v_t), providing predictive braking.")


# ==================================================
# TAB 2: LEARNING RATE & INITIALIZATION
# ==================================================

with tab2:
    st.subheader("2. Learning Rate Dynamics & Weight Initialization")
    st.caption("Empirical analysis of step-size sensitivity and hidden layer symmetry breaking.")

    # Section A: Learning Rate Dynamics
    st.markdown("### Part A: Learning Rate Dynamics")
    suite_lr = bench_data.get("suite_learning_rates", {})

    fig_lr = go.Figure()
    lr_colors = {
        "lr_low": CHART_COLORS["neutral_grey"],       # Slate/Grey (#627D98)
        "lr_moderate": CHART_COLORS["primary_blue"],  # Primary Blue (#176B9A)
        "lr_high": CHART_COLORS["accent_blue"],       # Accent Blue (#3B82B8)
        "lr_divergent": CHART_COLORS["danger_red"]    # Critical Alert (#B91C1C) - early oscillation
    }

    for key, item in suite_lr.items():
        epochs_x = list(range(1, len(item["train_losses"]) + 1))
        fig_lr.add_trace(go.Scatter(
            x=epochs_x,
            y=item["train_losses"],
            mode="lines",
            name=f"{item['label']} (Final: {item['final_train_loss']:.4f})",
            line=dict(color=lr_colors.get(key, CHART_COLORS["primary_blue"]), width=2)
        ))

    fig_lr.update_layout(
        get_plotly_layout(
            title="Training Loss Trajectories Across Learning Rates (Batch GD, 60 Epochs)",
            xaxis_title="Training Epoch",
            yaxis_title="Binary Cross-Entropy Loss",
            height=380
        )
    )
    st.plotly_chart(fig_lr, use_container_width=True)

    st.markdown("""
    <div class="clinical-card">
        <h4>Empirical Trajectory Findings:</h4>
        <ul>
            <li><strong>Low (η = 0.005)</strong>: Loss descends slowly (from 0.79 to 0.53). Step size is too small, requiring hundreds of additional epochs to approach convergence.</li>
            <li><strong>Moderate / Standard (η = 0.10)</strong>: Steady, monotonic exponential convergence (from 0.65 to 0.28) with zero upward oscillations.</li>
            <li><strong>High (η = 0.80)</strong>: Rapid initial descent (reaching 0.35 by epoch 2), settling at 0.14 without numerical instability.</li>
            <li><strong>Excessive (η = 2.50) — Early Oscillatory Instability</strong>: The loss does <em>not</em> diverge to infinity or NaN, but it exhibits severe early instability. Between Epoch 1 and Epoch 2, the loss spikes from 1.03 to 2.28 (+1.25 jump) due to overshooting the local valley, experiencing 7 separate upward oscillations before gradients diminish and the trajectory stabilizes.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Section B: Weight Initialization & Symmetry Breaking
    st.markdown("### Part B: Weight Initialization & Symmetry Breaking")
    suite_init = bench_data.get("suite_initializations", {})

    col_init_table, col_init_chart = st.columns([1, 1])

    with col_init_table:
        st.markdown("**Initialization Comparison Summary**")
        init_rows = []
        for key in ["zeros", "random_normal", "xavier_uniform", "he_normal"]:
            if key in suite_init:
                v = suite_init[key]
                init_rows.append({
                    "Method": v["name"],
                    "Final Train Loss": f"{v['final_train_loss']:.4f}",
                    "Final Val Loss": f"{v['final_val_loss']:.4f}",
                    "Final Val Accuracy": f"{v['final_val_acc'] * 100:.1f}%"
                })
        st.dataframe(pd.DataFrame(init_rows), use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="disclaimer-box">
            <strong>The Mathematical Reality of Zero Initialization:</strong><br>
            When weights are initialized to zero ($W=0$), all hidden neurons compute $z = 0$, producing $\\text{ReLU}(0) = 0$.
            During backpropagation, all incoming weights to a hidden layer receive the exact same gradient.
            Because the weights update identically, the hidden neurons never break symmetry.
            On balanced data, the network output remains at 50% accuracy ($P=0.5$), completely unable to learn non-linear boundaries.
        </div>
        """, unsafe_allow_html=True)

    with col_init_chart:
        fig_init = go.Figure()
        init_colors = {
            "zeros": CHART_COLORS["neutral_grey"],          # Neutral Grey (#627D98) - symmetry unbroken
            "random_normal": CHART_COLORS["accent_blue"],   # Accent Blue (#3B82B8)
            "xavier_uniform": CHART_COLORS["muted_teal"],   # Muted Teal (#0F766E)
            "he_normal": CHART_COLORS["primary_blue"]       # Primary Blue (#176B9A)
        }
        for key in ["zeros", "random_normal", "xavier_uniform", "he_normal"]:
            if key in suite_init:
                v = suite_init[key]
                epochs_x = list(range(1, len(v["val_losses"]) + 1))
                fig_init.add_trace(go.Scatter(
                    x=epochs_x,
                    y=v["val_losses"],
                    mode="lines",
                    name=v["name"],
                    line=dict(color=init_colors.get(key, CHART_COLORS["primary_blue"]), width=2)
                ))
        fig_init.update_layout(
            get_plotly_layout(
                title="Validation Loss by Initialization (Momentum, 60 Epochs)",
                xaxis_title="Training Epoch",
                yaxis_title="Validation Loss",
                height=340
            )
        )
        st.plotly_chart(fig_init, use_container_width=True)


# ==================================================
# TAB 3: OVERFITTING & REGULARIZATION
# ==================================================

with tab3:
    st.subheader("3. Overfitting & L2 Regularization (Weight Decay)")
    st.caption("Empirical demonstration of the generalization gap on small sample sizes (N_train = 60 noisy samples).")

    suite_reg = bench_data.get("suite_regularization", {})

    col_reg_metrics, col_reg_chart = st.columns([1, 2])

    with col_reg_metrics:
        unreg = suite_reg.get("unregularized", {})
        reg = suite_reg.get("l2_regularized", {})

        st.markdown("**Empirical Generalization Metrics**")
        st.markdown(f"""
        **Unregularized (Weight Decay = 0.0)**
        - Final Train Loss: `{unreg.get('final_train_loss', 0):.4f}`
        - Final Val Loss: `{unreg.get('final_val_loss', 0):.4f}`
        - **Generalization Gap**: `{unreg.get('final_generalization_gap', 0):+.4f}`
        
        ---
        **L2 Regularized (Weight Decay = 0.04)**
        - Final Train Loss: `{reg.get('final_train_loss', 0):.4f}`
        - Final Val Loss: `{reg.get('final_val_loss', 0):.4f}`
        - **Generalization Gap**: `{reg.get('final_generalization_gap', 0):+.4f}`
        """)

        st.caption(
            "L2 regularization penalizes large weight magnitudes, reducing the generalization gap by ~70% "
            "and improving validation loss from 0.61 to 0.38."
        )

    with col_reg_chart:
        fig_reg = go.Figure()

        if "unregularized" in suite_reg and "l2_regularized" in suite_reg:
            epochs_x = list(range(1, len(unreg["train_losses"]) + 1))

            # Unregularized
            fig_reg.add_trace(go.Scatter(
                x=epochs_x, y=unreg["train_losses"],
                mode="lines", name="Unreg: Train Loss",
                line=dict(color=CHART_COLORS["danger_red"], width=1.5, dash="dash")
            ))
            fig_reg.add_trace(go.Scatter(
                x=epochs_x, y=unreg["val_losses"],
                mode="lines", name="Unreg: Val Loss (Overfit)",
                line=dict(color=CHART_COLORS["danger_red"], width=2.5)
            ))

            # Regularized
            fig_reg.add_trace(go.Scatter(
                x=epochs_x, y=reg["train_losses"],
                mode="lines", name="L2 Reg: Train Loss",
                line=dict(color=CHART_COLORS["primary_blue"], width=1.5, dash="dash")
            ))
            fig_reg.add_trace(go.Scatter(
                x=epochs_x, y=reg["val_losses"],
                mode="lines", name="L2 Reg: Val Loss (Bounded)",
                line=dict(color=CHART_COLORS["primary_blue"], width=2.5)
            ))

            fig_reg.update_layout(
                get_plotly_layout(
                    title="Train vs. Validation Loss: Generalization Gap (120 Epochs)",
                    xaxis_title="Training Epoch",
                    yaxis_title="Binary Cross-Entropy Loss",
                    height=360
                )
            )
            st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("""
    <div class="clinical-card">
        <h4>Mathematical Mechanism of L2 Weight Decay:</h4>
        <p>In standard gradient descent, L2 regularization adds a quadratic penalty proportional to the squared Euclidean norm of the weight vectors:</p>
        <p style="text-align: center;">$$\\mathcal{L}_{\\text{reg}}(\\theta) = \\mathcal{L}(\\theta) + \\frac{\\lambda}{2} \\sum_l \\|W_l\\|_2^2$$</p>
        <p>Taking the gradient with respect to $W$ yields the modified parameter update:</p>
        <p style="text-align: center;">$$W_{t+1} = W_t - \\eta \\left( \\nabla_W \\mathcal{L} + \\lambda W_t \\right) = (1 - \\eta \\lambda) W_t - \\eta \\nabla_W \\mathcal{L}$$</p>
        <p>The term $(1 - \\eta \\lambda) < 1$ shrinks the weight vector at every iteration before subtracting the gradient step. This prevents individual weights from growing excessively large to fit spurious noise in small training samples.</p>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# TAB 4: LIVE TRAINING SANDBOX
# ==================================================

with tab4:
    st.subheader("4. Live Interactive Training Sandbox")
    st.caption("Interactively train the 641-parameter SimpleMLP on CPU with custom hyperparameters and visualize the resulting 2D decision boundary.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        sandbox_opt = st.selectbox(
            "Optimizer",
            ["Batch Gradient Descent (BGD)", "Mini-batch SGD (B=32)", "Polyak Momentum (γ=0.9)", "Nesterov Accelerated Gradient (NAG)"],
            index=2
        )
        sandbox_lr = st.slider("Learning Rate (η)", min_value=0.01, max_value=0.50, value=0.05, step=0.01)

    with col_ctrl2:
        sandbox_epochs = st.slider("Training Epochs", min_value=20, max_value=120, value=60, step=10)
        sandbox_init = st.selectbox("Weight Initialization", ["he_normal", "xavier_uniform", "random_normal", "zeros"], index=0)

    with col_ctrl3:
        sandbox_wd = st.slider("L2 Weight Decay (λ)", min_value=0.0, max_value=0.10, value=0.0, step=0.01)
        sandbox_seed = st.number_input("Deterministic Random Seed", min_value=1, max_value=9999, value=42, step=1)

    train_btn = st.button("Train Model (Live on CPU)", type="primary")

    if train_btn:
        with st.spinner("Training SimpleMLP on CPU..."):
            # Prepare dataset
            X_tr, y_tr, X_va, y_va = generate_benchmark_data(random_state=sandbox_seed)
            loss_fn = nn.BCEWithLogitsLoss()

            torch.manual_seed(sandbox_seed)
            model = SimpleMLP()
            model.initialize_weights(sandbox_init, seed=sandbox_seed)

            # Configure optimizer
            if sandbox_opt.startswith("Batch"):
                opt = torch.optim.SGD(model.parameters(), lr=sandbox_lr, weight_decay=sandbox_wd)
                batch_size = len(X_tr)
            elif sandbox_opt.startswith("Mini-batch"):
                opt = torch.optim.SGD(model.parameters(), lr=sandbox_lr, weight_decay=sandbox_wd)
                batch_size = 32
            elif sandbox_opt.startswith("Polyak"):
                opt = torch.optim.SGD(model.parameters(), lr=sandbox_lr, momentum=0.9, weight_decay=sandbox_wd)
                batch_size = len(X_tr)
            else:
                opt = torch.optim.SGD(model.parameters(), lr=sandbox_lr, momentum=0.9, nesterov=True, weight_decay=sandbox_wd)
                batch_size = len(X_tr)

            tr_losses = []
            va_losses = []

            t_start = time.time()
            n_samples = len(X_tr)

            for epoch in range(sandbox_epochs):
                model.train()
                if batch_size >= n_samples:
                    opt.zero_grad()
                    out = model(X_tr)
                    loss = loss_fn(out, y_tr)
                    loss.backward()
                    opt.step()
                else:
                    perm = torch.randperm(n_samples)
                    for start_idx in range(0, n_samples, batch_size):
                        b_idx = perm[start_idx : start_idx + batch_size]
                        opt.zero_grad()
                        loss = loss_fn(model(X_tr[b_idx]), y_tr[b_idx])
                        loss.backward()
                        opt.step()

                model.eval()
                with torch.no_grad():
                    tr_loss = float(loss_fn(model(X_tr), y_tr).item())
                    va_loss = float(loss_fn(model(X_va), y_va).item())
                    tr_losses.append(tr_loss)
                    va_losses.append(va_loss)

            total_time = time.time() - t_start

            # Compute final validation accuracy
            with torch.no_grad():
                val_probs = model.predict_proba(X_va)
                val_preds = (val_probs >= 0.5).float()
                val_acc = float((val_preds == y_va).float().mean().item())

        st.success(f"Training completed in {total_time:.3f} seconds on CPU ({sandbox_epochs} epochs).")

        col_sb_m1, col_sb_m2, col_sb_m3, col_sb_m4 = st.columns(4)
        col_sb_m1.metric("Final Train Loss", f"{tr_losses[-1]:.4f}")
        col_sb_m2.metric("Final Val Loss", f"{va_losses[-1]:.4f}")
        col_sb_m3.metric("Final Val Accuracy", f"{val_acc * 100:.1f}%")
        col_sb_m4.metric("CPU Latency", f"{total_time:.3f} s")

        # Results Plots: Loss Curve + Decision Boundary
        col_res_plot1, col_res_plot2 = st.columns(2)

        with col_res_plot1:
            fig_sb_loss = go.Figure()
            epochs_x = list(range(1, len(tr_losses) + 1))
            fig_sb_loss.add_trace(go.Scatter(
                x=epochs_x, y=tr_losses, mode="lines", name="Train Loss",
                line=dict(color=CHART_COLORS["primary_blue"], width=2)
            ))
            fig_sb_loss.add_trace(go.Scatter(
                x=epochs_x, y=va_losses, mode="lines", name="Val Loss",
                line=dict(color=CHART_COLORS["deep_navy"], width=2, dash="dash")
            ))
            fig_sb_loss.update_layout(
                get_plotly_layout(
                    title="Live Loss Trajectory",
                    xaxis_title="Epoch",
                    yaxis_title="BCEWithLogitsLoss",
                    height=340
                )
            )
            st.plotly_chart(fig_sb_loss, use_container_width=True)

        with col_res_plot2:
            # Generate 2D decision boundary contour
            x_min, x_max = float(X_tr[:, 0].min() - 0.5), float(X_tr[:, 0].max() + 0.5)
            y_min, y_max = float(X_tr[:, 1].min() - 0.5), float(X_tr[:, 1].max() + 0.5)
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
            grid_t = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

            with torch.no_grad():
                grid_probs = model.predict_proba(grid_t).numpy().reshape(xx.shape)

            fig_boundary = go.Figure()
            fig_boundary.add_trace(go.Contour(
                x=np.linspace(x_min, x_max, 50),
                y=np.linspace(y_min, y_max, 50),
                z=grid_probs,
                colorscale=[
                    [0.0, "#EAF3FA"],
                    [0.5, "#D9E5EF"],
                    [1.0, "#82B1D6"]
                ],
                showscale=False,
                opacity=0.6,
                contours=dict(start=0.1, end=0.9, size=0.1)
            ))

            # Overlay scatter points (Validation set)
            y_va_np = y_va.squeeze().numpy()
            X_va_np = X_va.numpy()
            fig_boundary.add_trace(go.Scatter(
                x=X_va_np[y_va_np == 0, 0], y=X_va_np[y_va_np == 0, 1],
                mode="markers", name="Class 0 (Val)",
                marker=dict(color=CHART_COLORS["primary_blue"], size=6, opacity=0.85)
            ))
            fig_boundary.add_trace(go.Scatter(
                x=X_va_np[y_va_np == 1, 0], y=X_va_np[y_va_np == 1, 1],
                mode="markers", name="Class 1 (Val)",
                marker=dict(color=CHART_COLORS["deep_navy"], size=6, opacity=0.85)
            ))

            fig_boundary.update_layout(
                get_plotly_layout(
                    title="Learned 2D Decision Boundary",
                    xaxis_title="Feature 1",
                    yaxis_title="Feature 2",
                    height=340
                )
            )
            st.plotly_chart(fig_boundary, use_container_width=True)
