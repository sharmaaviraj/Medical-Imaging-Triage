"""
Experiment runner for Deep Learning foundational experiments:
Optimizer comparisons, learning rate dynamics, weight initialization,
and L2 regularization benchmarks.

Outputs structured results to results/experiments/benchmarks.json.
All experiments run deterministically on CPU.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
import numpy as np

# Ensure project modules can be loaded
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CURRENT_DIR))

from models import SimpleMLP, generate_benchmark_data, generate_overfitting_data


def compute_metrics(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module
) -> Dict[str, float]:
    """Computes binary cross-entropy loss and classification accuracy."""
    with torch.no_grad():
        logits = model(X)
        loss = float(loss_fn(logits, y).item())
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        acc = float((preds == y).float().mean().item())
    return {"loss": loss, "accuracy": acc}


def run_optimizer_suite(seed: int = 42, epochs: int = 80, lr: float = 0.05) -> Dict[str, Any]:
    """
    Suite 1: Optimizer Convergence Comparison.
    Compares Batch Gradient Descent, Mini-batch SGD, Momentum, and Nesterov Accelerated Gradient.
    Uses identical data splits, identical architecture, identical initial weights, and same seed.
    """
    X_train, y_train, X_val, y_val = generate_benchmark_data(random_state=seed)
    loss_fn = nn.BCEWithLogitsLoss()

    # Create baseline model and save initial weights to guarantee identical starting point
    torch.manual_seed(seed)
    base_model = SimpleMLP()
    base_model.initialize_weights("he_normal", seed=seed)
    initial_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    configs = [
        {
            "name": "Batch Gradient Descent (BGD)",
            "key": "bgd",
            "batch_size": len(X_train),
            "momentum": 0.0,
            "nesterov": False,
            "updates_per_epoch": 1,
            "description": "Full dataset evaluation. Exact empirical gradient with zero stochastic variance."
        },
        {
            "name": "Mini-batch SGD (B=32)",
            "key": "sgd",
            "batch_size": 32,
            "momentum": 0.0,
            "nesterov": False,
            "updates_per_epoch": int(np.ceil(len(X_train) / 32)),
            "description": "Small batch stochastic updates. High variance gradient trajectory aiding local minimum escape."
        },
        {
            "name": "Polyak Momentum (γ=0.9)",
            "key": "momentum",
            "batch_size": len(X_train),
            "momentum": 0.9,
            "nesterov": False,
            "updates_per_epoch": 1,
            "description": "Velocity accumulation along consistent gradient directions; dampens ravines."
        },
        {
            "name": "Nesterov Accelerated Gradient (NAG)",
            "key": "nag",
            "batch_size": len(X_train),
            "momentum": 0.9,
            "nesterov": True,
            "updates_per_epoch": 1,
            "description": "Look-ahead gradient evaluation acting as a predictive brake to suppress momentum overshoot."
        }
    ]

    results = {}
    n_samples = len(X_train)

    for cfg in configs:
        model = SimpleMLP()
        model.load_state_dict({k: v.clone() for k, v in initial_state.items()})

        if cfg["nesterov"]:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg["momentum"], nesterov=True)
        elif cfg["momentum"] > 0:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg["momentum"])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        val_accuracies = []

        t0 = time.time()
        for epoch in range(epochs):
            model.train()
            b_size = cfg["batch_size"]

            if b_size >= n_samples:
                # Full batch gradient descent
                optimizer.zero_grad()
                logits = model(X_train)
                loss = loss_fn(logits, y_train)
                loss.backward()
                optimizer.step()
            else:
                # Mini-batch gradient descent
                perm = torch.randperm(n_samples)
                for start_idx in range(0, n_samples, b_size):
                    batch_idx = perm[start_idx : start_idx + b_size]
                    xb = X_train[batch_idx]
                    yb = y_train[batch_idx]
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = loss_fn(out, yb)
                    loss.backward()
                    optimizer.step()

            # Record epoch validation
            model.eval()
            tr_metrics = compute_metrics(model, X_train, y_train, loss_fn)
            va_metrics = compute_metrics(model, X_val, y_val, loss_fn)

            train_losses.append(tr_metrics["loss"])
            val_losses.append(va_metrics["loss"])
            val_accuracies.append(va_metrics["accuracy"])

        elapsed = time.time() - t0

        results[cfg["key"]] = {
            "name": cfg["name"],
            "description": cfg["description"],
            "updates_per_epoch": cfg["updates_per_epoch"],
            "batch_size": cfg["batch_size"],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_val_acc": val_accuracies[-1],
            "wall_clock_seconds": round(elapsed, 4)
        }

    return results


def run_learning_rate_suite(seed: int = 42, epochs: int = 60) -> Dict[str, Any]:
    """
    Suite 2: Learning Rate Dynamics.
    Observes convergence behavior across low, moderate, high, and divergent learning rates.
    """
    X_train, y_train, X_val, y_val = generate_benchmark_data(random_state=seed)
    loss_fn = nn.BCEWithLogitsLoss()

    torch.manual_seed(seed)
    base_model = SimpleMLP()
    base_model.initialize_weights("he_normal", seed=seed)
    initial_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    lr_configs = [
        {"key": "lr_low", "label": "Low (η=0.005)", "lr": 0.005, "behavior": "Sluggish, slow initial descent"},
        {"key": "lr_moderate", "label": "Moderate / Standard (η=0.10)", "lr": 0.10, "behavior": "Steady exponential convergence"},
        {"key": "lr_high", "label": "High (η=0.80)", "lr": 0.80, "behavior": "Aggressive initial descent with gradient oscillations"},
        {"key": "lr_divergent", "label": "Excessive (η=2.50)", "lr": 2.50, "behavior": "Overshooting local valley, loss instability"}
    ]

    results = {}
    for cfg in lr_configs:
        model = SimpleMLP()
        model.load_state_dict({k: v.clone() for k, v in initial_state.items()})
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])

        train_losses = []
        val_losses = []

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()

            tr_m = compute_metrics(model, X_train, y_train, loss_fn)
            va_m = compute_metrics(model, X_val, y_val, loss_fn)
            train_losses.append(tr_m["loss"])
            val_losses.append(va_m["loss"])

        results[cfg["key"]] = {
            "label": cfg["label"],
            "lr": cfg["lr"],
            "behavior": cfg["behavior"],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1]
        }

    return results


def run_initialization_suite(seed: int = 42, epochs: int = 60, lr: float = 0.05) -> Dict[str, Any]:
    """
    Suite 3: Weight Initialization Comparison.
    Compares Zero, Random Normal, Xavier Uniform, and He Normal initializations.
    """
    X_train, y_train, X_val, y_val = generate_benchmark_data(random_state=seed)
    loss_fn = nn.BCEWithLogitsLoss()

    methods = [
        {
            "key": "zeros",
            "name": "Zero Initialization",
            "method": "zeros",
            "note": "Weights and biases set to 0. Symmetry breaking fails across hidden units; hidden units receive identical gradients."
        },
        {
            "key": "random_normal",
            "name": "Random Normal (σ=0.5)",
            "method": "random_normal",
            "note": "Gaussian sampling with standard variance. Breaks symmetry, but variance is unscaled by layer fan-in."
        },
        {
            "key": "xavier_uniform",
            "name": "Xavier / Glorot Uniform",
            "method": "xavier_uniform",
            "note": "Variance scaled by 2 / (fan_in + fan_out). Designed to preserve activation variance across linear layers."
        },
        {
            "key": "he_normal",
            "name": "He / Kaiming Normal",
            "method": "he_normal",
            "note": "Variance scaled by 2 / fan_in specifically for ReLU non-linearities, countering dead activation collapse."
        }
    ]

    results = {}
    for item in methods:
        torch.manual_seed(seed)
        model = SimpleMLP()
        model.initialize_weights(item["method"], seed=seed)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        train_losses = []
        val_losses = []
        val_accs = []

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()

            tr_m = compute_metrics(model, X_train, y_train, loss_fn)
            va_m = compute_metrics(model, X_val, y_val, loss_fn)
            train_losses.append(tr_m["loss"])
            val_losses.append(va_m["loss"])
            val_accs.append(va_m["accuracy"])

        results[item["key"]] = {
            "name": item["name"],
            "note": item["note"],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accs,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_val_acc": val_accs[-1]
        }

    return results


def run_regularization_suite(seed: int = 42, epochs: int = 120, lr: float = 0.10) -> Dict[str, Any]:
    """
    Suite 4: Overfitting and L2 Regularization.
    Demonstrates generalization gap on small sample size (N_train=60) with and without weight decay.
    """
    X_train, y_train, X_val, y_val = generate_overfitting_data(n_total=500, n_train=60, noise=0.30, random_state=seed)
    loss_fn = nn.BCEWithLogitsLoss()

    torch.manual_seed(seed)
    base_model = SimpleMLP()
    base_model.initialize_weights("he_normal", seed=seed)
    initial_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    reg_configs = [
        {
            "key": "unregularized",
            "name": "Unregularized (Weight Decay = 0.0)",
            "weight_decay": 0.0,
            "description": "Model overfits small sample noise. Training loss falls near zero while validation loss diverges."
        },
        {
            "key": "l2_regularized",
            "name": "L2 Regularized (Weight Decay = 0.04)",
            "weight_decay": 0.04,
            "description": "L2 penalty (lambda/2 * ||W||^2) constrains weight magnitude, narrowing the generalization gap."
        }
    ]

    results = {}
    for cfg in reg_configs:
        model = SimpleMLP()
        model.load_state_dict({k: v.clone() for k, v in initial_state.items()})
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=cfg["weight_decay"])

        train_losses = []
        val_losses = []
        gen_gaps = []

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()

            tr_m = compute_metrics(model, X_train, y_train, loss_fn)
            va_m = compute_metrics(model, X_val, y_val, loss_fn)
            gap = float(va_m["loss"] - tr_m["loss"])

            train_losses.append(tr_m["loss"])
            val_losses.append(va_m["loss"])
            gen_gaps.append(gap)

        results[cfg["key"]] = {
            "name": cfg["name"],
            "weight_decay": cfg["weight_decay"],
            "description": cfg["description"],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "generalization_gaps": gen_gaps,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_generalization_gap": gen_gaps[-1]
        }

    return results


def run_all_benchmarks(output_path: Path) -> Dict[str, Any]:
    """
    Executes all 4 suites and compiles the canonical benchmark JSON file.
    """
    print("Executing Suite 1: Optimizer Convergence...")
    suite_opt = run_optimizer_suite()

    print("Executing Suite 2: Learning Rate Dynamics...")
    suite_lr = run_learning_rate_suite()

    print("Executing Suite 3: Weight Initializations...")
    suite_init = run_initialization_suite()

    print("Executing Suite 4: Overfitting & L2 Regularization...")
    suite_reg = run_regularization_suite()

    mlp_instance = SimpleMLP()
    total_params, layer_breakdown = mlp_instance.count_parameters()

    benchmark_data = {
        "metadata": {
            "module": "Foundational Deep Learning Experiments & Optimizer Comparison",
            "purpose": "Academic analysis of optimization dynamics, gradient backpropagation, and generalization.",
            "disclaimer": "Synthetic pedagogical benchmark module for Deep Learning study. Strictly independent from the clinical ISBI 2016 dermoscopy triage ResNet-18 and U-Net models.",
            "random_seed": 42,
            "device": "CPU",
            "pytorch_version": torch.__version__,
            "total_parameters": total_params,
            "parameter_breakdown": {k: {"shape": list(v[0]), "count": v[1]} for k, v in layer_breakdown.items()},
            "loss_function": "BCEWithLogitsLoss"
        },
        "suite_optimizers": suite_opt,
        "suite_learning_rates": suite_lr,
        "suite_initializations": suite_init,
        "suite_regularization": suite_reg
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"Benchmarks successfully written to {output_path}")
    return benchmark_data


if __name__ == "__main__":
    out_file = ROOT_DIR / "results" / "experiments" / "benchmarks.json"
    run_all_benchmarks(out_file)
