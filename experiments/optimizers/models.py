"""
SimpleMLP architecture and initialization methods for foundational
Deep Learning optimization and training dynamics experiments.

Architectural Specification:
- Input: 2 dimensions (synthetic non-linear feature space)
- Hidden Layer 1: Linear(2 -> 32) + ReLU
- Hidden Layer 2: Linear(32 -> 16) + ReLU
- Output Layer: Linear(16 -> 1) producing raw unnormalized logits
- Loss: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- Trainable Parameter Count: Exactly 641 parameters
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SimpleMLP(nn.Module):
    """
    Compact Multi-Layer Perceptron designed for transparent pedagogical explanation
    of gradient propagation, weight initialization, and optimizer mechanics.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16, bias=True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing raw scalar logits.
        Sigmoid activation is deliberately omitted here to allow numerically
        stable loss calculation via BCEWithLogitsLoss.
        """
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        logits = self.fc3(h2)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies logistic sigmoid to raw logits for probability evaluation.
        """
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def count_parameters(self) -> Tuple[int, Dict[str, Tuple[Tuple[int, ...], int]]]:
        """
        Calculates and verifies the exact trainable parameter count.

        Layer breakdown:
        - fc1: 2 inputs * 32 outputs + 32 biases = 96
        - fc2: 32 inputs * 16 outputs + 16 biases = 528
        - fc3: 16 inputs * 1 output + 1 bias = 17
        Total: 96 + 528 + 17 = 641
        """
        layer_stats = {}
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num = param.numel()
                total += num
                layer_stats[name] = (tuple(param.shape), num)
        return total, layer_stats

    def initialize_weights(self, method: str, std: float = 0.5, seed: Optional[int] = None) -> None:
        """
        Applies standard weight initialization schemes to all linear layers.

        Parameters:
            method: 'zeros', 'random_normal', 'xavier_uniform', or 'he_normal'.
            std: Standard deviation for random normal initialization.
            seed: Optional integer seed for reproducibility.

        Symmetry Breaking Note for Zero Initialization:
        When all weights are set to 0.0 and biases to 0.0, every hidden neuron receives
        an identical net input z=0 and produces an identical activation a=0. During
        backpropagation, every incoming weight in a hidden layer receives an identical
        gradient. Even if output biases shift to match the class prior (which may lower
        loss from -ln(0.5) to the dataset entropy), the hidden neurons never break
        symmetry and cannot learn diverse feature detectors.
        """
        if seed is not None:
            torch.manual_seed(seed)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == "zeros":
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)

                elif method == "random_normal":
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    nn.init.zeros_(module.bias)

                elif method == "xavier_uniform":
                    # Glorot uniform: useful for symmetric activations (tanh, sigmoid)
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

                elif method == "he_normal":
                    # Kaiming normal: variance adjusted specifically for ReLU non-linearities
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    nn.init.zeros_(module.bias)

                else:
                    raise ValueError(f"Unsupported initialization method: {method}")


def generate_benchmark_data(
    n_samples: int = 1000,
    noise: float = 0.20,
    test_size: float = 0.30,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a deterministic 2D non-linear classification dataset (make_moons),
    standardizes features to zero mean and unit variance, and returns PyTorch tensors.
    """
    X_raw, y_raw = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_raw, y_raw, test_size=test_size, random_state=random_state, stratify=y_raw
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val = torch.tensor(y_val_raw, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_val, y_val


def generate_overfitting_data(
    n_total: int = 500,
    n_train: int = 60,
    noise: float = 0.30,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a small training set (e.g. 60 samples with noise) alongside a larger
    validation set (440 samples) to clearly demonstrate overfitting and the regularizing
    effect of L2 weight decay.
    """
    X_raw, y_raw = make_moons(n_samples=n_total, noise=noise, random_state=random_state)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_raw)

    X_train = torch.tensor(X_norm[:n_train], dtype=torch.float32)
    y_train = torch.tensor(y_raw[:n_train], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_norm[n_train:], dtype=torch.float32)
    y_val = torch.tensor(y_raw[n_train:], dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_val, y_val
