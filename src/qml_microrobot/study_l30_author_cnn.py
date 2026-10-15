"""Reference CNN and deterministic training helpers for Study L localisation.

The model maps 24 fitted NV spectral features to two motor-stage coordinates
in micrometres. Training helpers preserve the source architecture and the
configured CPU recipe used by the Weng-data comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from qml_microrobot.study_l_session import NvSessionData, load_l1_session_data


@dataclass(frozen=True)
class AuthorSplit:
    """Integer row indices for the released 90/7/3 train/validation/test split."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


class AuthorLocalizationCNN(nn.Module):
    """PyTorch port of the public Keras network for 24-feature, 2-D localisation."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(18 * 256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.reset_keras_style_parameters()

    def reset_keras_style_parameters(self) -> None:
        """Apply Keras-compatible Xavier weights and zero biases."""

        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, 2]`` coordinate predictions from 24-feature rows."""

        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3 or features.shape[1:] != (1, 24):
            raise ValueError("expected features shaped [batch,24] or [batch,1,24]")
        return self.network(features)


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic Torch kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def trainable_parameters(model: nn.Module) -> int:
    """Count scalar parameters whose gradients are enabled."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def author_random_split(row_count: int) -> AuthorSplit:
    """Recreate the released random-state-42 split for ``row_count`` observations."""

    indices = np.arange(row_count)
    train, temporary = train_test_split(
        indices, test_size=0.1, random_state=42
    )
    validation, test = train_test_split(
        temporary, test_size=0.3, random_state=42
    )
    return AuthorSplit(
        train=np.asarray(train, dtype=int),
        validation=np.asarray(validation, dtype=int),
        test=np.asarray(test, dtype=int),
    )


def load_author_data(root: Path) -> NvSessionData:
    """Load 24-feature rows, micrometre targets and acquisition identifiers."""

    return load_l1_session_data(root)


def mean_euclidean_error_um(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute mean Euclidean error for paired ``[sample, 2]`` coordinates in micrometres."""

    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("truth and prediction must have identical [sample,2] shape")
    return float(np.linalg.norm(truth - prediction, axis=1).mean())


def fit_author_cnn(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int = 20260804,
    epochs: int = 3000,
    batch_size: int = 128,
    learning_rate: float = 0.005,
) -> tuple[AuthorLocalizationCNN, list[float]]:
    """Fit the reference CNN on selected rows and return a sparse loss trace.

    ``features`` and ``targets`` are shaped ``[sample, 24]`` and ``[sample, 2]``.
    The returned loss values are training MSE snapshots from the configured schedule;
    they are diagnostics rather than validation-based stopping decisions.
    """

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    set_deterministic_seed(seed)
    torch.set_num_threads(1)
    values = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    truth = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
    selected = torch.as_tensor(np.asarray(train_indices), dtype=torch.long)
    model = AuthorLocalizationCNN()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-7,
    )
    generator = torch.Generator().manual_seed(seed + 1)
    losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        order = selected[torch.randperm(len(selected), generator=generator)]
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(values[batch])
            loss = torch.mean((prediction - truth[batch]) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
        if epoch in {0, 9, 99, 299, 999, epochs - 1}:
            losses.append(epoch_loss / seen)
    return model, losses


@torch.inference_mode()
def predict_author_cnn(model: AuthorLocalizationCNN, features: np.ndarray) -> np.ndarray:
    """Predict ``[sample, 2]`` motor-stage coordinates from ``[sample, 24]`` features."""

    model.eval()
    values = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    return model(values).cpu().numpy().astype(float)
