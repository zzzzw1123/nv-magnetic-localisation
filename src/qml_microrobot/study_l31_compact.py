"""Compact nonlinear and classical candidate models for Study L localisation.

All predictors consume 24 fitted spectral features and return two coordinates
in micrometres.  Scaling is fitted from the supplied training partition only.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import pickle

import numpy as np
import torch
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch import nn

from qml_microrobot.study_l30_author_cnn import set_deterministic_seed


class CompactMLP(nn.Module):
    """One-hidden-layer ReLU regressor with configurable hidden width."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.network = nn.Sequential(
            nn.Linear(24, width), nn.ReLU(), nn.Linear(width, 2)
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map a ``[batch, 24]`` tensor to two standardised target coordinates."""

        return self.network(features)


@dataclass
class CompactMlpState:
    """Fitted network and training-partition scalers required for prediction."""

    model: CompactMLP
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray

    @property
    def fitted_parameters(self) -> int:
        """Return the number of fitted neural-network scalars."""

        return sum(value.numel() for value in self.model.parameters())

    def serialized_bytes(self) -> int:
        """Measure the complete Torch-serialised state, including four scalers."""

        buffer = io.BytesIO()
        torch.save(
            {
                "model": self.model.state_dict(),
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "target_mean": self.target_mean,
                "target_scale": self.target_scale,
            },
            buffer,
        )
        return len(buffer.getvalue())


def _mean_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return column means and population standard deviations, guarding zero scale."""

    mean = np.asarray(values, dtype=float).mean(axis=0)
    scale = np.asarray(values, dtype=float).std(axis=0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def fit_compact_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    width: int,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> CompactMlpState:
    """Fit a compact MLP with training-set input and target standardisation.

    Args:
        features: Training features with shape ``(n, 24)``.
        targets: Training coordinates with shape ``(n, 2)`` in micrometres.
        width: Number of hidden ReLU units.
        seed: Seed for model initialisation; batch order uses ``seed + 1``.
        epochs: Number of passes through the training rows.
        batch_size: Maximum rows in one optimiser update.
        learning_rate: Adam learning rate.

    Returns:
        A CompactMlpState containing the trained model and input/target
        scaling arrays for coordinate predictions in micrometres.

    Notes:
        The call resets Python, NumPy and PyTorch global random states to
        ``seed``, enables deterministic PyTorch algorithms, and sets the
        PyTorch intra-op thread count to one. These process-wide settings are
        not restored.
    """

    set_deterministic_seed(seed)
    torch.set_num_threads(1)
    feature_mean, feature_scale = _mean_scale(features)
    target_mean, target_scale = _mean_scale(targets)
    x = torch.as_tensor((features - feature_mean) / feature_scale, dtype=torch.float32)
    y = torch.as_tensor((targets - target_mean) / target_scale, dtype=torch.float32)
    model = CompactMLP(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x[batch]) - y[batch]) ** 2)
            loss.backward()
            optimizer.step()
    return CompactMlpState(
        model=model,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
    )


@torch.inference_mode()
def predict_compact_mlp(state: CompactMlpState, features: np.ndarray) -> np.ndarray:
    """Return micrometre-coordinate predictions for ``[sample, 24]`` feature rows."""

    state.model.eval()
    x = torch.as_tensor(
        (np.asarray(features) - state.feature_mean) / state.feature_scale,
        dtype=torch.float32,
    )
    scaled = state.model(x).cpu().numpy().astype(float)
    return scaled * state.target_scale + state.target_mean


def fitted_parameter_count(candidate_id: str) -> int:
    """Return the fitted-coefficient count for a configured candidate identifier."""

    if candidate_id.startswith("poly2"):
        return 650
    if candidate_id.startswith("rff64"):
        return 130
    if candidate_id.startswith("rff128"):
        return 258
    if candidate_id.startswith("mlp"):
        width = int(candidate_id.split("_", 1)[0][3:])
        return 24 * width + width + width * 2 + 2
    raise ValueError(candidate_id)


def fit_classical_candidate(candidate_id: str, features: np.ndarray, targets: np.ndarray):
    """Fit a configured polynomial-Ridge or random-Fourier-Ridge candidate."""

    parts = candidate_id.split("_")
    alpha = float(parts[-1][1:])
    if candidate_id.startswith("poly2"):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
    elif candidate_id.startswith("rff"):
        dimension = int(parts[0][3:])
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "rff",
                    RBFSampler(
                        gamma=1.0 / 24.0,
                        n_components=dimension,
                        random_state=20260804,
                    ),
                ),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
    else:
        raise ValueError(candidate_id)
    return model.fit(features, targets)


def classical_state_bytes(model) -> int:
    """Return the pickle-protocol-5 size of a fitted classical pipeline."""

    return len(pickle.dumps(model, protocol=5))
