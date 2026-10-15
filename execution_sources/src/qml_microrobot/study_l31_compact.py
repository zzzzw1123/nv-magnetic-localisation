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
        return self.network(features)


@dataclass
class CompactMlpState:
    model: CompactMLP
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray

    @property
    def fitted_parameters(self) -> int:
        return sum(value.numel() for value in self.model.parameters())

    def serialized_bytes(self) -> int:
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
    state.model.eval()
    x = torch.as_tensor(
        (np.asarray(features) - state.feature_mean) / state.feature_scale,
        dtype=torch.float32,
    )
    scaled = state.model(x).cpu().numpy().astype(float)
    return scaled * state.target_scale + state.target_mean


def fitted_parameter_count(candidate_id: str) -> int:
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
    return len(pickle.dumps(model, protocol=5))
