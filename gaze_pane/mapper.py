"""Affine fit from 4-dim gaze features -> 2-dim normalized screen coords."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .gaze import GazeFeatures


CALIBRATION_VERSION = 6   # added face_scale; validation/refit phase produces these files
RIDGE_LAMBDA = 0.1        # in normalized-feature space; weights stay O(1)


class GazeMapper:
    """Ridge-regularized affine map from gaze features to top-left normalized screen.

    Features are z-score normalized before the fit so a single λ behaves consistently
    across feature dims. Without ridge, near-collinear features (L vs R iris) blow
    up the weights and predict() extrapolates to absurd values.
    """

    def __init__(self) -> None:
        self.wx: np.ndarray | None = None
        self.wy: np.ndarray | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.residuals: tuple[float, float] | None = None
        self.feature_dim: int | None = None
        # Free-form metadata that travels with the JSON: started_at, finished_at,
        # duration_seconds, grid, n_points, samples_per_point.
        self.metadata: dict = {}

    def fit(self, samples: Sequence[tuple[float, float, GazeFeatures]]) -> None:
        if not samples:
            raise ValueError("no calibration samples")
        F = np.array([s[2].as_vec() for s in samples], dtype=np.float64)  # (N, d)
        d = F.shape[1]
        if len(samples) <= d:
            raise ValueError(
                f"need at least {d + 1} calibration samples (have {len(samples)}); "
                f"feature dim is {d}"
            )
        # z-normalize features. Replace std=0 with 1 so the column degenerates to zero
        # (still safe: ridge handles it).
        mean = F.mean(axis=0)
        std = F.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        F_n = (F - mean) / std
        X = np.hstack([F_n, np.ones((F.shape[0], 1))])  # (N, d+1) with bias
        tx = np.array([s[0] for s in samples])
        ty = np.array([s[1] for s in samples])

        # Ridge regression. Skip the bias column from the penalty.
        n = X.shape[1]
        I = np.eye(n)
        I[-1, -1] = 0.0
        A = X.T @ X + RIDGE_LAMBDA * I
        self.wx = np.linalg.solve(A, X.T @ tx)
        self.wy = np.linalg.solve(A, X.T @ ty)

        pred_x = X @ self.wx
        pred_y = X @ self.wy
        self.residuals = (
            float(np.sqrt(np.mean((pred_x - tx) ** 2))),
            float(np.sqrt(np.mean((pred_y - ty) ** 2))),
        )
        self.feature_dim = d
        self.feature_mean = mean
        self.feature_std = std

    def predict(self, features: GazeFeatures) -> tuple[float, float]:
        if self.wx is None or self.wy is None:
            raise RuntimeError("GazeMapper not fit/loaded yet")
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("normalization params missing; recalibrate")
        v = features.as_vec()
        v_n = (v - self.feature_mean) / self.feature_std
        v_aug = np.concatenate([v_n, [1.0]])
        return float(v_aug @ self.wx), float(v_aug @ self.wy)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.wx is None or self.wy is None or self.feature_mean is None:
            raise RuntimeError("nothing to save")
        path.write_text(json.dumps({
            "version": CALIBRATION_VERSION,
            "metadata": self.metadata,
            "feature_dim": self.feature_dim,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "wx": self.wx.tolist(),
            "wy": self.wy.tolist(),
            "residuals": self.residuals,
        }, indent=2))

    def load(self, path: Path) -> None:
        d = json.loads(Path(path).read_text())
        version = int(d.get("version", 1))
        if version < CALIBRATION_VERSION:
            raise ValueError(
                f"calibration is v{version} (now v{CALIBRATION_VERSION}). "
                "Re-run:  gaze-pane calibrate --camera <N>"
            )
        self.wx = np.array(d["wx"], dtype=np.float64)
        self.wy = np.array(d["wy"], dtype=np.float64)
        if self.wx.shape != self.wy.shape or self.wx.ndim != 1:
            raise ValueError(
                f"calibration shape mismatch: wx={self.wx.shape}, wy={self.wy.shape}"
            )
        self.feature_mean = np.array(d["feature_mean"], dtype=np.float64)
        self.feature_std = np.array(d["feature_std"], dtype=np.float64)
        self.feature_dim = int(d.get("feature_dim", self.feature_mean.shape[0]))
        self.residuals = tuple(d.get("residuals") or (0.0, 0.0))
        self.metadata = dict(d.get("metadata") or {})


def default_calibration_path() -> Path:
    return Path.home() / ".config" / "gaze-pane" / "calibration.json"
