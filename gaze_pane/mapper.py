"""Affine fit from 4-dim gaze features -> 2-dim normalized screen coords."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .gaze import GazeFeatures


class GazeMapper:
    """Two-output least-squares affine map. 5 unknowns per axis (4 features + bias)."""

    def __init__(self) -> None:
        self.wx: np.ndarray | None = None  # shape (5,)
        self.wy: np.ndarray | None = None
        self.residuals: tuple[float, float] | None = None

    def fit(self, samples: Sequence[tuple[float, float, GazeFeatures]]) -> None:
        """samples = [(norm_x, norm_y, features), ...]; norms in [0, 1] top-left origin."""
        if len(samples) < 5:
            raise ValueError(f"Need >=5 calibration samples, got {len(samples)}")
        X = np.array([list(s[2].as_vec()) + [1.0] for s in samples])  # (N, 5)
        tx = np.array([s[0] for s in samples])
        ty = np.array([s[1] for s in samples])
        self.wx, rx, *_ = np.linalg.lstsq(X, tx, rcond=None)
        self.wy, ry, *_ = np.linalg.lstsq(X, ty, rcond=None)
        # rx/ry may be empty if underdetermined; compute fallback residuals.
        pred_x = X @ self.wx
        pred_y = X @ self.wy
        self.residuals = (
            float(np.sqrt(np.mean((pred_x - tx) ** 2))),
            float(np.sqrt(np.mean((pred_y - ty) ** 2))),
        )

    def predict(self, features: GazeFeatures) -> tuple[float, float]:
        if self.wx is None or self.wy is None:
            raise RuntimeError("GazeMapper not fit/loaded yet")
        v = np.concatenate([features.as_vec(), [1.0]])
        return float(v @ self.wx), float(v @ self.wy)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.wx is None or self.wy is None:
            raise RuntimeError("nothing to save")
        path.write_text(json.dumps({
            "version": 1,
            "wx": self.wx.tolist(),
            "wy": self.wy.tolist(),
            "residuals": self.residuals,
        }, indent=2))

    def load(self, path: Path) -> None:
        d = json.loads(Path(path).read_text())
        self.wx = np.array(d["wx"], dtype=np.float64)
        self.wy = np.array(d["wy"], dtype=np.float64)
        if self.wx.shape != (5,) or self.wy.shape != (5,):
            raise ValueError(f"calibration shape mismatch: {self.wx.shape}, {self.wy.shape}")
        self.residuals = tuple(d.get("residuals") or (0.0, 0.0))


def default_calibration_path() -> Path:
    return Path.home() / ".config" / "gaze-pane" / "calibration.json"
