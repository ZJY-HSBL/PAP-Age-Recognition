from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        raise ValueError("MAE requires at least one sample")
    return float(np.mean(np.abs(yt - yp)))


def cumulative_scores(y_true, y_pred, max_error: int = 10) -> dict[int, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        raise ValueError("CS requires at least one sample")
    err = np.abs(yt - yp)
    return {m: float(np.mean(err <= m) * 100.0) for m in range(max_error + 1)}
