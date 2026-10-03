from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import PCA


def fit_pca(train_features: np.ndarray, variance: float = 0.95) -> PCA:
    pca = PCA(n_components=variance, svd_solver="full")
    pca.fit(train_features)
    return pca


def save_pca(pca: PCA, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, path)


def load_pca(path: str | Path) -> PCA:
    return joblib.load(path)
