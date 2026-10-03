from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .config import load_config
from .pca import fit_pca, save_pca


def fit_pca_stage(config_path: str):
    cfg = load_config(config_path)
    base = Path(cfg["project"]["output_dir"])
    src = np.load(base/"features"/"features.npz", allow_pickle=False)
    features = src["features"]
    splits = src["splits"].astype(str)
    pca = fit_pca(features[splits == "train"], float(cfg["pca"].get("variance", 0.95)))
    out = base/"pca"; out.mkdir(parents=True, exist_ok=True)
    save_pca(pca, out/"pca.joblib")
    transformed = pca.transform(features).astype(np.float32)
    np.savez_compressed(
        out/"features_pca.npz", features=transformed, ages=src["ages"],
        person_ids=src["person_ids"], paths=src["paths"], splits=src["splits"]
    )
    info = {
        "input_dimension": int(features.shape[1]),
        "output_dimension": int(transformed.shape[1]),
        "retained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "expected_dimension": cfg["pca"].get("expected_dimension"),
    }
    (out/"pca_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return out/"features_pca.npz"
