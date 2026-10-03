from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data.metadata import load_metadata
from .data.image_dataset import ImageAgeDataset
from .models.cnn import InceptionAgeModel
from .utils import get_device, seed_everything


def extract_features(config_path: str):
    cfg = load_config(config_path)
    seed_everything(int(cfg["project"]["seed"]))
    out = Path(cfg["project"]["output_dir"]) / "features"
    out.mkdir(parents=True, exist_ok=True)
    df = load_metadata(cfg["data"]["metadata"])
    if "split" not in df:
        raise ValueError("Metadata needs split column")
    ds = ImageAgeDataset(df, image_size=int(cfg["data"]["image_size"]), train=False)
    loader = DataLoader(ds, batch_size=max(1, int(cfg["cnn"]["batch_size"])), shuffle=False,
                        num_workers=int(cfg["data"].get("num_workers", 4)), pin_memory=True)
    age_min = int(cfg["data"].get("age_min", 0))
    age_max = int(cfg["data"].get("age_max", 100))
    model = InceptionAgeModel(
        backbone=cfg["cnn"].get("backbone", "inception_v4"),
        num_classes=age_max-age_min+1,
        pretrained=False,
        dropout=float(cfg["cnn"].get("dropout", 0.8)),
    )
    ckpt_path = Path(cfg["project"]["output_dir"]) / "cnn" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    device = get_device(); model.to(device); model.eval()

    feats=[]; ages=[]; people=[]; paths=[]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract"):
            f = model.extract_features(batch["image"].to(device)).cpu().numpy()
            feats.append(f)
            ages.extend(batch["age"].numpy().tolist())
            people.extend(batch["person_id"])
            paths.extend(batch["path"])
    features = np.concatenate(feats, axis=0)
    np.savez_compressed(
        out/"features.npz", features=features, ages=np.asarray(ages, np.float32),
        person_ids=np.asarray(people, dtype=str), paths=np.asarray(paths, dtype=str),
        splits=np.asarray(df["split"].astype(str).tolist(), dtype=str),
    )
    print(f"Saved {features.shape} to {out/'features.npz'}")
    return out/"features.npz"
