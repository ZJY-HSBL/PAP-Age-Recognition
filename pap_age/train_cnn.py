from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data.metadata import load_metadata
from .data.image_dataset import ImageAgeDataset
from .models.cnn import InceptionAgeModel, expected_age_from_logits
from .utils import get_device, seed_everything


def _optimizer(name, params, lr, weight_decay):
    if name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def train_cnn(config_path: str):
    cfg = load_config(config_path)
    seed_everything(int(cfg["project"]["seed"]))
    out = Path(cfg["project"]["output_dir"]) / "cnn"
    out.mkdir(parents=True, exist_ok=True)
    df = load_metadata(cfg["data"]["metadata"])
    if "split" not in df:
        raise ValueError("Metadata needs a split column. Run tools/split_metadata.py first.")
    train_df = df[df.split == "train"]
    val_df = df[df.split.isin(["val", "test"])]
    if val_df.empty:
        val_df = train_df.sample(min(len(train_df), max(1, len(train_df)//10)), random_state=0)

    image_size = int(cfg["data"]["image_size"])
    train_ds = ImageAgeDataset(train_df, image_size=image_size, train=True)
    val_ds = ImageAgeDataset(val_df, image_size=image_size, train=False)
    bs = int(cfg["cnn"]["batch_size"])
    nw = int(cfg["data"].get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    age_min = int(cfg["data"].get("age_min", 0))
    age_max = int(cfg["data"].get("age_max", 100))
    n_classes = age_max - age_min + 1
    model = InceptionAgeModel(
        backbone=cfg["cnn"].get("backbone", "inception_v4"),
        num_classes=n_classes,
        pretrained=bool(cfg["cnn"].get("pretrained", True)),
        dropout=float(cfg["cnn"].get("dropout", 0.8)),
    )
    device = get_device()
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["cnn"].get("label_smoothing", 0.0)))
    opt = _optimizer(
        cfg["cnn"].get("optimizer", "adam"), model.parameters(),
        float(cfg["cnn"]["learning_rate"]), float(cfg["cnn"].get("weight_decay", 0.0))
    )

    best_mae = float("inf")
    history = []
    for epoch in range(1, int(cfg["cnn"]["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"CNN {epoch}", leave=False):
            x = batch["image"].to(device)
            target = (batch["age"].round().long() - age_min).clamp(0, n_classes - 1).to(device)
            logits = model(x)
            loss = criterion(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * len(x)

        model.eval()
        abs_errors = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                age = batch["age"].to(device)
                pred = expected_age_from_logits(model(x), age_min=age_min)
                abs_errors.extend(torch.abs(pred - age).cpu().tolist())
        val_mae = sum(abs_errors) / max(1, len(abs_errors))
        record = {"epoch": epoch, "train_loss": train_loss/max(1,len(train_ds)), "val_mae": val_mae}
        history.append(record)
        print(json.dumps(record))
        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_mae": val_mae}
        torch.save(ckpt, out / "last.pt")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(ckpt, out / "best.pt")

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    return out / "best.pt"
