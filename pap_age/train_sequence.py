from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data.sequence_dataset import FeatureSequenceDataset, collate_sequences
from .models.pap_lstm import PAPLSTM
from .utils import get_device, seed_everything


def _masked_loss(pred, target, mask, kind="mae"):
    diff = pred[mask] - target[mask]
    if kind == "mse":
        return (diff ** 2).mean()
    if kind == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(pred[mask], target[mask])
    return diff.abs().mean()


def train_sequence(config_path: str):
    cfg = load_config(config_path)
    seed_everything(int(cfg["project"]["seed"]))
    base = Path(cfg["project"]["output_dir"])
    z = np.load(base/"pca"/"features_pca.npz", allow_pickle=False)
    kw = dict(features=z["features"], ages=z["ages"], person_ids=z["person_ids"], splits=z["splits"],
              min_sequence_length=int(cfg["sequence"].get("min_sequence_length", 2)))
    train_ds = FeatureSequenceDataset(split="train", **kw)
    eval_split = "val" if np.any(z["splits"].astype(str)=="val") else "test"
    val_ds = FeatureSequenceDataset(split=eval_split, **kw)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("No valid sequences. Check identity split and min_sequence_length.")
    bs = int(cfg["sequence"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_sequences)

    model = PAPLSTM(
        input_size=z["features"].shape[1],
        hidden_size=int(cfg["sequence"].get("hidden_size", 256)),
        num_layers=int(cfg["sequence"].get("num_layers", 1)),
        attention_dim=int(cfg["sequence"].get("attention_dim", 128)),
        bidirectional=bool(cfg["sequence"].get("bidirectional", False)),
        causal_attention=bool(cfg["sequence"].get("causal_attention", True)),
    )
    device = get_device(); model.to(device)
    if cfg["sequence"].get("optimizer", "adam").lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=float(cfg["sequence"]["learning_rate"]), momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["sequence"]["learning_rate"]),
                               weight_decay=float(cfg["sequence"].get("weight_decay", 0.0)))
    out = base/"sequence"; out.mkdir(parents=True, exist_ok=True)
    best = float("inf"); history=[]
    for epoch in range(1, int(cfg["sequence"]["epochs"])+1):
        model.train(); total=0.0; n=0
        for batch in tqdm(train_loader, desc=f"SEQ {epoch}", leave=False):
            x=batch["features"].to(device); y=batch["ages"].to(device); mask=batch["mask"].to(device)
            pred,_=model(x, batch["lengths"], mask)
            loss=_masked_loss(pred,y,mask,cfg["sequence"].get("loss","mae"))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += float(loss.item())*int(mask.sum().item()); n += int(mask.sum().item())
        model.eval(); errors=[]
        with torch.no_grad():
            for batch in val_loader:
                x=batch["features"].to(device); y=batch["ages"].to(device); mask=batch["mask"].to(device)
                pred,_=model(x,batch["lengths"],mask)
                errors.extend(torch.abs(pred[mask]-y[mask]).cpu().tolist())
        val_mae=sum(errors)/max(1,len(errors))
        rec={"epoch":epoch,"train_loss":total/max(1,n),"val_mae":val_mae}; history.append(rec)
        print(json.dumps(rec))
        ckpt={"model":model.state_dict(),"config":cfg,"input_size":z["features"].shape[1],"epoch":epoch,"val_mae":val_mae}
        torch.save(ckpt,out/"last.pt")
        if val_mae < best:
            best=val_mae; torch.save(ckpt,out/"best.pt")
    pd.DataFrame(history).to_csv(out/"history.csv",index=False)
    return out/"best.pt"
