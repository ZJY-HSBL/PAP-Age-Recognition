from __future__ import annotations

from pathlib import Path
import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .config import load_config
from .data.sequence_dataset import FeatureSequenceDataset, collate_sequences
from .models.pap_lstm import PAPLSTM
from .metrics import mae, cumulative_scores
from .utils import get_device, save_json


def evaluate(config_path: str):
    cfg = load_config(config_path)
    base=Path(cfg["project"]["output_dir"])
    z=np.load(base/"pca"/"features_pca.npz",allow_pickle=False)
    split="test" if np.any(z["splits"].astype(str)=="test") else "val"
    ds=FeatureSequenceDataset(z["features"],z["ages"],z["person_ids"],z["splits"],split,
                              int(cfg["sequence"].get("min_sequence_length",2)))
    loader=DataLoader(ds,batch_size=int(cfg["sequence"]["batch_size"]),shuffle=False,collate_fn=collate_sequences)
    ckpt=torch.load(base/"sequence"/"best.pt",map_location="cpu",weights_only=False)
    model=PAPLSTM(
        input_size=int(ckpt["input_size"]), hidden_size=int(cfg["sequence"].get("hidden_size",256)),
        num_layers=int(cfg["sequence"].get("num_layers",1)), attention_dim=int(cfg["sequence"].get("attention_dim",128)),
        bidirectional=bool(cfg["sequence"].get("bidirectional",False)), causal_attention=bool(cfg["sequence"].get("causal_attention",True)))
    model.load_state_dict(ckpt["model"]); device=get_device(); model.to(device); model.eval()
    y_true=[]; y_pred=[]; rows=[]
    with torch.no_grad():
        for batch in loader:
            x=batch["features"].to(device); mask=batch["mask"].to(device)
            pred,weights=model(x,batch["lengths"],mask)
            for i,person in enumerate(batch["person_ids"]):
                n=int(batch["lengths"][i])
                for t in range(n):
                    yt=float(batch["ages"][i,t]); yp=float(pred[i,t].cpu())
                    y_true.append(yt); y_pred.append(yp)
                    rows.append({"person_id":person,"time_index":t,"age":yt,"prediction":yp,"abs_error":abs(yt-yp)})
    max_err=int(cfg["evaluation"].get("max_cs_error",10))
    m=mae(y_true,y_pred); cs=cumulative_scores(y_true,y_pred,max_err)
    out=base/"evaluation"; out.mkdir(parents=True,exist_ok=True)
    save_json({"split":split,"n_predictions":len(y_true),"mae":m,"cumulative_score":{str(k):v for k,v in cs.items()}},out/"metrics.json")
    with (out/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["person_id","time_index","age","prediction","abs_error"]); w.writeheader(); w.writerows(rows)
    plt.figure(figsize=(6.4,4.8)); plt.plot(list(cs.keys()),list(cs.values()),marker="o")
    plt.xlabel("Absolute age error threshold (years)"); plt.ylabel("Cumulative score (%)"); plt.ylim(0,100); plt.grid(True,alpha=.3); plt.tight_layout(); plt.savefig(out/"cs_curve.png",dpi=200); plt.close()
    print(json.dumps({"mae":m,"cs":cs},indent=2))
    return out/"metrics.json"
