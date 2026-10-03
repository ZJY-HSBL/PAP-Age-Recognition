from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import load_config
from .models.cnn import InceptionAgeModel
from .models.pap_lstm import PAPLSTM
from .pca import load_pca
from .utils import get_device


def predict_sequence(config_path: str, image_paths: list[str]):
    if not image_paths:
        raise ValueError("At least one image is required")
    cfg=load_config(config_path); base=Path(cfg["project"]["output_dir"]); device=get_device()
    age_min=int(cfg["data"].get("age_min",0)); age_max=int(cfg["data"].get("age_max",100))
    cnn=InceptionAgeModel(cfg["cnn"].get("backbone","inception_v4"),age_max-age_min+1,False,float(cfg["cnn"].get("dropout",0.8)))
    cnn.load_state_dict(torch.load(base/"cnn"/"best.pt",map_location="cpu",weights_only=False)["model"]); cnn.to(device).eval()
    size=int(cfg["data"]["image_size"])
    tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    xs=[]
    with torch.no_grad():
        for p in image_paths:
            with Image.open(p) as im: x=tf(im.convert("RGB")).unsqueeze(0).to(device)
            xs.append(cnn.extract_features(x).cpu().numpy()[0])
    pca=load_pca(base/"pca"/"pca.joblib"); feat=pca.transform(np.asarray(xs,np.float32)).astype(np.float32)
    ckpt=torch.load(base/"sequence"/"best.pt",map_location="cpu",weights_only=False)
    seq=PAPLSTM(int(ckpt["input_size"]),int(cfg["sequence"].get("hidden_size",256)),int(cfg["sequence"].get("num_layers",1)),int(cfg["sequence"].get("attention_dim",128)),bool(cfg["sequence"].get("bidirectional",False)),bool(cfg["sequence"].get("causal_attention",True)))
    seq.load_state_dict(ckpt["model"]); seq.to(device).eval()
    x=torch.from_numpy(feat).unsqueeze(0).to(device); lengths=torch.tensor([len(image_paths)]); mask=torch.ones((1,len(image_paths)),dtype=torch.bool,device=device)
    with torch.no_grad(): pred,att=seq(x,lengths,mask)
    return {"predicted_ages":pred[0].cpu().tolist(),"attention":att[0].cpu().tolist()}
