from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train_cnn import train_cnn
from .extract_features import extract_features
from .fit_pca import fit_pca_stage
from .train_sequence import train_sequence
from .evaluate import evaluate
from .infer import predict_sequence


def main():
    p=argparse.ArgumentParser(prog="pap-age")
    sub=p.add_subparsers(dest="cmd",required=True)
    for name in ["train-cnn","extract-features","fit-pca","train-sequence","evaluate"]:
        s=sub.add_parser(name); s.add_argument("--config",required=True)
    inf=sub.add_parser("infer"); inf.add_argument("--config",required=True); inf.add_argument("images",nargs="+")
    smoke=sub.add_parser("smoke-test"); smoke.add_argument("--data-root",default="data/toy")
    a=p.parse_args()
    if a.cmd=="train-cnn": train_cnn(a.config)
    elif a.cmd=="extract-features": extract_features(a.config)
    elif a.cmd=="fit-pca": fit_pca_stage(a.config)
    elif a.cmd=="train-sequence": train_sequence(a.config)
    elif a.cmd=="evaluate": evaluate(a.config)
    elif a.cmd=="infer": print(json.dumps(predict_sequence(a.config,a.images),indent=2))
    elif a.cmd=="smoke-test":
        root=Path(a.data_root); meta=root/"metadata_split.csv"
        if not meta.exists():
            raise FileNotFoundError(f"{meta} not found. Run scripts/make_toy_data.py first.")
        cfg=Path("configs/toy.yaml")
        train_cnn(str(cfg)); extract_features(str(cfg)); fit_pca_stage(str(cfg)); train_sequence(str(cfg)); evaluate(str(cfg))

if __name__=="__main__": main()
