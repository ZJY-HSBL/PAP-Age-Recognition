from __future__ import annotations
import argparse
from pathlib import Path
from pap_age.data.metadata import load_metadata, split_by_identity


def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    p.add_argument("--train-ratio",type=float,default=0.8); p.add_argument("--val-ratio",type=float,default=0.0); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); df=load_metadata(a.input); out=split_by_identity(df,a.train_ratio,a.val_ratio,a.seed)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False)
    print(out.groupby("split")["person_id"].nunique())

if __name__=="__main__": main()
