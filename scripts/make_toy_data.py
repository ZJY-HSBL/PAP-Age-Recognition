from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pap_age.data.metadata import split_by_identity


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="data/toy"); p.add_argument("--people",type=int,default=12); p.add_argument("--images-per-person",type=int,default=4); p.add_argument("--seed",type=int,default=7)
    a=p.parse_args(); rng=np.random.default_rng(a.seed); root=Path(a.output); imgdir=root/"images"; imgdir.mkdir(parents=True,exist_ok=True); rows=[]
    for person in range(a.people):
        base_age=int(rng.integers(8,45))
        for t in range(a.images_per_person):
            age=base_age+t*5; arr=np.zeros((64,64,3),dtype=np.uint8); arr[:]=np.clip(40+age*2,0,255)
            im=Image.fromarray(arr); d=ImageDraw.Draw(im); d.ellipse((14,8,50,54),outline=(255,255,255),width=2); d.text((4,54),f"{person}:{age}",fill=(255,255,255))
            path=imgdir/f"p{person:03d}_a{age:02d}.png"; im.save(path); rows.append({"path":str(path.resolve()),"person_id":f"p{person:03d}","age":age})
    df=pd.DataFrame(rows); df=split_by_identity(df,0.8,0.0,a.seed); root.mkdir(parents=True,exist_ok=True); df.to_csv(root/"metadata_split.csv",index=False); print(root/"metadata_split.csv")

if __name__=="__main__": main()
