from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


def _square_crop(im, box=None):
    w,h=im.size
    if box is None:
        side=min(w,h); left=(w-side)//2; top=(h-side)//2; return im.crop((left,top,left+side,top+side))
    x1,y1,x2,y2=map(float,box); cx=(x1+x2)/2; cy=(y1+y2)/2; side=max(x2-x1,y2-y1)*1.2
    x1=max(0,int(cx-side/2)); y1=max(0,int(cy-side/2)); x2=min(w,int(cx+side/2)); y2=min(h,int(cy+side/2)); return im.crop((x1,y1,x2,y2))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--metadata",required=True); p.add_argument("--output-root",required=True); p.add_argument("--output-metadata",required=True)
    p.add_argument("--detector",choices=["center","mtcnn","boxes"],default="center"); p.add_argument("--boxes"); p.add_argument("--size",type=int,default=299)
    a=p.parse_args(); df=pd.read_csv(a.metadata); outroot=Path(a.output_root); outroot.mkdir(parents=True,exist_ok=True)
    boxes={}
    if a.detector=="boxes":
        if not a.boxes: raise ValueError("--boxes is required for detector=boxes")
        bdf=pd.read_csv(a.boxes)
        for _,r in bdf.iterrows(): boxes[str(Path(r.path))]=(r.x1,r.y1,r.x2,r.y2)
    mtcnn=None
    if a.detector=="mtcnn":
        try:
            from facenet_pytorch import MTCNN
        except ImportError as e: raise ImportError("pip install -e '.[face]' for mtcnn mode") from e
        mtcnn=MTCNN(image_size=a.size,margin=20,post_process=False,select_largest=True)
    rows=[]
    for i,r in tqdm(df.iterrows(),total=len(df)):
        src=Path(r.path)
        with Image.open(src) as im:
            im=im.convert("RGB")
            if mtcnn is not None:
                face=mtcnn(im)
                if face is None: face=_square_crop(im).resize((a.size,a.size),Image.Resampling.LANCZOS)
                else:
                    import numpy as np
                    arr=face.permute(1,2,0).byte().cpu().numpy(); face=Image.fromarray(arr)
            else:
                box=boxes.get(str(src)) if a.detector=="boxes" else None
                if a.detector=="boxes" and box is None: raise KeyError(f"No box for {src}")
                face=_square_crop(im,box).resize((a.size,a.size),Image.Resampling.LANCZOS)
        dst=outroot/f"{i:07d}_{src.stem}.jpg"; face.save(dst,quality=95)
        row=r.to_dict(); row["path"]=str(dst.resolve()); rows.append(row)
    Path(a.output_metadata).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output_metadata,index=False)
    print(f"Saved {len(rows)} aligned images")

if __name__=="__main__": main()
