from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--images-root",default="")
    p.add_argument("--path-col",required=True); p.add_argument("--person-col",required=True); p.add_argument("--age-col",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(); src=pd.read_csv(a.source); root=Path(a.images_root) if a.images_root else None
    rows=[]
    for _,r in src.iterrows():
        path=Path(str(r[a.path_col])); path=(root/path if root and not path.is_absolute() else path)
        rows.append({"path":str(path.resolve()),"person_id":str(r[a.person_col]),"age":float(r[a.age_col])})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False); print(f"Saved {len(rows)} rows")

if __name__=="__main__": main()
