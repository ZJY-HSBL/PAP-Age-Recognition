from __future__ import annotations
import argparse,re
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(); p.add_argument("--images",required=True); p.add_argument("--output",required=True)
    p.add_argument("--regex",default=r"(?P<person>\d+)A(?P<age>\d+)")
    a=p.parse_args(); root=Path(a.images); rx=re.compile(a.regex,re.I); rows=[]
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".jpg",".jpeg",".png",".bmp"}: continue
        m=rx.search(path.stem)
        if m: rows.append({"path":str(path.resolve()),"person_id":m.group("person"),"age":int(m.group("age"))})
    if not rows: raise RuntimeError("No filenames matched. Adjust --regex.")
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False); print(f"Saved {len(rows)} rows")

if __name__=="__main__": main()
