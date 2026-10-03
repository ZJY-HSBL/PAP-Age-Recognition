from __future__ import annotations

"""Generate ablation configurations for controlled experiments.

Variants:
- full: PCA + LSTM + attention
- no_attention: PCA + LSTM, attention context effectively disabled by model edit in research branch
- no_pca: feed raw CNN features directly to sequence model (requires a custom run)
- cnn_only: evaluate the image-level CNN expected-age output

This file intentionally generates configs and experiment labels rather than claiming
unrun results. Use it as the experiment registry for your own measurements.
"""

import argparse, copy
from pathlib import Path
import yaml


def main():
    p=argparse.ArgumentParser(); p.add_argument("--base",required=True); p.add_argument("--output",default="configs/ablations")
    a=p.parse_args(); cfg=yaml.safe_load(Path(a.base).read_text(encoding="utf-8")); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    variants={"full":{},"bidirectional":{"sequence":{"bidirectional":True}},"noncausal_attention":{"sequence":{"causal_attention":False}}}
    for name,patch in variants.items():
        x=copy.deepcopy(cfg); x["project"]["name"] += f"_{name}"; x["project"]["output_dir"] += f"_{name}"
        for section,vals in patch.items(): x.setdefault(section,{}).update(vals)
        (out/f"{name}.yaml").write_text(yaml.safe_dump(x,sort_keys=False,allow_unicode=True),encoding="utf-8")
    print(f"Wrote {len(variants)} configs to {out}")

if __name__=="__main__": main()
