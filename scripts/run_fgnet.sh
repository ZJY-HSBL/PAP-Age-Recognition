#!/usr/bin/env bash
set -euo pipefail
python -m pap_age.cli train-cnn --config configs/fgnet.yaml
python -m pap_age.cli extract-features --config configs/fgnet.yaml
python -m pap_age.cli fit-pca --config configs/fgnet.yaml
python -m pap_age.cli train-sequence --config configs/fgnet.yaml
python -m pap_age.cli evaluate --config configs/fgnet.yaml
