#!/usr/bin/env bash
set -euo pipefail
python -m pap_age.cli train-cnn --config configs/morph.yaml
python -m pap_age.cli extract-features --config configs/morph.yaml
python -m pap_age.cli fit-pca --config configs/morph.yaml
python -m pap_age.cli train-sequence --config configs/morph.yaml
python -m pap_age.cli evaluate --config configs/morph.yaml
