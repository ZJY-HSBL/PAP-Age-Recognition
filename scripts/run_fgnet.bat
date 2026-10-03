@echo off
python -m pap_age.cli train-cnn --config configs/fgnet.yaml || exit /b 1
python -m pap_age.cli extract-features --config configs/fgnet.yaml || exit /b 1
python -m pap_age.cli fit-pca --config configs/fgnet.yaml || exit /b 1
python -m pap_age.cli train-sequence --config configs/fgnet.yaml || exit /b 1
python -m pap_age.cli evaluate --config configs/fgnet.yaml || exit /b 1
