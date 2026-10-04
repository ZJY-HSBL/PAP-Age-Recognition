# PAP-Age Recognition | 个体老化模式人脸年龄识别

[English](#english) · [中文](#中文)

## English

PAP-Age is a complete research codebase for **personalized aging-pattern facial age estimation**. The system models age not only from the static appearance of a single face, but also from the chronological facial trajectory of the same person.

The pipeline is organized as:

```text
Face images
   ↓
Face detection / alignment / crop / normalization
   ↓
Inception-v4 CNN
   ↓ 1536-D GAP feature
PCA manifold projection (95% retained variance)
   ↓ compact age-discriminative feature
Chronological sequence by identity
   ↓
LSTM temporal modeling
   ↓
Temporal attention
   ↓
Age prediction at each valid time step
```

The repository includes dataset preparation, deterministic identity-level splitting, CNN training, 1536-D feature extraction, PCA fitting, chronological sequence construction, LSTM-attention training, MAE/CS evaluation, inference, ablation experiments, plots, configuration files for MORPH and FG-NET, unit tests, and CI.

### Core design

- **CNN appearance encoder:** Inception-v4 extracts a 1536-dimensional global-average-pooled facial representation.
- **Manifold projection:** PCA keeps 95% of the training-set feature variance and removes redundant dimensions.
- **Personalized temporal modeling:** images are grouped by identity and sorted by age to form variable-length aging sequences.
- **Attention-enhanced LSTM:** the recurrent model learns age progression while attention reweights informative historical states.
- **Evaluation:** Mean Absolute Error (MAE) and Cumulative Score (CS) from error thresholds 0–10 years are implemented directly.

### Protocol configurations

The two supplied protocol files mirror the study settings:

| Dataset | CNN LR | CNN epochs | CNN batch | CNN dropout | LSTM LR | LSTM epochs | LSTM batch | PCA variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MORPH | 0.001 | 100 | 80 | 0.8 | 0.001 | 300 | 20 | 0.95 |
| FG-NET | 0.001 | 100 | 2 | 0.8 | 0.001 | 300 | 1 | 0.95 |

With the original dataset distributions, the 1536-D Inception-v4 representation is expected to reduce to a dataset-dependent PCA dimension. The method specification reports 111 dimensions for MORPH and 171 for FG-NET at 95% retained variance; the exact dimension can vary if preprocessing, split, or samples differ.

### Installation

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional MTCNN preprocessing:

```bash
pip install -e ".[face]"
```

### Dataset metadata

The project does not redistribute MORPH or FG-NET images. Prepare a CSV with:

```csv
path,person_id,age
/path/to/image1.jpg,0001,16
/path/to/image2.jpg,0001,21
/path/to/image3.jpg,0002,35
```

For FG-NET filename-style data, use:

```bash
python tools/prepare_fgnet.py --images /data/FGNET/images --output data/fgnet.csv
```

For MORPH or another dataset that ships metadata:

```bash
python tools/prepare_from_csv.py \
  --source /data/morph/metadata.csv \
  --images-root /data/morph/images \
  --path-col file --person-col subject_id --age-col age \
  --output data/morph.csv
```

Create a deterministic identity-level 80/20 split:

```bash
python tools/split_metadata.py --input data/morph.csv --output data/morph_split.csv --seed 42
```

### End-to-end workflow

For MORPH:

```bash
python -m pap_age.cli train-cnn --config configs/morph.yaml
python -m pap_age.cli extract-features --config configs/morph.yaml
python -m pap_age.cli fit-pca --config configs/morph.yaml
python -m pap_age.cli train-sequence --config configs/morph.yaml
python -m pap_age.cli evaluate --config configs/morph.yaml
```

For FG-NET replace the config with `configs/fgnet.yaml`.

### Face preprocessing

Three reproducible preprocessing modes are supported:

1. `center`: deterministic square center crop, useful for already-cropped face datasets.
2. `mtcnn`: automatic detection/alignment with `facenet-pytorch`.
3. `boxes`: crops using a supplied bounding-box CSV. This mode can consume detections produced by an external DPM face detector, preserving compatibility with a DPM-based preprocessing protocol.

Example:

```bash
python tools/preprocess_faces.py \
  --metadata data/morph_split.csv \
  --output-root data/morph_aligned \
  --output-metadata data/morph_aligned.csv \
  --detector mtcnn --size 299
```

### Outputs

Each experiment is written under `runs/<dataset>/`:

```text
runs/morph/
├── cnn/best.pt
├── features/features.npz
├── pca/pca.joblib
├── pca/features_pca.npz
├── sequence/best.pt
└── evaluation/
    ├── metrics.json
    ├── predictions.csv
    └── cs_curve.png
```

### Smoke test without external datasets

```bash
python scripts/make_toy_data.py --output data/toy --people 12 --images-per-person 4
python -m pap_age.cli smoke-test --data-root data/toy
pytest -q
```

---

## 中文

PAP-Age 是一个完整的**个体老化模式人脸年龄识别**研究代码库。系统不仅利用单张人脸的静态外观特征，还对同一个体按时间顺序排列的人脸变化轨迹进行建模，从而学习个体化的年龄演化规律。

整体流程为：

```text
人脸图像
   ↓
人脸检测 / 对齐 / 裁剪 / 归一化
   ↓
Inception-v4 CNN
   ↓ 1536维 GAP 特征
PCA 流形投影（保留95%方差）
   ↓ 紧凑判别特征
按身份构建年龄时间序列
   ↓
LSTM 时序建模
   ↓
时间注意力机制
   ↓
对每个有效时间步输出年龄预测
```

仓库已经包含数据准备、身份级确定性划分、CNN训练、1536维特征提取、PCA拟合、个体时间序列构建、LSTM+注意力训练、MAE/CS评价、单序列推理、消融实验、结果绘图、MORPH与FG-NET配置、单元测试和GitHub Actions CI。

### 核心设计

- **CNN外观编码器：** Inception-v4 通过全局平均池化得到1536维人脸特征。
- **流形投影：** PCA仅在训练集上拟合，保留95%的特征方差并去除冗余维度。
- **个体化时序建模：** 按身份分组，并按照年龄升序排列，构成长度不固定的个体老化序列。
- **注意力增强LSTM：** LSTM学习年龄随时间的演化关系，注意力机制进一步重加权历史状态中的关键信息。
- **评价指标：** 直接实现平均绝对误差MAE与0–10年误差阈值下的累计正确率CS。

### 协议参数

仓库提供的两个配置文件按照研究设定写入核心训练参数：

| 数据集 | CNN学习率 | CNN轮数 | CNN批量 | CNN丢弃率 | LSTM学习率 | LSTM轮数 | LSTM批量 | PCA保留方差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MORPH | 0.001 | 100 | 80 | 0.8 | 0.001 | 300 | 20 | 0.95 |
| FG-NET | 0.001 | 100 | 2 | 0.8 | 0.001 | 300 | 1 | 0.95 |

Inception-v4 的GAP特征固定为1536维。在原始数据分布和相应处理条件下，95%方差PCA的目标维数会由数据自动决定；方法设定中MORPH对应111维、FG-NET对应171维。若预处理、划分或样本集合发生变化，实际维数也可能变化。

### 安装

推荐 Python 3.10–3.12：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -U pip
pip install -e .
```

如需MTCNN自动人脸检测/对齐：

```bash
pip install -e ".[face]"
```

### 数据格式

仓库不分发MORPH和FG-NET原始图像。统一使用如下CSV：

```csv
path,person_id,age
/path/to/image1.jpg,0001,16
/path/to/image2.jpg,0001,21
/path/to/image3.jpg,0002,35
```

FG-NET可直接解析常见的`001A02.jpg`式文件名：

```bash
python tools/prepare_fgnet.py --images /data/FGNET/images --output data/fgnet.csv
```

MORPH或其他带元数据表的数据集可执行：

```bash
python tools/prepare_from_csv.py \
  --source /data/morph/metadata.csv \
  --images-root /data/morph/images \
  --path-col file --person-col subject_id --age-col age \
  --output data/morph.csv
```

按身份划分80%训练集和20%测试集：

```bash
python tools/split_metadata.py --input data/morph.csv --output data/morph_split.csv --seed 42
```

### 完整运行流程

MORPH：

```bash
python -m pap_age.cli train-cnn --config configs/morph.yaml
python -m pap_age.cli extract-features --config configs/morph.yaml
python -m pap_age.cli fit-pca --config configs/morph.yaml
python -m pap_age.cli train-sequence --config configs/morph.yaml
python -m pap_age.cli evaluate --config configs/morph.yaml
```

FG-NET只需将配置替换为`configs/fgnet.yaml`。

### 人脸预处理

实现了三种模式：

1. `center`：固定中心方形裁剪，适用于本身已经较好裁剪的人脸图像。
2. `mtcnn`：使用`facenet-pytorch`自动检测和对齐。
3. `boxes`：读取外部边界框CSV进行裁剪。如果已经使用DPM检测器产生人脸框，可直接通过该模式接入，从而兼容DPM预处理流程。

示例：

```bash
python tools/preprocess_faces.py \
  --metadata data/morph_split.csv \
  --output-root data/morph_aligned \
  --output-metadata data/morph_aligned.csv \
  --detector mtcnn --size 299
```

### 输出目录

每个实验结果写入`runs/<dataset>/`：

```text
runs/morph/
├── cnn/best.pt
├── features/features.npz
├── pca/pca.joblib
├── pca/features_pca.npz
├── sequence/best.pt
└── evaluation/
    ├── metrics.json
    ├── predictions.csv
    └── cs_curve.png
```

### 无外部数据快速自检

```bash
python scripts/make_toy_data.py --output data/toy --people 12 --images-per-person 4
python -m pap_age.cli smoke-test --data-root data/toy
pytest -q
```
