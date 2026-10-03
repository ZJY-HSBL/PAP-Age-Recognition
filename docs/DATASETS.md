# Dataset Guide / 数据集指南

## English

### FG-NET
FG-NET contains multiple ages per identity and is naturally suited to longitudinal sequence construction. The helper `tools/prepare_fgnet.py` parses the common `<person>A<age>` filename convention. Check the generated CSV before training because local copies can use slightly different suffixes.

### MORPH
MORPH distributions are typically accompanied by metadata. Because filename conventions and licensed distributions differ, `tools/prepare_from_csv.py` maps arbitrary source columns to the unified schema. Keep the original subject identifier intact; identity-level splitting depends on it.

### Leakage prevention
For the temporal model, an identity must never appear in both train and test splits. `tools/split_metadata.py` performs identity-level splitting and propagates the split label to every image of that identity.

### Preprocessing
The target image size for Inception-v4 is 299×299. For a DPM-based detector, export detections as a CSV with `path,x1,y1,x2,y2`; then use `tools/preprocess_faces.py --detector boxes --boxes boxes.csv`.

## 中文

### FG-NET
FG-NET中同一身份包含多个年龄阶段的人脸图像，适合构建纵向年龄序列。`tools/prepare_fgnet.py`可解析常见的`<person>A<age>`文件名。由于不同本地副本的后缀可能不同，训练前应检查生成的CSV。

### MORPH
MORPH通常随数据分发元数据。考虑到不同授权版本的文件命名和字段可能不同，`tools/prepare_from_csv.py`允许将任意字段映射到统一格式。务必保留原始身份ID，因为身份级数据划分依赖该字段。

### 防止数据泄漏
时序模型训练时，同一个身份不得同时进入训练集和测试集。`tools/split_metadata.py`按身份进行划分，并将同一身份的全部图像写入同一个split。

### 预处理
Inception-v4的目标输入尺寸为299×299。如果使用DPM检测器，可将检测结果导出为`path,x1,y1,x2,y2`格式CSV，再执行`tools/preprocess_faces.py --detector boxes --boxes boxes.csv`。
