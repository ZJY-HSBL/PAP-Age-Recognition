# Research Overview / 研究说明

## English

### 1. Problem definition

For a person with chronologically ordered facial images, let the CNN feature at time step `t` be `c_t ∈ R^1536`. A PCA projection maps this high-dimensional feature to a compact vector `x_t ∈ R^d`, where `d` is selected automatically so that at least 95% of the training-feature variance is retained.

For identity `i`, the sequence is:

`X_i = (x_1, x_2, ..., x_T)` with ages `Y_i = (y_1, y_2, ..., y_T)`.

The recurrent state is modeled as:

`s_t = f(s_{t-1}, x_t; θ)`.

The implementation uses an LSTM for `f`, then applies additive temporal attention. With hidden states `h_j` and query `h_t`:

`e_{t,j} = vᵀ tanh(W_h h_j + W_q h_t)`

`α_{t,j} = softmax_j(e_{t,j})`

`context_t = Σ_j α_{t,j} h_j`.

For causal attention, only `j ≤ t` contributes. The age head predicts from `[h_t ; context_t]`.

### 2. CNN stage

Inception-v4 is used as the appearance encoder. Its global average pooling representation is 1536-D. The repository trains an auxiliary discrete-age classification head and uses the expectation of the softmax distribution as an interpretable continuous age estimate during CNN validation. The classification head is discarded for the sequence stage; only the 1536-D representation is exported.

### 3. PCA stage

PCA is fitted on training identities only. `n_components=0.95` makes scikit-learn retain the minimum number of components explaining at least 95% of variance. This avoids using test-set statistics and automatically adapts the dimension to the actual processed dataset.

### 4. Sequence stage

All samples are grouped by `person_id` and sorted by age. Variable-length sequences are padded only inside a batch. LSTM packing prevents padded positions from affecting the recurrent states, and the attention mask prevents padded keys from receiving probability mass.

### 5. Evaluation

Mean Absolute Error:

`MAE = (1/N) Σ |y_i - ŷ_i|`.

Cumulative Score at tolerance `m` years:

`CS(m) = 100 × count(|y_i - ŷ_i| ≤ m) / N`, for `m = 0, 1, ..., 10`.

The evaluator saves scalar MAE, every CS point, raw predictions, and a CS curve.

### 6. Experiment controls

The protocol-specific parameters are in `configs/morph.yaml` and `configs/fgnet.yaml`. Parameters not fixed by the research specification—optimizer, hidden size, attention dimension, exact augmentation, and seed—are exposed explicitly so experiments can be audited.

---

## 中文

### 1. 问题定义

对同一个体按时间排序的人脸图像，设第`t`个时刻由CNN提取的特征为`c_t ∈ R^1536`。PCA将其映射到低维向量`x_t ∈ R^d`，其中`d`由训练特征自动确定，使累计解释方差不低于95%。

对于第`i`个个体：

`X_i = (x_1, x_2, ..., x_T)`，对应年龄序列为`Y_i = (y_1, y_2, ..., y_T)`。

递归状态写为：

`s_t = f(s_{t-1}, x_t; θ)`。

实现中使用LSTM表示`f`，随后加入加性时间注意力。对于隐藏状态`h_j`和当前查询`h_t`：

`e_{t,j} = vᵀ tanh(W_h h_j + W_q h_t)`

`α_{t,j} = softmax_j(e_{t,j})`

`context_t = Σ_j α_{t,j} h_j`。

启用因果注意力时，仅允许`j ≤ t`的历史状态参与。最终年龄回归头基于`[h_t ; context_t]`输出年龄。

### 2. CNN阶段

外观编码器采用Inception-v4，其全局平均池化特征为1536维。仓库使用离散年龄分类头训练CNN，并在CNN验证阶段通过Softmax年龄分布的期望值得到连续年龄估计。进入时序阶段后丢弃分类头，仅导出1536维特征。

### 3. PCA阶段

PCA只在训练身份上拟合。`n_components=0.95`表示自动保留能够解释至少95%训练特征方差的最小主成分数，避免利用测试集统计信息，并使降维维数适应实际处理后的数据。

### 4. 时序阶段

所有样本按`person_id`分组并按年龄排序。不同个体序列长度可以不同，只在batch内部进行补齐。LSTM使用packed sequence避免padding影响循环状态，注意力掩码则保证padding位置不会获得权重。

### 5. 评价指标

平均绝对误差：

`MAE = (1/N) Σ |y_i - ŷ_i|`。

误差阈值为`m`年时的累计正确率：

`CS(m) = 100 × count(|y_i - ŷ_i| ≤ m) / N`，其中`m = 0, 1, ..., 10`。

评估程序会保存MAE、所有CS点、逐样本预测结果和CS曲线。

### 6. 实验控制

MORPH和FG-NET的核心协议参数分别放在`configs/morph.yaml`与`configs/fgnet.yaml`中。对于研究设定中未明确规定的优化器、隐藏层宽度、注意力维度、数据增强与随机种子等内容，仓库全部显式参数化，便于审计和复现实验条件。
