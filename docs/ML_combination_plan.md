# BigAlpha 2026 多模型 Alpha Framework 项目计划书

> Version: v1.0
> Goal: 构建统一的 Alpha Machine Learning Framework，以最终 Competition Score（J）为唯一优化目标，实现 Rule、Linear、Tree、Neural 等多模型协同生成 Alpha。

---

# 一、项目目标

## 最终目标

不是训练某一种模型。

而是构建能够持续产生高质量 Alpha 的统一机器学习平台。

最终输出：

```
date
instrument
factor
```

优化目标：

```
Competition Score (J)

J = 0.3 A + 0.7 B
```

其中：

- A：IC / ICIR / Sharpe / Stress 等表现
- B：进入官方因子池后的增量贡献

所有模型最终均以 Rolling OOS J 为评价标准。

---

# 二、总体架构

```
Raw Data
      │
      ▼
Feature Engineering
      │
      ▼
Shared Factor Pool
      │
      ├──────────────┐
      │              │
      ▼              ▼
 Rule Route      Machine Learning Routes
                     │
                     ├── ElasticNet
                     ├── LightGBM
                     ├── CNN
                     ├── Transformer
                     ├── MLP
                     └── Future Models
                             │
                             ▼
                 Model-generated Factors
                             │
                             ▼
               Cross-model Combination
                             │
                             ▼
               CompetitionScoreReference
                             │
                             ▼
                       Rolling J
```

---

# 三、核心设计原则

## 1. Model Agnostic

任何模型都可以接入。

例如：

- ElasticNet
- XGBoost
- LightGBM
- CatBoost
- CNN
- Transformer
- MLP
- TabNet
- FT-Transformer
- Graph Model

统一接口：

```python
fit()

predict()

save()

load()
```

无需修改整体框架。

---

## 2. Shared Feature Pool

所有模型使用统一特征池。

```
Raw Data

↓

Feature Engineering

↓

Unified Alpha Feature Store

↓

Model
```

避免：

- 每个模型重新做 Feature
- 重复维护

---

## 3. Strict Time Causality

任何模型不得看到未来。

Rolling：

```
Train

↓

Predict

↓

Evaluate
```

禁止：

- future leakage
- full sample normalization
- future label

---

## 4. OOS First

所有模型必须输出：

```
Strict OOS Factor
```

禁止：

```
Train Prediction
```

最终拼接：

```
Fold1

Fold2

Fold3

↓

Complete OOS Factor
```

---

# 四、数据层

## Raw Data

因子构造只允许：

```
Minute Bar (`bar1m`)

Financial
```

`instruments` 仅作为辅助数据，用于确定输出股票集合和
`date/instrument` 键，不作为模型特征。

---

## Feature Layer

统一输出：

```
Unified Alpha Feature Store
```

要求：

```
shape

mask

dtype

date

instrument
```

统一管理。

---

## Feature Processing

统一完成：

```
Missing Mask

Cross-sectional Normalization

Winsorization

Feature Projection

Feature Cache
```

---

# 五、Model Interface

统一接口：

```python
class BaseModel:

    fit()

    predict()

    save()

    load()
```

所有模型继承：

```
ElasticNetModel

LightGBMModel

CNNModel

TransformerModel

MLPModel
```

---

# 六、Neural Network Architecture

目标：

构建统一 Alpha Encoder。

---

## Input

```
Candidate462
```

支持：

```
Missing Mask

Variable Length

Variable Universe
```

---

## Projection

```
462

↓

128
```

统一 Embedding。

---

## Temporal Encoder

可配置：

```
MLP

CNN

Transformer

CNN + Transformer
```

支持模块插拔。

---

### CNN

多尺度：

```
kernel

3

5

15
```

Depthwise Conv。

---

### Transformer

建议：

```
2 Layers
```

支持：

```
Causal Mask
```

---

## Cross-sectional Context

可选：

```
DeepSets

Attention Pool

Mean Pool

None
```

全部可配置。

---

## Head

MLP Head：

```
128

↓

64

↓

1
```

输出：

```
Alpha
```

---

# 七、Model Factory

统一创建：

```
create_model()

↓

ElasticNet

LightGBM

CNN

Transformer

...
```

避免：

```
if else
```

堆积。

---

# 八、Training Framework

统一 CLI：

```
train

predict

evaluate
```

例如：

```
python train.py

python predict.py

python evaluate.py
```

所有模型共享入口。

---

# 九、Rolling Framework

统一 Rolling。

```
Fold1

Train

Predict

Evaluate

↓

Fold2

↓

Fold3
```

最终：

```
OOS Factor
```

---

# 十、Evaluation Framework

唯一评价器：

```
CompetitionScoreReference
```

输出：

```
A

B

J
```

另外输出：

```
IC

RankIC

Sharpe

Turnover

Coverage
```

方便分析。

---

# 十一、Model Repository

保存：

```
Rule Factor

ElasticNet Factor

LightGBM Factor

CNN Factor

Transformer Factor
```

统一管理。

---

# 十二、Model Combination

最终组合：

```
Rule

+

ElasticNet

+

LightGBM

+

CNN

+

Transformer

↓

Final Alpha
```

支持：

```
Equal Weight

Rank Average

Linear Blend

Future Meta Model
```

---

# 十三、实验管理

每次实验记录：

```
Config

Seed

Fold

Training Time

GPU

Feature Version

Git Commit
```

保证可复现。

---

# 十四、自动测试

必须包含：

## 数据

- Shape
- Dtype
- Missing
- Date Alignment

---

## 时间

- Future Leakage
- Rolling Split
- Label Alignment

---

## Feature

- Cross-sectional Standardization
- Mask
- Projection

---

## Model

- Forward
- Backward
- Save
- Load

---

## GPU

Smoke Test。

---

# 十五、实验流程

```
Build Feature

↓

Train Model

↓

Predict OOS

↓

CompetitionScoreReference

↓

Rolling J

↓

Compare

↓

Select

↓

Combine

↓

Submit
```

---

# 十六、开发阶段

## Phase 1

基础设施

- Feature Pipeline
- Rolling Framework
- Evaluation
- CLI
- Cache

---

## Phase 2

Baseline Models

- Rule
- ElasticNet
- LightGBM

---

## Phase 3

Neural Models

- MLP
- CNN
- Transformer

---

## Phase 4

Cross-sectional Context

- DeepSets
- Attention

---

## Phase 5

Model Ensemble

- Multi-model Alpha
- Combination
- Final Submission

---

# 十七、设计原则总结

整个项目遵循四个原则：

**统一数据**

所有模型共享同一 Feature Pool。

**统一接口**

所有模型遵循相同 API。

**统一评价**

所有实验最终以 Rolling Competition Score（J）作为唯一模型选择标准。

**统一组合**


---

# 十八、因子赛道提交硬约束

## 训练周期与请求计算边界

- 离线模型训练使用 2019–2024 历史期，并按时间滚动产生严格 OOS 预测。
- “提前半年”只约束平台单次请求中构造因子可读取的历史窗口，不是模型训练期。
- 时序模型默认使用最近 60 个交易日作为序列 lookback；60 日也不是训练期。
- 所有截面统计必须按当日独立计算，不得使用完整验证区间。

## 数据源白名单

- 因子值只能由 `bar1m` 和 `financial` 构造。
- `instruments` 仅可用于确定当日输出股票集合及 `date/instrument` 键。
- 禁止将 `factorlib`、`exposure`、行业、风险模型或其他基础特征输入模型。
- 每个生成特征必须记录来源清单，并通过自动白名单检查。

## 提交产物

- 最终同时生成语义等价的 `.py` 与 `.ipynb`。
- 两者都必须暴露平台要求的 `main(datasources, start_date, end_date)`。
- 导出的 Python 文件需通过编译、短窗口实数输出和 full/cutoff 前视检查。
