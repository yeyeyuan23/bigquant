# M_raw 外部产物约定

最终提交模型为 17 通道 M_raw。本目录公开模型架构、特征构造与推理源码，不包含行情、暴露、标签、预测、训练权重或 checkpoint。

## 配置

自行准备有权使用、来源可信的 PyTorch checkpoint，内容须包含 `config` 和 `state_dict`，并与 `MicrostructureModel` 的网络配置一致。然后在自己的运行环境设置：

```python
import os
from pathlib import Path

os.environ["BIGQUANT_M_RAW_CHECKPOINT"] = str(
    Path("../private-artifacts/m_raw.pt").resolve()
)
```

从 `submissions/m_raw/` 运行 `unified_m_raw.ipynb`，或把该目录加入 Python 模块路径后导入 `unified_m_raw.main`。保留同目录的全部 Python 文件。平台数据接口 `dai`、PyTorch 和其他数值计算依赖需要在授权环境中另外提供。

缺失权重、文件内容缺少必要字段、网络配置或权重不兼容时会明确报错，不会退回随机模型。加载使用 `weights_only=True`；不支持这一安全模式的 PyTorch 版本会失败，不会重试不受限的 pickle 加载。

权重始终保存在仓库外，或已忽略的 `artifacts/local/` 中。生成器可能产生含权重的私人上传包：`scripts/build_unified_m_raw_submission.py` 默认写入 `artifacts/local/submissions/m_raw_240/`，这些文件不属于公开源码。

## 验证边界

本次只做公开内容规则、源码语法和加载契约检查，未加载训练权重、重跑平台推理或复现比赛成绩。历史平台分数不验证这一去除权重后的源码快照。运行者需要自行检查数据 schema、特征顺序、输出合同、前缀因果性及平台环境。

其他研究路线的权重结构见 [历史研究产物约定](../experiments/legacy_submissions/ARTIFACT_CONTRACT.md)。
