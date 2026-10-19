# 最终提交源码：M_raw

比赛最终实际上传的是 `m_raw/`：17 通道原始分钟微观结构模型，已由项目作者确认。此目录只保留这一版的推理入口和必要配套源码。

- [m_raw/unified_m_raw.ipynb](m_raw/unified_m_raw.ipynb)：平台 notebook 入口
- [m_raw/unified_m_raw.py](m_raw/unified_m_raw.py)：数据查询、特征构造与推理
- `m_raw/unified_m_microstructure.py`、`unified_m_temporal.py`、`unified_m_base.py`：独立运行所需的模型与辅助代码
- `m_raw/public_artifacts.py`：外部权重的安全加载

这是公开源码版，训练权重不公开，也不通过网络自动下载。配置和运行要求见 [外部产物约定](ARTIFACT_CONTRACT.md)。保留的 notebook 已清空执行输出；本次未重新运行平台推理或复现比赛成绩。

[source_manifest.json](source_manifest.json) 只记录当前唯一入口和权重环境变量，是源码使用说明，不是提交证明、评分文件或训练权重。原来的构建清单、校验记录及模型权重 JSON 均不在此目录发布。

旧 X、T、M-l5、ElasticNet、多轴模型等研究示例见 [历史研究源码](../experiments/legacy_submissions/README.md)。它们不属于最终提交版本。
