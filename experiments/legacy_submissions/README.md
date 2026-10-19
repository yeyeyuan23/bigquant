# 历史研究源码

这里保留非最终提交路线的算法和推理研究示例，不代表比赛最终上传版本。最终版只在 [`submissions/m_raw`](../../submissions/m_raw/) 中维护。这里没有训练权重、行情数据、旧上传 JSON 清单或执行输出。

| 路线 | 源码位置 |
| --- | --- |
| Candidate454 MLP | `unified_x.py` 及同目录因子定义 |
| M-l5 | `m_l5/` |
| 多轴分钟模型 | `../../src/alpha_models/multiaxis.py`；训练入口 `../../scripts/train_m_multiaxis.py`；历史推理示例 `m_multiaxis/predict.py` |
| ElasticNet | `elasticnet454_formal/` |
| ElasticNet + T residual | `en454_t_residual_020/` |
| Temporal | `t_protocol_best/` |
| LightGBM | `x_tree/` |

通用交接说明、只负责 `import main` 的 notebook 包装和重复的 M-l5 OOS 副本已移除。各研究目录保留必要的 sibling modules，避免改变历史公式和特征顺序；这些独立示例中的相同辅助源码用于支持各自运行。权重结构见 [研究产物约定](ARTIFACT_CONTRACT.md)。

多轴模型的代码按职责整理为：

```text
src/alpha_models/multiaxis.py        模型与数据变换
scripts/train_m_multiaxis.py        训练入口
scripts/predict_m_multiaxis_store.py 本地预测
scripts/build_m_multiaxis_submission.py 私有上传包导出
tests/models/test_multiaxis.py      模型测试
```

先按根目录依赖说明安装项目，例如 `python -m pip install -e ".[neural]"`。再使用 `python scripts/train_m_multiaxis.py --help` 查看显式数据路径参数。导出器的 `--source-dir` 指项目根目录，输出必须放到 `artifacts/local/` 或仓库外。所有私有数据、拟合统计量和模型权重由运行者自行准备；本次没有重新训练或验证这些历史实验。
