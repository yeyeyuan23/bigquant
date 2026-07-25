# BigAlpha 2026 双因子研究工程

本工程实现两个独立的日频参赛因子：

1. `hf_pressure_underreaction`：持续盘口压力与价格反应不足；
2. `quality_flow_interaction`：现金流质量与盘口确认。

原始比赛数据只在 BigQuant AIStudio 内读取。本地测试使用合成数据，不保存或导出比赛数据。
合成数据指标只验证代码和方向响应，不代表真实比赛表现。

## 目录

- `src/bigalpha2026/`：共享数据处理、两类因子、评估和约束搜索；
- `submissions/`：可上传的独立 Notebook，每个 Notebook 只产生一个因子；
- `research/constrained_search.ipynb`：平台内48候选约束搜索和筛选入口；
- `scripts/build_submission_notebooks.py`：从受测源代码生成自包含 Notebook；
- `scripts/run_synthetic_demo.py`：本地合成数据演示；
- `tests/`：接口、防泄漏、累计字段、PIT 和评估测试；
- `docs/ai_methodology.md`：AI赛道复现和审计记录模板。

## 本地验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/run_synthetic_demo.py
python3 scripts/build_submission_notebooks.py --check
```

## 平台提交

分别上传以下文件，不要将两个 Notebook 放进同一次提交：

- `submissions/factor_hf_pressure.ipynb`
- `submissions/factor_quality_interaction.ipynb`

每个 Notebook 都定义：

```python
def main(datasources, start_date, end_date):
    ...
```

并且只返回 `date、instrument、factor` 三列。

研究Notebook不是提交文件，不能与因子Notebook放在同一次提交中。
