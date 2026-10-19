# 测试说明

`tests/` 保留 49 个测试文件、229 个测试函数，按验证对象分组。参数化测试会展开成更多测试项；完整测试还包含 `experiments/` 和 `backtest/` 中的测试。

| 目录 | 文件数 | 验证内容 |
| --- | ---: | --- |
| `publication/` | 3 | 公开源码的敏感内容、内嵌产物、符号链接与打包源码检查 |
| `models/` | 11 | 模型、输入契约、分钟特征、前缀不变性和模型导出 |
| `evaluation/` | 6 | 标签对齐、评分、稳定性和评估接口 |
| `factors/` | 16 | 因子公式、输入契约、候选因子生成 |
| `research/` | 11 | 保留的研究工具、历史导出器和因果性审计 |
| `integration/` | 2 | 安装后的模块导入和保留的 shell 入口语法 |

## 保留原则

只要测试仍验证公开源码的实际行为，即使关联的是非最终研究模型，也予以保留。模型、因子、评分和因果性测试没有因整理目录而删除。

删除的 10 个旧测试函数涉及私有权重和清单依赖、历史产物及目录布局的存在性断言、已完成包迁移的文本扫描。其中原 `test_frozen_six_year_artifacts.py` 整体移除；X_tree 文件只保留仍有用的读取列行为测试，改名为 `research/test_legacy_x_tree_helpers.py`。导入和 shell 语法检查仍保留，不再要求已删除的传输脚本存在。

## 运行

发布检查只依赖 Python 标准库，可在安装依赖前运行：

```bash
python scripts/check_public_release.py
python -m unittest discover -s tests/publication -p "test_public*.py" -v
```

项目完整检查与 CI 一致：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q -o addopts="" experiments/finals_pre/e9_tcn_architecture/test_protocol.py
python -m pytest -q -o addopts="" experiments/finals_pre/e8_loss_comparison/test_e8.py
coverage run -m pytest -q -rs
coverage combine
coverage report --include="src/*" --fail-under=75
```

请在仓库根目录运行完整测试，不要只指定 `tests/`，否则会漏掉 `experiments/tests/`、E7 与 `backtest/src/` 的检查。E8、E9 使用各自封存的模型源码，因此按 CI 配置分别运行，避免同名模块互相影响。

源码发布版不携带 AutoDL 私有数据，相关的两项外部数据集成检查会说明原因并跳过。E8、E9 的历史结果重算也需要私有结果文件；它们的协议测试仍须通过。`src/` 的覆盖率门槛保留为 75%，不因本次整理而降低。
