# 决赛复盘实验（finals_pre）

**这里只放"是什么、在哪"，不放结果数字。**
全部实验的结论、A 四小项与 N 分、判读口径，在 `BigAlpha_Pre/experiment_log.md`
（Mac 端私有仓库），那份由脚本从本仓库现读生成。

> 这份 README 曾经带着一张结果表，用的是一套**更早的编号**（E0=epoch、E1=双通路、
> E3=主结果），数字停在换完整 Barra 中性化之前。两套编号并存了三天没人发现，
> 拿着仓库目录名反查实验时会指到错误的实验上。所以现在这里不再复制任何结果数字——
> 复制就会漂移，而漂移不报错。

## 编号对照

| 编号 | 是什么 | 脚本 | 产物 `reports/dependencies/finals_pre/` |
|---|---|---|---|
| **E1** | 通道置换重要性（冻结模型上跨股票打乱一组通道） | `e1_channel_importance/` | `e1_channel_importance/` |
| **E2** | 卷积核尺度：单尺度（a）／数值敏感性（b）／种子稳健性（c） | `e2_kernel/` | `e2a_*`、`e2b_*`、`e2c_seed_robustness/` |
| **E3** | 双通路消融 + Linear-85 对照 | `e3_pathway_ablation/` | `e3_pathway_ablation/` |
| **E3b** | DeepSets 加行业分组 context | `e3b_industry_context/` | `e3b_industry_context/` |
| **E4** | walk-forward 26 块 | `e4_walkforward/` | `e4_walkforward/` |
| **E5** | 逐 epoch 的 OOS 曲线 | `e5_epoch_curve/` | `e5_epoch_curve/`、`e5_reference_*` |
| **E6** | 主结果分析包 + 平台收益口径确认（o2o） | `e6_main_analysis/` | `e6_main_analysis/` |
| **E6b** | 只换训练标签 o2c → o2o | `e6b_o2o_label/` | `e6b_o2o_label/` |
| **E7** | N 分：对 LightGBM-454 池残差的增量预测力 | `e7_nscore/` | `e7_nscore/` |
| **E8** | EN454 掺混剂量-响应 | `e8_en_blend/` | `e8_en_blend/` |
| **E9** | 种子与标签矩阵（把每个消融补到 3 个种子） | `e9_seed_and_label_matrix/` | `e9_seed_and_label_matrix/` |
| **E10** | 通道组重训消融（拿掉整组再从零训） | `e10_channel_ablation/` | `e10_channel_ablation/` |
| **E11** | 盘口挂单笔数与四五档价量（17 → 21 通道） | `e11_book_orders/` | `e11_book_orders/` |

`E0` 不是独立实验，是 E2 那批的启动器（`e2_kernel/e0_e2_launcher.sh`）。
**没有 E12。**

## 目录约定

```text
experiments/finals_pre/
├── <E 名>/              一 E 一目录，该实验的全部脚本
├── common/              跨实验的库代码（fastpack、score_o2o、prefetch、industry）
├── price_level_audit/   数据审计，不是 E 编号实验
└── 根上 6 个跨实验工具  rescore_*（三个互相 import）、run_nscore_all_arms.sh、
                        rerun_scoring.sh、run_final_batch.sh
```

产物同构：`reports/dependencies/finals_pre/<E 名>/`，另有 `logs/`（纯日志）
与 `shared/`（跨实验共用：`o2o_labels.parquet`、`o2o_decomposition.csv`、J 打分结果等）。

**一处例外**：`e7_nscore/noise_null.log` 虽是 `.log`，但它是噪声标定的数据源、
写在 `BigAlpha_Pre/results/audit_numbers.py` 的 MANIFEST 里，所以不在 `logs/`。

## 统一协议

expanding 训练、起点 2019-01-02、标签隔离 1 个交易日（代码硬断言）、
epochs=3、lr=4e-4、max_stocks=1200、2024 整年 untouched 预测、RTX 4090D。
seed 组 {20260801, 20260812, 20260823, 20260904, 20260915}。
消融只动被测部件，其余保持一致，因此差异可归因。
每个 checkpoint 的 sha256 记在同目录的 `oos_metrics.json` 里。

## 数据管线

**打包是瓶颈，不是 GPU。** 单个训练日 1.24 s 里，读 parquet 只占 0.06 s，
打包成张量占 0.81 s（65%），GPU 前向反向 0.37 s。慢在逐股票的 Python 循环。

`common/fastpack.py` 提供 polars 打包路径，由训练器的 `--fast-pack` 启用
（**默认关闭**）；冻结的提交 bundle 自带一份 pandas 实现，未做任何改动。
打包 0.81 s → 0.13 s（4×），单个完整 run 约 2 小时 → 37 分钟。

**等价性验证三层，全部逐字节通过**：随机 20 天；5 个边界用例（空列表、单只、
不存在的代码、全量、分区缺失——空列表那个测出真 bug：polars 把空列推断成 Null
导致 join 崩）；端到端复用冻结 checkpoint，逐位复现
`rank_ic_mean 0.041769313009848055`。

**E11 的 sidecar**：4 个新通道单独落盘（1456 天 / 4.7 GB，在
`/root/autodl-tmp/e11_sidecar`，不入库），由 `--sidecar` 按
`(instrument, timestamp)` 左连接追加在 17 个通道之后。**现有 store 一个字节不动**——
原通道一旦变了字节，已有的 5 个 baseline seed 就不能再当对照组。
取数脚本见 `e11_book_orders/sidecar_build/`。

**尚未解决——冗余打包**：一个 run 打包 8730 次但只涉及 1213 个不同交易日（冗余 7 倍）。
不能简单按日期缓存，因为训练循环每个 epoch 用 `rng.choice` 重新抽样 1200 只股票。
正确做法是缓存整天全量张量再按当轮抽样切片（29 GB 内存放得下，磁盘放不下）；
预计再快 1.3 倍，需重新做逐字节对拍。

## 环境注意

1. `bigalpha_2026_factors` 的 editable 安装曾指向已退役的老工作树，已从本工作树重装。
2. 新写的打分路由必须遵守两条合同：ambient `bigalpha2026` import；
   标签全域覆盖并以中性 0 填补。
3. 数据根一律用 `/root/autodl-tmp/data`。老路径
   `/root/autodl-tmp/projects/bigquant/data` 目前是指向它的软链（清理老树时留的垫片），
   **别在新脚本里用**——垫片一旦被删就断。
