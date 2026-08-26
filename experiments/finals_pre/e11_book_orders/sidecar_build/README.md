# E11 sidecar 的取数与校验

sidecar 只含 4 个新通道，键 `(instrument, timestamp)` 与
`unified_microstructure_store_v2_2019_2024` 对齐，训练时由 fastpack 左连接追加在
17 个通道之后。**现有 store 一个字节不动** —— 磁盘放不下第二份 24 G，而且原通道
一旦变了字节，已有的 5 个 o2o baseline seed 就不能再当对照组。

## 为什么必须去 AIStudio 取

本地 store 只存了 17 个 channel，21 个 raw 列算完就丢；raw 源
（`data/e2e_data` 的月度 feather）已在 2026-08-20 清理老树时删除。
所以 `num_orders` 与四五档价量在本地**无论如何算不出来**。

## 跑法

在 AIStudio（`ssh aistudio`，2 核 / 8 G / 无 GPU，只取数不训练）：

```bash
python3 e11_extract.py --start 2019-01-01 --end 2025-01-01 --out ~/e11/sidecar
python3 verify_sidecar.py          # 逐日与 store 键比对，必须 0 缺 0 多
```

`store_keys.parquet` 由 AutoDL 侧导出（store 的 `(trade_date, instrument)` 清单，
0.8 MB），用于过滤——AIStudio 表每天 1707–2179 只股票，store 只要约 1000 只。

全量约 14 分钟、1456 个分区、4.7 GB。

## 回传

Mac 中转实测只有 0.22 MB/s（4.7 G 要 6 小时）。改用 AIStudio 直连 AutoDL
（`connect.cqa1.seetacloud.com:38527` 公网可达），实测 16.4 MB/s，约 5 分钟。
需要在 AutoDL 上临时授权一把专用公钥，**传完立即撤销**。

## 两处数据边界

- `num_orders` 在 **2019-06 之前恒为 0，不是 NULL**（非空率 100%，任何只查
  NULL 的检查都会放行）。取数脚本把那一段的通道 1、2 写成 NaN，由 `observed_mask`
  标成未观测。不能把 0 当真值——那会落进「有值但恒定」，日级 std = 0。
- 四五档价量 **2019-01 就有**（2019-03-01 覆盖 99.1%），通道 3、4 全程可用。

## 一处口径选择

`order_count_imbalance_l1` 在单边无报价时按 0 笔计，给出 ±1，与已有的
`depth_imbalance_l1` 惯例一致（`_positive_book_side` 把无报价一侧记 0）。
第一版写成缺失，实测 2020-02-03（新冠暴跌复市首日）只有 22.2% 的分钟买方有
报价，那一版会把观测率打到 17.5% —— **恰好在压力日把信号扔掉**，而压力日
IC/IR 是 A 类四项之一。改判之后该日观测率 99.2%、均值 −0.780（极度卖方压倒）。

`log_book_gap_ticks` 反过来保持缺失：单边书的「稀疏程度」没有定义，硬凑会让
通道在不同日子含义不同。由 `observed_fraction` 告诉模型「今天盘口是单边的」。
