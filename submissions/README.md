# Submission artifacts

冻结的平台提交包。每个目录自带 manifest、校验记录与 AIStudio handoff；
上传严格按各目录 `submission_bundle_manifest.json` 声明的文件执行。

| 目录 | 角色 |
|---|---|
| `m_expanding_history_e3_20260803/` | **正式冠军**：M_raw 扩窗 e3（公榜 0.9548 / 私榜 0.8874，提交 id `6d0fe02c-…`） |
| `m_l5/`, `m_l5_oos2023/` | M 五档挑战线提交包（未超越冠军，保留证据） |
| `m_multiaxis_full_20260804/` | M multiaxis 挑战线提交包 |
| `elasticnet454_formal/` | EN454 基线正式包 |
| `en454_t_residual_020/` | EN454 + T 残差 0.20 路线 |
| `t_protocol_best/` | T 协议最优路线 |
| `x_tree/`（及根目录 `unified_x*` 文件） | X-tree 路线与其运行时 |

构建/刷新 M 包：

```bash
python scripts/build_unified_m_raw_submission.py --help
python scripts/build_unified_m_l5_submission.py --help
```

这些包是拼装式冻结产物（运行时用 exec 合并命名空间），已从 ruff 检查中排除；
不要手工编辑包内文件——任何修改都会使 manifest 的 SHA-256 失效。
S/I/T 时代的提交构建器已退役，见 git 历史。
