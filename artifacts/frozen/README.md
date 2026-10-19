# Frozen artifacts

本目录只保存已经提交或正式冻结、必须随 Git 复现的不可变配置。

- `int_001.json`：已提交的 `FR-002/HF-001` 等权冒烟因子记录。
- `joint_elastic_net_v02.json`：历史 I 路线冻结快照、提交文件和本地 J 代理记录。
- `joint_lightgbm_v03.json`：历史 T 路线冻结快照、提交文件和本地 J 代理记录。

运行产生的 CSV、临时模型和待选择结果继续放在 `reports/`，不进入 Git。
新 S/I/T 运行在完成真实 AIStudio 验证前不得覆盖这些已追踪冻结快照。
