import json,csv,hashlib,zipfile
from pathlib import Path
root=Path('/root/autodl-tmp/projects/bigquant-default')
code=root/'experiments/finals_pre/tcn_architecture_o2c'
out=root/'reports/dependencies/finals_pre/tcn_architecture_o2c'
rows=json.loads((out/'results/summary.json').read_text())
by={x['arm']:x for x in rows}
timing=list(csv.DictReader((out/'results/timing_summary.csv').open()))
runs=list(csv.DictReader((out/'results/per_run.csv').open()))
notes={
'baseline':('统一重新训练的参照','参照配置，不代表已证明最优。'),
'depth1':('减少到一块','RankIC 均值几乎相同且更省时；现有实验不足以区分，未支持三块的必要性。'),
'depth2':('减少到两块','均值略低且更省时；未检出可靠差异，不能说三块可靠优于两块。'),
'depth4':('增加到四块','三个种子均下降且通过 Holm 校正；支持原核宽四块在三轮预算下更差。'),
'single15':('仅保留 15 分钟分支','均值较高、耗时更少，但未通过校正；未支持三个尺度的必要性。'),
'single60':('仅保留 60 分钟分支','三个种子均较低，但 Holm 不显著；不能称基线显著更好。'),
'branches2':('只保留 15/60','均值较低、耗时略低；未检出可靠差异。'),
'branches4':('增加 9 分钟分支','均值较低，两个种子为正；未见可靠收益，显存增加。'),
'branches5':('增加 9 和 30 分钟分支','三个种子均较低且成本更高；支持不扩展到五分支的成本取舍，不能称显著更差。'),
'kernels_2_10_45':('换成较短窗口','均值接近且更省时；未检出可靠差异，未证明原窗口最优。'),
'kernels_5_30_120':('扩大中长窗口','主指标接近、耗时更高；部分辅助指标更高，未确认主指标收益。'),
'kernels_5_60_120':('进一步扩大中长窗口','三个种子主指标均较低且耗时更高；校正后不显著，无可靠替换依据。'),
'same_scale60':('三条独立的 60 分钟分支','三个种子均较低且更耗时，校正后不显著；不能据此证明显式多尺度必要。'),
'rf179_depth2':('两块、最大覆盖接近基线','均值略高且耗时略低；未检出可靠差异，不能证明三块必要或覆盖是主因。'),
'rf177_depth4':('四块、最大覆盖接近基线','三个种子均较高但不显著；原核宽四块的劣势不能泛化到所有四块结构。')}
link='../../../reports/dependencies/finals_pre/tcn_architecture_o2c/'
parts=['# TCN 架构实验：做了什么、结果与选型结论','',
'**已完成 15 个配置 × 3 个种子，共 45 次独立训练、评分和统计复核。每次统一训练三轮；整轮约 23 小时 46 分钟。正式提交模型和权重未替换。**','',
'## 先说结论','',
'- 没有候选配置在预先约定的 Holm 校正规则下可靠改善主指标。',
'- 原核宽 3/15/60 从三块加到四块，RankIC 三个种子均下降，校正后 p=0.0084；这是本轮唯一通过校正的性能差异。',
'- 一块、两块与三块尚不足以区分；单独 15 分钟的均值更高且更省时，但没有通过校正。因此不能写“三块必要”“三分支必要”或“3/15/60 最优”。',
'- 五分支的 RankIC 三个种子均低于基线，训练加推理总耗时增加 6.3%，峰值已分配显存增加约 30.2%。这支持在当前预算下不增加到五分支的成本取舍；Holm p=0.1833，不构成显著更差的证据。',
'- 接近相同最大覆盖的两块和四块对照都未与基线形成可靠差异，不能证明三层本身必要，也不能证明收益主要来自覆盖。','',
'## 实验做了什么','',
'只改变 TCN 的串联块数或并行核宽，隐藏宽度固定 96；统计通路、DeepSets、last/mean/attention、loss 与打分头固定。输入是 240 个固定交易分钟位置、17 通道。每个配置都使用三个种子 20260801、20260812、20260823 从头训练，包括基线；不混用旧 O2O 权重或结果。','',
'训练使用 2019–2023 的 1,213 个日期，并保留一个交易日标签隔离。训练与评分标签均为下一交易日 O2C。所有配置使用相同的 2024 年 241 个有效评分日期、240,567 个股票日期键。2024 年此前已经参与研究，因此这是历史验证期结构比较，不是全新未见测试。','',
'主指标为行业与 Barra 中性化后的平均 RankIC。以下点估计是三个种子的均值；差值始终为候选减基线。对三个固定种子的每日配对差值取平均后，同步日期执行 10 日移动块 bootstrap 10,000 次，报告未作多重校正的 95% 区间；双侧检验对 14 个候选统一做 Holm 校正。改善要求正增量、至少 2/3 种子为正并通过校正。区间跨零表示不足以区分，不表示等价。日期重采样不覆盖全部初始化不确定性。','',
'## 每个配置检验什么、得到什么结论','']
groups=[('深度：为什么串联三块',['baseline','depth1','depth2','depth4']),('分支数：为什么三个并行窗口',['baseline','single15','single60','branches2','branches4','branches5']),('核宽：为什么是 3/15/60',['baseline','kernels_2_10_45','kernels_5_30_120','kernels_5_60_120']),('解释对照：滤波器数量与近似覆盖',['baseline','same_scale60','rf179_depth2','rf177_depth4'])]
for title,arms in groups:
    parts += ['### '+title,'','| 配置：块数；核宽 | 检验什么 | RankIC | 较基线差值 | 95% 区间 | Holm p | 正增量种子 | 结论 |','|---|---|---:|---:|---|---:|---:|---|']
    for arm in arms:
        x=by[arm]; purpose,conclusion=notes[arm]
        ci='—' if x['ci_low'] is None else f"[{x['ci_low']:+.6f}, {x['ci_high']:+.6f}]"
        p='—' if x['p_holm'] is None else f"{x['p_holm']:.4f}"
        seeds='—' if x['positive_seeds'] is None else str(x['positive_seeds'])+'/3'
        parts.append(f"| {arm}：{x['depth']}；{x['kernels']} | {purpose} | {x['rank_ic']:.6f} | {x['delta_rank_ic']:+.6f} | {ci} | {p} | {seeds} | {conclusion} |")
    parts.append('')
parts += ['### “增加滤波器”究竟指什么','',
'这里的滤波器就是一组可学习的时间卷积权重。每个分支对 96 个隐藏通道各有一个深度卷积核：单分支每块有 96 个，三条独立分支每块有 288 个。60/60/60 与 3/15/60 的分支数和这类卷积核数量相同，但窗口长度、参数量和计算成本不同；三个 60 分钟分支不共享权重。60 分钟核也能学到短期形态，所以这个对照不能被解释为“完全没有多尺度能力”。','',
'### 接近 178 个位置的对照能回答到哪一步','',
'两块 4/22/90 的最大理论覆盖为 179，四块 2/11/45 为 177，基线为 178。这是在大致固定最大覆盖时观察深度差异是否仍然存在，不是为了故意把模型推到某个效果极限。本轮两个对照均未与基线形成可靠差异；原先也未检出三块优于两块，因此不能进一步把所谓三块收益归因给覆盖。','',
'最大覆盖相近不等于结构等同：核宽、参数量、路径数量和非线性次数仍不同。这不是完全隔离机制的证明。理论覆盖超过 240 的配置会接触边界填充，不会获得额外真实分钟；全日标准化、统计路和时间汇总本身已使整网依赖全天，不能说小覆盖模型“看不到上午”。','',
'## 时间、参数和显存：是否值得增加计算','',
'下表时间是三个种子的均值，显存是三个种子的最大值。训练阶段计时从配置进程启动到最后训练步，包含预检查、初始化和数据读取；总耗时另包含 checkpoint 写入、推理和因子写入，不含随后统一评分。这些都不是纯 GPU 算子计时。峰值为 PyTorch CUDA allocated/reserved，不能与 nvidia-smi 进程占用直接混用。','',
'| 配置 | 训练阶段（分） | 训练加推理总耗时（分） | 参数 | 峰值 allocated / reserved（GiB） |','|---|---:|---:|---:|---:|']
for x in timing:
    parts.append(f"| {x['arm']} | {float(x['mean_startup_through_training_minutes']):.2f} | {float(x['mean_total_minutes']):.2f} | {int(x['parameters']):,} | {float(x['peak_gpu_allocated_gib']):.3f} / {float(x['peak_gpu_reserved_gib']):.3f} |")
parts += ['','### 三分支与五分支的逐种子比较','','| 种子 | 三分支 RankIC | 五分支 RankIC | 差值 | 三分支总分钟 | 五分支总分钟 |','|---|---:|---:|---:|---:|---:|']
for seed in sorted({x['seed'] for x in runs}):
    a=next(x for x in runs if x['seed']==seed and x['arm']=='baseline'); b=next(x for x in runs if x['seed']==seed and x['arm']=='branches5')
    parts.append(f"| {seed} | {float(a['rank_ic']):.6f} | {float(b['rank_ic']):.6f} | {float(b['rank_ic'])-float(a['rank_ic']):+.6f} | {float(a['elapsed_seconds'])/60:.2f} | {float(b['elapsed_seconds'])/60:.2f} |")
parts += ['','可采用的选型表述：**在相同三轮训练预算下，五分支没有带来可确认的排序收益，三个种子的主指标均较低，同时总耗时增加 6.3%、参数增加 30.5%、峰值已分配显存增加 30.2%；因此本轮没有足够理由承担增加到五分支的成本。**这不证明三分支优于所有更小结构，也不把五分支写成统计显著更差。','',
'### 之前不是要一小时吗','',
'旧 E0 日志的一次三轮运行耗时 60 分 38 秒，仍是 1,213 个训练日期和 217,953 个参数；这次基线训练阶段平均 29.86 分钟、含推理平均 31.45 分钟。两者属于不同运行，不能将新时间归因于少训练几轮，也不能把它写成架构带来两倍加速。当前采用新的数据打包路径；旧运行硬件与完整运行条件未对齐，尚不能分解时间变化的原因。','',
'## 辅助指标','',
'RankIC IR 使用非年化定义；五分位多空 Sharpe 按 sqrt(252) 年化，不扣交易成本；压力 ICIR 使用固定的 61 个压力日期。它们作为诊断，不能用来替换预先指定的主指标选胜者。','',
'| 配置 | RankIC IR | 五分位多空 Sharpe | 压力 ICIR |','|---|---:|---:|---:|']
for x in rows:
    parts.append(f"| {x['arm']} | {x['rank_ic_ir']:.4f} | {x['long_short_sharpe']:.4f} | {x['stress_ic_ir']:.4f} |")
parts += ['','## 结果文件与复核','',
'- [完整汇总 CSV]('+link+'results/summary.csv)、[逐次结果]('+link+'results/per_run.csv)、[逐种子配对差值]('+link+'results/paired_deltas.csv)。',
'- [训练时间汇总]('+link+'results/timing_summary.csv)、[逐次训练计时]('+link+'results/timing_per_run.csv)。',
'- [独立完成审计]('+link+'audits/completion_audit.json)：45 次运行、样本与评分键、初始化、源码和数据校验和、checkpoint 与因子校验和、每日指标、bootstrap 与 Holm 均通过。',
'- 逐日指标在结果目录的 runs/<配置>/<seed>/ 对应运行目录中；实际目录结构以产物为准。压缩的每日 RankIC 数据立方体见 [daily_rankic_cube.npz]('+link+'results/daily_rankic_cube.npz)。',
'- 代码、原始分钟数据、checkpoint 和完整因子留在 AutoDL；本地仅保留说明、小型汇总、manifest 与逐日指标。代码入口为 /root/autodl-tmp/projects/bigquant-default/experiments/finals_pre/tcn_architecture_o2c/。',
'- 所有性能结论只适用于本历史验证期、三轮训练预算及当前候选集合；不推断充分训练上限，不自动替换正式提交模型。','',
'## 实施协议与原始结构假设','',
'以下保留实施前的固定协议。其结构理由是提出候选的假设，不冒充已被实验验证；实际发现以上方结果为准。','']
old=(code/'README.md').read_text(); protocol=old[old.index('## 设计假设'):]
(code/'README.md').write_text('\n'.join(parts)+protocol)
assert all(arm in (code/'README.md').read_text() for arm in by)
print('README written', (code/'README.md').stat().st_size)
handoff=out/'handoff'; handoff.mkdir(exist_ok=True)
files=[code/'README.md',out/'status.json']
files += list((out/'results').glob('*'))
files += list((out/'audits').glob('*.json'))+list((out/'audits').glob('*.csv'))
files += list(out.glob('seed*_complete.json'))
for name in ('prepared.json','preflight.json'):
    p=out/'prepared'/name
    if p.exists(): files.append(p)
for name in ('daily_metrics.csv','metrics.json','manifest.json','status.json'): files+=list((out/'runs').rglob(name))
files=sorted(set(p for p in files if p.is_file() and p.suffix in ('.md','.csv','.json','.npz')))
assert len(list((out/'runs').rglob('daily_metrics.csv')))==45
inventory={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
receipt=handoff/'results_inventory.json'; receipt.write_text(json.dumps(inventory,indent=2)+'\n'); files.append(receipt)
with zipfile.ZipFile(handoff/'results_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p,str(p.relative_to(root)))
bundle=handoff/'results_bundle.zip'
print(json.dumps({'files':len(files),'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()}))