# 小模型最新实验与本地分析

本次发布包含提示词预算实验、同源配对实验、微观/宏观消融，以及仅接受改写（accepted-only）面板分析的代码。原始实验数据、模型权重、LoRA适配器和服务器凭据不随代码上传。

## 实验入口

在仓库根目录、原实验Python环境中运行：

```bash
python scripts/run_prompt_budget.py prepare --root RUN_DIR --base SCALING_RUN --pilot PILOT_RUN
python scripts/run_prompt_budget.py search --root RUN_DIR --base SCALING_RUN --pilot PILOT_RUN
python scripts/evaluate_prompt_budget.py --help
python scripts/run_paired_prompt_budget.py --root PAIRED_RUN --old RUN_DIR
python scripts/run_generation_ablation.py --root ABLATION_RUN --previous RUN_DIR
```

这些是依赖既有冻结实验资产的入口，不是自动下载模型的空服务器安装程序。`SCALING_RUN`需要模型与攻击配置，`PILOT_RUN`需要搜索/开发攻击、适配器和校准记录。检查各入口的`--help`及相应PLAN；模型、适配器与依赖路径须在本地存在。

- 预算实验：生成敏感提示词并用开发集排序。
- 同源配对：按同一来源映射原文与三种宏观代理的优化产物；失败回退原文；独立开发排序为补充。
- 消融：ordinary、micro_only、macro_only、joint_js。仅微观跳过宏观计算与门槛；仅宏观跳过微观计算；各组使用相同来源与生成种子。
- 正式预算矩阵：同源与消融为m=1/2/5/10/20，n=25/50/100；旧预算脚本仍保留其历史n=10配置，不能与新协议混为一谈。
- 这些流程支持已有文件断点恢复；不应把技术失败当优化失败，不应根据测试结果重选候选。阶段缓存不是完整配置迁移机制，变更实验设置请使用新目录。

## 本地审计和报告

分析脚本统一要求`--data-dir`，指定此前完整备份目录。需要Python、NumPy；绘图另需Matplotlib。原始备份目录结构及文件名保持不变。

```bash
python analysis/prompt_budget/audit_results.py --data-dir /data/prompt_budget_20260919
python analysis/prompt_budget/gen_fig_budget.py --data-dir /data/prompt_budget_20260919

python analysis/paired_prompt_budget/audit_local.py --data-dir /data/paired_prompt_budget
python analysis/paired_prompt_budget/gen_fig_paired.py --data-dir /data/paired_prompt_budget
python analysis/paired_prompt_budget/accepted_only_analysis.py --data-dir /data/paired_prompt_budget

python analysis/generation_ablation/verify_extract.py --data-dir /data/generation_ablation
python analysis/generation_ablation/audit_local.py --data-dir /data/generation_ablation
python analysis/generation_ablation/figures/gen_fig_ablation.py --data-dir /data/generation_ablation
```

提示词预算审计使用该次冻结归档的固定哈希；同源配对审计还需要相邻的`prompt_budget_20260919/prompt-budget-v1/search/`搜索记录。消融审计从备份中的代码快照导入检测代码，并要求完整归档、ARCHIVE.json及解压目录。绘图与报告脚本针对已完成实验，包含该次结果解释，不是适用于任意新实验的通用结论生成器。

Accepted-only分析仅从接受改写来源枚举所有合法组合，与同来源、同顺序原文配对；保留实际m对应的判定阈值。输出`accepted_only/`中的Markdown报告、CSV明细和审计记录，不新增模型查询，也不按测试结果筛选候选。三种方法的接受池不同，另报告共同接受来源对照。

## 检查与研究边界

```bash
python -m pytest -q tests/test_prompt_budget.py tests/test_paired_prompt_budget.py tests/test_generation_ablation.py
```

发布时11项相关测试通过；已完成的服务器实验及本地结果审计见本地归档。本次代码发布不代表已完成新种子确认，也不包含尚在讨论中的新MCC接入、自适应权重或扩大校准提示词集的实现。不要将旧攻击上的探索性结果解释为独立确认。
