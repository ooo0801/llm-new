# 小模型实验版本管理说明

## 来源与分支边界

本分支从 `experiment/h9-stage1-sensitivity-propagation` 的
`a3e2f592f5bedbdace77256627f0fa767971307c` 派生。2026-09-08 通过 SSH
读取服务器 `/root/autodl-tmp/llm` 及五份执行目录，下载限定源码与汇总证据，
对 1,314 个文件验证传输前后的 SHA256。服务器原目录没有 `.git`。

源码以 `stage2_pilot_execution_20260905` 最新版本优先，R2 v2 与基础目录
补齐缺少的阶段文件，不把不同版本的同名文件重复放入 Python 模块目录。
67 个导入源码/配置/协议文件中，64 个来自服务器，以下 3 个仅在本地分析工作区发现：

- `scripts/analyze_stage2_existing_data.py`
- `scripts/audit_stage2_saved_search.py`
- `experiments/h9-stage1/R1_PROTOCOL.md`

逐文件来源已在 `SOURCE_PROVENANCE.json` 中区别记录。未导入本地旧研究状态覆盖文件，
未包含与实验无关的 Codex 聊天修复代码。原 32B 分支、H9 分支和 `main` 不修改。

## 当前实验到底进行到哪里

| 阶段 | 本分支保存的内容 | 解释边界 |
|---|---|---|
| Stage1 / R1 / R2 | 攻击统计校准、旧数据测量修复、独立负例验证实现与协议 | 本次是代码归档，不重新认定历史实验通过条件 |
| Stage2 初版 | 增强 H6、严格任务验收、六个内部诊断指标 | 不是最新三代理试验的有效参数入口 |
| 攻击重新校准 | 32 条独立效用题、三强度/三 seed 的 Gaussian 和 LoRA、温和覆盖攻击 | 效用合格不等于可被指纹检出 |
| 三代理 v3 pilot | JS、Top-K continuous、raw-logit L2，扩大提示池和放松搜索语义约束 | 已完成 pilot；没有代理达到晋级门槛 |
| 后续正式搜索 / held-out | 代码中保留执行路径 | 未运行，不能作为已有结果 |

最新模型为 `Qwen/Qwen2.5-1.5B-Instruct`，revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，BF16。
历史 Stage1 模型是 Qwen2.5-0.5B。

最新三代理参数以归档的 `evidence/stage2_three_proxy_ceiling_v3_20260906/PLAN.json`
为准，不以较早 `configs/stage2_qwen15b.json` 代替。提示池 144 条，pilot 搜索
每代理 16 个来源。搜索允许放松 PPL、表面语义与任务保持约束，行为侧仍要求 intact 稳定性。
pilot 每提示/端点 4 次响应，使用中强度 Gaussian/LoRA 的两个 seed；这不等同于
原定正式验证的三强度、三 seed。六个响应 bank 的计划总量为 1,368 条。

晋级要求每个代理在 Gaussian 和 LoRA 两族分别至少有 2 条稳定敏感提示，
两 seed 平均文本 TV 门槛为 0.25。冻结 `PROMOTION.json` 的结论是
`STOP_NO_MEASURABLE_GAUSSIAN_LORA_SIGNAL`，`promoted_proxies=[]`。
该状态名表示未达到规则，不是所有提示的效应严格为零。

| 代理 | Gaussian 敏感 / 计入提示 | LoRA 敏感 / 计入提示 | 晋级 |
|---|---:|---:|---|
| JS | 1 / 4 | 2 / 4 | 否 |
| Top-K continuous | 1 / 3 | 1 / 3 | 否 |
| raw-logit L2 | 2 / 6 | 1 / 6 | 否 |

表内分母是晋级汇总计入的提示数量，不是完整池大小，也不是正式公平排序样本量。
不能据此声称 JS 显著优于 raw，亦不能宣称 H6 行为确认问题已解决。

## 检查与复现

只核验归档，不调用模型：

```bash
python scripts/verify_small_model_snapshot.py
```

CPU 单元测试：

```bash
python -m pytest tests/test_macro_proxy.py tests/test_stage1_r1.py tests/test_stage1_r2.py tests/test_stage1_r2_small.py tests/test_stage2_contract.py tests/test_stage2_search_parameters.py tests/test_stage2_wording_v2.py tests/test_stage2_pilot.py tests/test_stage2_evaluation.py tests/test_h9_stage1_attack_stat_calibration.py
```

Python / PyTorch 等依赖声明在 `pyproject.toml`；实际 pilot 的包版本在行为
`PLAN.json` 的 `packages` 字段。本次没有更新服务器环境或重新安装依赖。
需要新环境时可使用 `python -m pip install -e '.[dev,semantic,lora,quantization]'`，
但这仅安装声明范围，不保证等同冻结运行环境；严谨复现还需对照冻结版本。

GitHub 保存源码、设计和精简证据，不保存权重、LoRA adapter、完整响应 bank、
语义特征 bank 或数百 MB checkpoint。仅凭克隆不能重新计算全部历史 MMD。
本服务器上完整运行数据仍在原执行目录，归档中的哈希用于核对外部数据。
历史 `.sh` 有绝对路径，v3 驱动还依赖 v1/v2 的冻结产物，不能把它当成全新目录
一条命令从零运行的通用入口。重新实验必须另定输出目录和冻结计划。

## 操作警告与后续工作流

- **不要运行 `scripts/run_stage2_calibration_then_behavior.sh` 作为默认入口**：
  它是历史归档脚本，退出 trap 内包含自动关机。归档保留原样用于来源核验，
  本次未执行。后续要复用时应先在新提交中移除该行为；用户当前不允许保护程序擅自关机。
- 当前 v3 launcher 不含关机命令，但仍会启动昂贵实验并写历史结果，不能用于版本检查。
- Git 管理目录和旧运行目录分开；不向运行中的目录直接 checkout，不复制旧目录的
  `results/`、模型或缓存到 Git 暂存区。
- 后续修改在本分支或其子分支进行。每次运行记录 commit、dirty 状态、冻结参数、
  模型 revision 和数据哈希。完成后提交代码/配置，再单独提交精选证据。
- 不使用 `git add -f results/`，不 force-push，不复用已经看过的 held-out 数据宣称独立验证。

测试及已知历史失败见归档目录 README；本次发布是研究快照，不是已完成的算法修复。
