# LLM Integrity Fingerprint

研究主题：利用敏感提示词检验开源大语言模型是否被修改。

**当前进度：已完成7B首token检测、MCC/Top指纹比较及同来源原文对照。** 本次整理日期为2026-10-07，最新纳入实验完成于2026-10-04；整理没有启动新实验。

## 从这里阅读

1. [当前研究进度与结论边界](docs/CURRENT_STATUS.md)
2. [全部研究阶段与代码入口](docs/EXPERIMENT_INDEX.md)
3. [数据备份、依赖与恢复位置](docs/DATA_LOCATIONS.md)
4. [本次导入来源和逐文件SHA256](docs/IMPORT_PROVENANCE_20261007.json)

AI阅读指引见 [AGENTS.md](AGENTS.md)，机器可读当前状态见 [project-status.json](project-status.json)。

## 最新结果

在Qwen2.5-7B-Instruct上，使用同样4条提示词、每条25次首token查询：

| 指纹选择 | 敏感提示词检出攻击 | 对应原文检出攻击 | 敏感/原文正常误报 |
|---|---:|---:|---:|
| Top敏感度 | 40/40 | 17/40 | 0/20、0/20 |
| MCC | 37/40 | 22/40 | 0/20、0/20 |

这支持当前已选来源上的改写收益；原文对照是后验配对诊断，不是全新独立确认。20个正常面板来自一个模型的重复采样，40/40与0/20不是总体性能保证。

- [7B敏感指纹比较报告](reproducibility/7b_mcc_vs_top_20261003/7B指纹选择与检测实验报告.md)
- [7B敏感与同来源原文对照报告](reproducibility/7b_paired_original_20261004/7B敏感指纹与同源原文检测对照报告.md)
- [小模型配对、消融和校准过程](reproducibility/research_progress_20261007/README.md)

当前没有证明MCC带来额外检测优势，也没有证明微观项相对仅JS有稳定增量贡献。当前首token协议尚未完成32B验证或未知攻击家族验证。

## 历史路线仍然保留

此前已经开展14B、32B及旧文本特征/MMD等检测实验；见[实验索引](docs/EXPERIMENT_INDEX.md)。这些历史结果与当前首token方法分开记录，不能把“旧32B实验完成”等同于“新方法32B验证完成”。

`docs/history/`保存旧首页、交接和研究清单。历史日志中的“当前”“正在运行”“关机”等只代表记录当时，不是当前服务器状态。根目录 `PROJECT_MANIFEST.json` 是早期快照，不是最新全仓库清单。

## 目录与运行边界

| 目录 | 用途 |
|---|---|
| `src/`、`scripts/` | 算法实现、实验入口 |
| `configs/`、`tests/` | 配置与检查 |
| `analysis/` | 已有数据的审计、统计和绘图 |
| `reproducibility/` | 历史与当前实验核心证据 |
| `docs/` | 当前进度、实验索引和历史说明 |

代码与核心证据可通过Git同步。原始响应、模型、适配器和大归档另行备份。克隆不等于恢复完整运行环境；部分执行脚本依赖历史冻结目录与资产，详见各证据包说明。不要直接启动7B历史300来源循环，正式范围以 `SCOPE_80.json` 为准。

在已安装项目依赖的环境中，以下是CPU证据检查与相关测试，不会启动模型实验：

```bash
python reproducibility/7b_mcc_vs_top_20261003/verify_core_results.py
python -m pytest tests/test_scaling_matrix_resume.py tests/test_prompt_budget.py tests/test_paired_prompt_budget.py tests/test_generation_ablation.py -q
```

这些检查不代替原始响应全量审计或GPU独立复现。最新本地整理验证见 [LOCAL_VALIDATION_20261007.json](docs/LOCAL_VALIDATION_20261007.json)。
