# 数据与恢复位置

Git仓库记录代码、配置、精简证据、报告和索引。完整原始响应、模型和适配器通过独立备份保存，克隆仓库不会自动恢复这些依赖。

本地原工作区：`C:/Users/22489/Documents/shumo`。服务器历史数据根：`/root/autodl-tmp/token-integrity`。服务器地址和端口会变，不把旧端口作为实验身份。

| 实验 | 本地相对shumo位置 | 服务器相对数据根位置 |
|---|---|---|
| 7B敏感检测 | `work/fingerprint_7b_mcc_vs_top/detection80_backup/runs/fingerprint-7b-mcc-vs-top-v1` | `runs/fingerprint-7b-mcc-vs-top-v1` |
| 7B原文配对 | `work/fingerprint_7b_mcc_vs_top/paired_original_backup/runs/fingerprint-7b-paired-original-v1` | `runs/fingerprint-7b-paired-original-v1` |
| 7B百条校准 | `work/calibration_7b_100_geo/calibration-7b-100-geo-v1` | `runs/calibration-7b-100-geo-v1` |
| 7B八十条校准 | `work/calibration_7b_80/calibration-7b-80-v1` | `runs/calibration-7b-80-v1` |
| 尺度比较 | `work/calibration_scale_comparison/calibration-scale-v1` | `runs/calibration-scale-v1` |
| 同源配对 | `work/paired_prompt_budget/paired-prompt-budget-v2` | `runs/paired-prompt-budget-v2` |
| 消融 | `work/generation_ablation/generation-ablation-v1` | `runs/generation-ablation-v1` |
| 提示词预算 | `work/prompt_budget_20260919/prompt-budget-v1` | `runs/prompt-budget-v1` |
| 强度矩阵 | `work/scaling_validation_20260919` | `runs/scaling-v1` |
| 首token试验 | `work/first_token_pilot_20260917` | `runs/fresh-v1`～`runs/fresh-v4` |

最新归档位于本地 `work/fingerprint_7b_mcc_vs_top/`：

- `fingerprint-detection80-final.tar.gz`：1,614,760,979字节，SHA256 `7d7b158316a0b8918579a09b89b735bb0f2c5ec60982c0331bd26d866c9cd996`。
- `fingerprint-paired-original-v1.tar.gz`：609,746字节，SHA256 `fc847e8df8367c823e7649fa8cd846768f16241b21c576761ffce421075ab708`。

原文归档依赖父敏感实验已有模型/适配器，不是独立完整环境。校准适配器也需恢复。本次核验新导入文件与本地来源一致，没有再次下载这些大归档或重新计算GPU输出。

## 文档适用范围

- `docs/history/`中的交接、原研究清单和原首页保留历史原文，其相对路径通常以原shumo目录为基准。
- 证据包里的 `PUBLICATION*.json` 记录的是旧仓库发布，不代表本次内容已推送到 `llm-new`。
- 完成、审计、关机记录是历史证据，不能判断当前服务器状态。
- 根目录 `PROJECT_MANIFEST.json` 是早期文件快照，不能作为当前全仓库文件清单；本次导入见 `IMPORT_PROVENANCE_20261007.json`。

## 后续新增实验

每次使用独立运行编号，记录代码提交、工作区是否干净、模型revision、配置与种子、依赖实验、输出目录及归档哈希。若运行时存在未提交改动，另存补丁/代码快照；不以事后发布提交冒充运行版本。
