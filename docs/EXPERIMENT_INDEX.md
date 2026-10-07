# 实验索引

按方法路线和研究阶段查阅；“历史归档”不代表本次重新审计通过。

| 阶段 | 方法/模型 | 已有证据与边界 | 仓库入口 |
|---|---|---|---|
| V6、V1/V2 | 早期7B构造及文本特征检测 | 历史闭环；检测规则与当前路线不同 | [V6](../reproducibility/v6_20260729)、[V1](../reproducibility/fingerprint_v1_20260730)、[V2](../reproducibility/fingerprint_v2_global_20260731) |
| 14B A～H5 | 迁移、构造、鲁棒性、MCC及配对检测 | 含通过与失败阶段，按原报告解释 | [14B A](../reproducibility/experiment_a_qwen14b_20260805)、[H5](../reproducibility/fingerprint_h5_qwen14b_mismatch_20260808) |
| 32B H6/H8 | 32B重构、校准、开发和旧检测确认 | 历史32B已做；不是新首token协议的32B结果 | [H6](../reproducibility/fingerprint_h6_qwen32b_mcc_20260812)、[H8校准](../reproducibility/h8_qwen32b_mmd_precalibration_20260818)、[H8最终归档](../reproducibility/h8_qwen32b_final_confirmation_20260823) |
| 小模型Stage1/2 | 0.5B/1.5B探索 | 旧Stage2三代理pilot未达冻结晋级条件 | [阶段说明](SMALL_MODEL_STAGE2.md)、[来源证据](../reproducibility/small-model-stage2-20260908) |
| fresh-v1～v4 | 首token机制试验 | v1～v3失败修订；v4技术完成，效用边界仍有限 | [记录与结果](../reproducibility/research_progress_20261007/first_token_pilot) |
| scaling-v1 | 首token强度×种子矩阵 | 当前固定面板结果，不是任意预算保证 | [报告](../reproducibility/research_progress_20261007/scaling_validation_20260919/实验结果报告.md) |
| prompt-budget-v1 | 提示词数量/采样预算 | 含历史跨来源比较，应结合后续同源实验 | [报告](../reproducibility/research_progress_20261007/prompt_budget_20260919/实验结果报告.md) |
| paired-prompt-budget-v2 | 同源原文与JS/Top-K/logit | v1迁移参考检查失败，v2重采完成；非新攻击独立确认 | [报告](../reproducibility/research_progress_20261007/paired_prompt_budget/实验结果报告.md) |
| generation-ablation-v1 | 原文/微观/JS/联合 | 未支持微观在JS上有稳定增量收益 | [报告](../reproducibility/research_progress_20261007/generation_ablation/实验结果报告.md) |
| calibration-scale-v1 | 校准尺度比较 | 阶段性校准结果 | [报告](../reproducibility/research_progress_20261007/calibration_scale_comparison/校准尺度计算报告.md) |
| calibration-7b-80-v1 | 7B八十条校准 | 历史校准版本 | [报告](../reproducibility/research_progress_20261007/calibration_7b_80/7B校准尺度计算报告.md) |
| calibration-7b-100-geo-v1 | 7B百条分层几何校准 | 当前7B主实验依赖；不证明未见模板泛化 | [报告](../reproducibility/research_progress_20261007/calibration_7b_100_geo/7B百条几何平均校准报告.md) |
| fingerprint-7b-mcc-vs-top-v1 | 7B首token检测 | 80来源、61候选；MCC/Top各4/8条、12配置 | [核心证据](../reproducibility/7b_mcc_vs_top_20261003) |
| fingerprint-7b-paired-original-v1 | 7B同已选来源原文对照 | 后验配对诊断；不重选面板，不独立训练新攻击 | [核心证据](../reproducibility/7b_paired_original_20261004) |

## 代码入口

- 首token统计：[resf_token.py](../src/llm_integrity/resf_token.py)。
- 预算与配对：[run_prompt_budget.py](../scripts/run_prompt_budget.py)、[run_paired_prompt_budget.py](../scripts/run_paired_prompt_budget.py)、[evaluate_prompt_budget.py](../scripts/evaluate_prompt_budget.py)。
- 消融：[run_generation_ablation.py](../scripts/run_generation_ablation.py)。
- 校准：[run_calibration_7b_100_geo.py](../scripts/run_calibration_7b_100_geo.py)。
- 7B生成和比较：[run_7b_fingerprint_comparison.py](../scripts/run_7b_fingerprint_comparison.py)。运行范围以冻结的 `SCOPE_80.json` 为准，不直接启动原300来源循环。
- 原文对照执行脚本：见[原文配对证据包](../reproducibility/7b_paired_original_20261004)中的 `execution/`。
- 已有数据的CPU分析：[analysis/README.md](../analysis/README.md)。部分脚本需要仓库外完整备份，不是空服务器一键运行入口。
