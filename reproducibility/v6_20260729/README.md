# V6 敏感提示词门控基线

本目录保存 V6 正式实验的精简、可版本化证据。完整原始结果约数百 MB，仅保存在实验服务器的 `results/`，不进入 Git。

## 文件

- `fixed_hybrid_calibration_v1.json`：12 条开发初始提示词拟合的固定微观/宏观校准。
- `joint_summary_60.json`：60 条内层候选生成摘要。
- `development_selection_report.json`：开发集外层冻结选择报告。
- `development_selected_accepted.jsonl`：开发集冻结的 22 条候选。
- `test_attack_manifest.jsonl`：冻结测试使用的预注册五家族均衡攻击清单。
- `test_manifest_design_report.json`：冻结测试清单设计说明。
- `final_test_report.json`：冻结测试总体门控报告。
- `final_test_validated_prompts.jsonl`：22 条冻结候选的逐条结果，其中 16 条严格通过。
- `checksums/`：原实验阶段生成的代码、配置、数据与设计校验和。

## 关键结论

- 内层 60/60 技术完成，32 条代理接受；
- 开发集冻结选择 22 条，超过要求的 12 条；
- 冻结测试严格通过 16 条，超过要求的 12 条；
- 开发选择未使用冻结测试结果；
- 当前结论只覆盖建立指纹前的敏感提示词生成与硬验收阶段。

原始服务器目录：

```text
/root/autodl-tmp/llm/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728/13_v6_formal_60
```
