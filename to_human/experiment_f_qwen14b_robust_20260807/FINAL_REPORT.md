# Experiment F1：Qwen2.5-14B 多种子鲁棒选择结果

## 结论

F1 的全部开发端点均技术完成，但冻结的“每类别至少 3 个、来源唯一、共 30 个”选择规则只能得到 29 个候选，因此形成 **development No-Go**。这不是运行失败，而是预注册选择策略未达到进入独立确认的条件。

由于 F1 开发门槛未通过：

- 独立确认未运行；
- G1 稳定组件提取与 MCC12 未运行；
- H1 提示词分层 MMD 未运行。

上述三个阶段均是按条件协议停止，不应解释为模型端点失败。

## 技术完成情况

- 候选对：64
- 任务验证：128/128 条记录
- 完整参数块微扰评分：102/102 条去重记录，展开后 128/128
- 宏观攻击评分：1,530/1,530 条去重记录，展开后 1,920/1,920
- 攻击设计：5 个家族，每家族 3 个新变体，共 15 个变体
- 技术错误：0
- 技术门槛：通过

## 开发结果

- 两端任务均保持：63/64
- 微扰增益为正：60/64
- 等权宏观增益为正：39/64
- 满足开发资格：37/64
- 结构化剪枝不退化：42/64
- 描述性 legacy 保留：36/64
- 冻结选择器实际选出：29/30

29 个候选的类别分布：

- code：5
- instruction：5
- knowledge：3
- logic：2
- reasoning：4
- safety：6
- structured：3
- summary：1

失败原因是 logic 只能提供 2 个合格且来源唯一的候选，summary 只能提供 1 个，均低于冻结配额 3。选择器最终也因此少于规定的 30 个。

## 科学解释

多种子与结构化剪枝优先的排序明显扩大了描述性鲁棒候选池，但原 64 行组合中 logic 和 summary 的合格来源不足，导致类别覆盖约束成为新的瓶颈。结果不支持继续执行 H-F1 的独立确认流程。

实验结束后没有降低类别配额、补入候选、修改种子、调整攻击或改变阈值。若继续研究，应单独预注册 F2，在新实验中扩充 logic 与 summary 的构造来源，而不能回填或修复 F1。

## 证据与版本

- 机器可读总报告：`reproducibility/experiment_f_qwen14b_robust_20260807/FINAL_REPORT.json`
- 开发分析：`reproducibility/experiment_f_qwen14b_robust_20260807/01_development/analysis/report.json`
- 逐提示判定：`reproducibility/experiment_f_qwen14b_robust_20260807/01_development/analysis/per_prompt_decisions.jsonl`
- 冻结选择输出：`reproducibility/experiment_f_qwen14b_robust_20260807/01_development/analysis/frozen30_pairs.jsonl`（实际 29 行）
- 完整校验和：`reproducibility/experiment_f_qwen14b_robust_20260807/SHA256SUMS`
- 结果证据提交：`9fe8ac7`

本报告的结论边界是固定 Qwen2.5-14B revision、冻结候选池、冻结开发变体与冻结选择规则；不外推到任意未来模型或攻击实例。
