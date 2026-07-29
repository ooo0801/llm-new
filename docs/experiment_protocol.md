# 实验协议

## 研究问题

1. 高敏感度提示词是否比随机提示词更容易检测模型修改？
2. MCC 是否在同样提示词数量下提高组件覆盖率和检测率？
3. 多层次特征和配对统计检验是否降低随机生成导致的误报？
4. 方法对模型替换、量化、剪枝、扰动和 LoRA 的效果如何？

## 默认模型

- 调试：Qwen2.5-0.5B-Instruct。
- 主实验：Qwen2.5-1.5B-Instruct。
- 扩展：Qwen2.5-7B-Instruct。
- 论文对齐：Llama2-7B（需要单独接受 Meta 许可）。

必须记录模型仓库、revision、权重 dtype、Transformers 版本、chat template 和生成参数。

## 提示词划分

candidate 只用于指纹生成；validation 用于选择阈值；test 只用于最终报告。不得根据 test 结果重新选择提示词。

## 基线和方法

- Random：固定种子随机选择。
- Top-Sensitivity：按 macro 或 hybrid 排序。
- MCC：在高敏感候选池上最大化加权新增覆盖。

## 实验单位

一次验证 trial 包含 `k` 个提示词、每条提示词 `r` 次响应和一次完整统计判定。对 intact 和每种修改至少重复 50 个 trial，变化随机种子或提示词子集。最终报告 Accuracy、Precision、Recall、F1、FPR、API 次数和时间。

## 公平性控制

原始与待测模型使用相同 tokenizer、chat template、system prompt 和解码配置。把解码配置变化作为单独的良性干扰实验，不应混入参数攻击结果。
