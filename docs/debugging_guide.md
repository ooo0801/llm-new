# 调试指南

## CUDA OOM

先把 `batch_size` 改为 1、缩短 `max_input_tokens`，再降低激活候选池大小。微观敏感度使用 `max_parameter_tensors` 限制梯度张量。7B 梯度建议使用 48GB 显存。

## attention 返回 None

配置 `eager_attention: true`。某些 Transformers 版本使用 SDPA/Flash Attention 时不能返回权重，加载器会尝试切换 eager；如仍失败，记录版本并只使用 FFN/残差组件完成诊断。

## bitsandbytes 加载失败

确认 Linux、CUDA 和 bitsandbytes 版本兼容。0.5B/1.5B 上可先跳过 INT8/INT4，只运行剪枝与噪声，避免阻塞整条链路。

## MMD p 值总是 1

检查特征是否全部为常数、是否错误地重复同一响应，以及标准化后是否出现 NaN。小样本下增加提示词数量或每提示词重复数。

## 原始模型误报

固定 chat template 和 system prompt，增加参考响应数，检查 temperature/top-p。比较 pooled MMD 与 paired sign-flip，判断是否由不同提示词不满足交换性导致。

## MCC 不如 Top-k

先验证组件集合不是空集；绘制每类组件覆盖率；分别关闭 attention、FFN、residual 权重；检查高覆盖提示词是否牺牲了过多敏感度。MCC 只优化覆盖，不保证输出差异最大。

## 提示词优化产生乱码

减小 epsilon 和学习率，提高 semantic_weight，缩小 candidate_tokens，检查 PPL。不要把软提示优化结果直接纳入最终测试集，必须重新计算离散提示词敏感度。
