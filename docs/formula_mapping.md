# 申请书公式与代码映射

| 申请书 | 工程化定义 | 实现位置 |
|---|---|---|
| 式 (2) 参数敏感度 | 原始与修改模型的输出距离 | `distances.py` |
| 式 (7) 微观敏感度 | 选定参数子集上的伪目标 NLL 梯度平方范数 | `sensitivity.py::micro_sensitivity_one` |
| 式 (8)(9) 宏观敏感度 | 多种修改模型 next-token logits 距离均值 | `scripts/score_prompts.py` |
| 式 (10)(11) 综合敏感度 | robust scaling 后的自适应加权 | `sensitivity.py::combine_hybrid` |
| 式 (13)-(17) 嵌入优化 | 局部候选 token 软组合、退火、L2 投影和离散映射 | `prompt_optimization.py` |
| 式 (24) 注意力熵 | 每层每头在 query/token 维度上的平均熵 | `activations.py` |
| 式 (25)-(27) MCC | 注意力、FFN、残差组件的带权贪心最大覆盖 | `mcc.py` |
| 式 (28)(29) MMD | 无偏 RBF-MMD | `statistics.py::mmd2_unbiased` |
| 式 (30) 置换检验 | 使用 `(count+1)/(M+1)` 修正的置换 p 值 | `statistics.py::mmd_permutation_test` |
| 式 (31)(32) 功效 | 正态近似功效与最小样本量 | `statistics.py` |

## 必要的工程化偏离

申请书没有定义 LLM 输出 `fθ(x)`。本项目默认用相同提示上下文最后一个有效位置的 logits，并用 Jensen-Shannon divergence 比较。最终黑盒验证只使用生成文本特征。

完整 Jacobian 在实际 LLM 上不可承受，因此微观敏感度使用一个可微标量目标并限制参数子集。该定义保留“输入导致参数梯度放大”的核心思想，但不是对申请书未定义 Jacobian 的逐元素构造。

申请书把不同提示词的响应视为同分布样本。项目既保留 pooled MMD，也提供同提示词配对的 sign-flip permutation，后者用于检验统计假设是否影响结论。
