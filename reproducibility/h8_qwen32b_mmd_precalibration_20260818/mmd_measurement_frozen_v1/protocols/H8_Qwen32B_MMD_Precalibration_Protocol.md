# H8 / Qwen2.5-32B MMD 预校准协议（修订版）

状态：pre-sampling review。当前只允许 CPU 测试和 24 条 `smoke_only`；1200 条正式 Calibration 未获授权。

## 1. 研究边界

H8 在固定查询协议下检验目标端输出分布是否与指定 Qwen2.5-32B 版本一致。本阶段不研究 temperature、system prompt、推理后端、路由或模型版本变化；这些变化均视为协议外状态。本阶段也不运行攻击、不选择 Top-r/Max/Energy、不生成 held-out 数据。

M0 数据的唯一角色为 `mmd_precalibration_fit_only`。它只可用于拟合 feature scaler、每个 prompt 的 RBF bandwidth 和 intact-only 数值稳定性诊断，永远不得并入未来正式 Reference、held-out intact 或攻击 Target。

## 2. 冻结对象

- 模型：`Qwen/Qwen2.5-32B-Instruct`
- revision：`5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`
- 指纹：H6 归档的 MCC12，必须逐条核验 prompt 文本 SHA256
- dtype：BF16；attention：eager；3 张 GPU，各限 30 GiB
- 生成：batch=1，max input 512，max new 64，T=0.7，top-p=0.9，top-k=50，无 system prompt
- EOS：`[151645, 151643]`
- 每条响应使用全局唯一 seed；不得把同一 seed 作为一个 batch 的公共 seed

权威可执行参数位于 `configs/h8_qwen32b_mmd_precalibration.yaml`。

## 3. 采样和失败语义

Primary 固定 `batch=1`。每个 response ID 对应唯一 generation seed。技术异常只允许用原 seed 重试，最多两个 attempt；禁止换 seed 补样。每次失败与重试均记录 exception 类型、消息、attempt index 和 seed。

如果首个生成 token 是任一合法 EOS，则空字符串是有效模型响应，必须保留，并记录 `legal_first_token_eos=true`；不得将其误判为技术失败或另换 seed 补齐。

每条记录必须保存：prompt/response ID、prompt 文本与 SHA、seed、rendered prompt 与 SHA、input token IDs 与 SHA、completion token IDs 与 SHA、原始响应、stop reason、EOS ID、模型与 tokenizer revision、chat-template SHA、generation config SHA、运行时版本和 attempt 轨迹。

## 4. Feature schema 与 Primary scaling

特征提取器沿用 surface（11维）、BGE semantic（预期512维）和 task（5维），总维数预期528。运行时必须生成 feature schema，冻结每维 index、name、family、feature type、family slice、dtype 和 schema SHA；维数与预期不符直接 FAIL。

对 prompt j 的第 k 维，只用该 prompt 的 M0 fit-only 数据计算

\[
z^*_{j,k}=\frac{z_{j,k}-\mu_{j,k}}{s_{j,k}},\quad ddof=0.
\]

常量维度按类型处理：

- binary/bounded：尺度固定为1，标记 `unit_range_constant`；
- continuous：使用全部 prompt 的 M0 fit-only 数据在同一维上的 pooled std（ddof=0）；
- continuous 在 pooled 数据中仍为常量：直接 FAIL，禁止静默填 epsilon。

完成逐维缩放后，对每个 family f 再乘 `1/sqrt(d_f)`。因此 surface、semantic、task 三个 family 在欧氏距离中具有相同的先验总权重；这是 H8 Primary。所有计算固定 NumPy `float64`。

## 5. H8 bandwidth（禁止复用 H4）

对每个 prompt 的已缩放 M0 fit-only 特征，计算所有 a<b 的正欧氏距离：

\[
d_{ab}=\lVert z_a^*-z_b^*\rVert_2,\qquad
\sigma_j=\operatorname{median}\{d_{ab}:d_{ab}>0\}.
\]

这不是旧 H4 的 `sqrt(0.5 * median(squared distance))`。H8 代码与序列化工件必须写入精确方法名 `median_positive_pairwise_euclidean_distance`；loader 发现旧方法名必须 fail closed。

## 6. 两类 bandwidth stability test

每个 prompt 重复100次，从100条 M0 记录中无放回抽80条；PCG64 seed 固定。

1. `fixed_full_scaler`：保持完整100条拟合的 scaler，仅在80条子样本上重算 sigma。
2. `refit_subsample_scaler`：在各 prompt 同步抽取80%后，重新拟合 per-prompt 与 pooled fallback scaler，再重算 sigma。

均报告 `sigma_sub/sigma_full` 的 q05、median、q95、relative MAD 与 CV：

- q05>=0.80 且 q95<=1.20：PASS；
- 否则若 q05>=0.70 且 q95<=1.30：WARN，需要人工审查；
- 其余或非有限值：FAIL。

总 gate 取两类测试中较差等级。这是工程稳定性门，不是置信区间。

## 7. MMD 定义

RBF kernel：

\[
k(x,y)=\exp[-\lVert x-y\rVert^2/(2\sigma_j^2)].
\]

Primary 为允许 n!=m 的 generalized unbiased MMD²：

\[
\widehat{MMD}_u^2=
\frac{\sum_{i\ne i'}k(r_i,r_{i'})}{n(n-1)}+
\frac{\sum_{l\ne l'}k(q_l,q_{l'})}{m(m-1)}-
\frac{2\sum_{i,l}k(r_i,q_l)}{nm}.
\]

有限样本下该估计量可以为负，禁止截断为0。Biased MMD²只作 sensitivity。

## 8. 工件和 fail-closed 规则

feature schema、每个 prompt 的 scaler、bandwidth、stability 结果均使用 UTF-8、sorted-key、compact、禁止 NaN 的 canonical JSON，并保存 payload SHA256。Bandwidth 工件必须绑定 scaler payload SHA、feature schema SHA 与 M0 manifest SHA。正式 detector 只能加载冻结工件；类型、版本、hash、prompt ID、schema、data role 或 bandwidth convention 任一不一致即拒绝运行。

## 9. 当前获准的 smoke

只运行 `12 prompts × 2 unique seeds = 24` 条，batch=1，数据角色为 `smoke_only_not_calibration`。它们不属于 M0 Calibration，也不能进入任何后续统计数据集。

Smoke 的 PASS 条件：

- 24/24 请求完成且 seed 全局唯一；
- token IDs、stop reason、EOS、rendered prompt 和 provenance 字段完整；
- CPU 测试证明首 token EOS 空响应保留和 same-seed retry；
- 模型 revision、MCC12 文本 hash 与配置一致；
- 退出后无遗留 GPU compute worker。

完成后生成 preflight report 并停止。未获得下一次明确批准前，任何入口都必须拒绝1200条 Calibration。
