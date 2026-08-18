# H8 / Qwen2.5-32B M0-G/H 离线 scaler 与 bandwidth 协议

状态：M0-G/H offline analysis authorized。M0-F 的1200条
`mmd_precalibration_fit_only` 响应已经完成并冻结。本阶段禁止任何新的模型采样，只允许以这
1200条响应为唯一输入执行 deterministic feature extraction、Feature QA、Primary scaler
candidate 和 bandwidth candidate/stability 计算。

## 1. 研究边界

H8 在固定查询协议下检验目标端输出分布是否与指定 Qwen2.5-32B 版本一致。本阶段不研究
temperature、system prompt、推理后端、路由或模型版本变化；这些变化均视为协议外状态。
M0-G/H 不运行攻击、不选择 Top-r/Max/Energy、不生成 Reference 或 held-out 数据，也不执行
MMD pseudo trials 或 D→S 拟合。

M0 数据的唯一角色为 `mmd_precalibration_fit_only`。它只可用于拟合 feature scaler、每个
prompt 的 RBF bandwidth 和 intact-only 数值稳定性诊断，永远不得并入未来正式 Reference、
held-out intact 或攻击 Target。

## 2. 冻结输入

- 模型响应：M0-F 已冻结 responses，SHA256 必须与授权配置一致；
- 指纹：H6 MCC12，每个 prompt 必须恰有100条响应；
- 数据总量：1200，seed、response ID 全局唯一，schedule 必须逐位置匹配；
- 每条记录必须为 `mmd_precalibration_fit_only`，且 formal Reference、held-out、attack 资格均为 false；
- M0-F sampling report 必须为 PASS，manifest 文件和 canonical payload hash 必须匹配。

任何一项不一致立即 NO-GO。脚本不得扫描或读取其他响应目录。

## 3. Feature extraction 与 QA

特征提取器固定为 surface（11维）、BGE semantic（512维）和 task（5维），总维数528。运行时
生成 feature schema，冻结每维 index、name、family、feature type、family slice、dtype 和
schema SHA。全部计算使用 NumPy `float64`。

每个 fingerprint 报告 raw-response unique count、completion-token-sequence unique count、
feature-vector unique count、三类 feature dimensions、每个 family 的 exact-constant 与
near-zero-nonconstant dimension 数、NaN/Inf、EOS/length-stop 比例及 token-length 分布。

## 4. Primary family-balanced scaler

对 prompt j 的第 k 维，只用该 prompt 的 M0 fit-only 数据计算

\[
z^*_{j,k}=\frac{z_{j,k}-\mu_{j,k}}{s_{j,k}},\quad ddof=0.
\]

常量维度按类型处理：

- binary/bounded：尺度固定为1，标记 `unit_range_constant`；
- continuous：使用全部 prompt 的 M0 fit-only 数据在同一维上的 pooled std（ddof=0）；
- continuous 在 pooled 数据中仍退化：写入唯一 global exclusion mask，并在所有 prompt scaler
  中将该维变换结果固定为0；禁止静默填 epsilon，也禁止 per-prompt mask。

完成逐维缩放后，对 family f 乘 `1/sqrt(d_f_active)`。`d_f_active` 是该 family 未被 global
exclusion mask 排除的维数。binary/bounded 常量维仍为 active，但中心化结果为0。任何 family
的 active dimension 为0均为 FAIL。

## 5. Prompt-specific bandwidth 与 degeneracy fallback

对每个 prompt 的已缩放 M0 fit-only 特征，只计算所有 a<b 的 within-prompt 欧氏距离：

\[
d_{ab}=\lVert z_a^*-z_b^*\rVert_2.
\]

若至少存在一个正距离：

\[
\sigma_j=\operatorname{median}\{d_{ab}:d_{ab}>0\}.
\]

该 prompt 必须保持 `bandwidth_source=prompt_specific`。即使其 stability 为 FAIL，也禁止改用
global fallback，必须停止审查。

若且仅若某 fingerprint 的全部 within-prompt distances 均为0，标记结构性
`feature_degenerate`，并使用

\[
\sigma_{global}=\operatorname{median}\left(\bigcup_{j\in J_+}
\{d^{(j)}_{ab}:d^{(j)}_{ab}>0\}\right),
\]

其中 `J_+` 只含存在正 within-prompt distance 的非退化 fingerprint。禁止 cross-prompt
distance，禁止 Target、attack、formal Reference 或 held-out 数据。若所有 fingerprint 均退化，
则 FAIL。fallback 工件写 `bandwidth_source=global_degenerate_fallback`。

H8 方法名固定为 `median_positive_pairwise_euclidean_distance`；禁止旧 H4 的
`sqrt(0.5 * median(squared distance))`。

## 6. 两类 bandwidth stability

固定100次重复，每次从每个 prompt 的100条记录中同步无放回抽80条，PCG64 seed 固定。

1. `fixed_full_scaler`：保持完整1200条拟合的 scaler/mask，只在子样本上重算 bandwidth；
2. `refit_subsample_scaler`：用同步子样本重新拟合 pooled mask 与各 prompt scaler，再重算 bandwidth。

对 `sigma_global`，每次仍只汇总非退化 fingerprint 的正 within-prompt distances，禁止
cross-prompt distance。报告 `sigma_sub/sigma_full` 的 q05、median、q95、relative MAD 和 CV：

- q05>=0.80 且 q95<=1.20：PASS；
- 否则若 q05>=0.70 且 q95<=1.30：WARN；
- 其余或非有限值：FAIL。

总 gate 取 A/B 较差等级。这是工程稳定性门，不是置信区间。Global fallback candidate 自身也
必须执行 A/B stability。

## 7. 工件与 fail-closed 规则

feature schema、global exclusion mask、12个 scaler candidates、12个 bandwidth candidates、
global bandwidth candidate、Feature QA 与 stability report 均使用 UTF-8、sorted-key、compact、
禁止 NaN 的 canonical JSON 并保存 SHA256。二进制 float64 feature matrix 和 row index 另存并
哈希。Bandwidth 必须绑定 scaler payload SHA、feature schema SHA、global exclusion mask SHA 和
M0 manifest SHA。

所有输出均标记 `candidate`。在后续明确批准冻结前，不得作为正式 detector 参数。类型、版本、
hash、prompt ID、schema、data role、bandwidth source 或 convention 任一不一致即拒绝加载。

## 8. Provenance 和停止边界

M0-F 的 `sampling_code_commit`、`sampling_execution_commit` 与 `sampling_archive_commit` 分别记录，
禁止混写。旧 `sampling_progress.json` 若保持 `status=running`，只能标记为 transient snapshot；
最终完成状态以 M0-F PASS report 和版本化 archive 为准。

本轮生成 `H8_M0GH_SCALER_BANDWIDTH_REPORT.json` 后立即停止。禁止 MMD pseudo trials、D→S、
Top-r、global test、Reference/held-out/attack 读取或生成以及任何模型采样。
