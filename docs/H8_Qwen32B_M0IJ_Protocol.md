# H8 / Qwen2.5-32B M0-I/J MMD 数值与置换预校验协议

状态：M0-I/J offline numerical validation authorized。禁止任何新模型采样，禁止读取或生成
formal Reference、held-out、attack 数据。本阶段唯一输入为 M0-G/H PASS 后已哈希的 feature
schema、global exclusion mask、12个 family-balanced scaler candidates、12个 bandwidth
candidates、global fallback bandwidth、1200×528 feature matrix 与 row index。

本阶段开始后，禁止依据 pseudo-MMD 结果修改 feature schema、exclusion mask、scaler、family
weight、bandwidth 或 stability gate。

## 1. Intact-only pseudo-MMD

对每个 fingerprint 固定四种 `(N_R,N_Q)`：`(40,10)`、`(40,20)`、`(60,10)`、
`(60,20)`。每种结构执行200次 deterministic PCG64 resampling。每次从该 prompt 的100条 M0
fit-only features 中无放回抽 `N_R+N_Q` 条，按冻结顺序划为 pseudo-reference 与
pseudo-target。

Primary 为 generalized unequal-size unbiased MMD²，允许并保留负值；sensitivity 为 biased
MMD²。trial 内禁止 refit scaler、bandwidth 或 schema。每个 prompt×structure 报告 mean、
median、std(ddof=0)、MAD、q05、q95、min、max、negative fraction、biased summary 和
NaN/Inf count。

对 feature-degenerate prompt，所有 transformed feature 必须完全相同，且每个 trial 满足
`abs(unbiased_MMD²)<=1e-12`、`abs(biased_MMD²)<=1e-12`。失败即 FAIL，不得通过改 σ 规避。
本阶段不依据 pseudo-MMD 选择样本量，报告固定
`sample_size_selection_performed=false`。

## 2. Representative permutation sanity

每个 prompt×structure 在 MMD 计算前冻结一个 representative pseudo split，共48个。每个固定
`B_precheck=999`，保持原 `N_R:N_Q`，只随机置换 group label。observed statistic 独立计算一次，
不得放入 permutation loop；经验值固定为

\[
p=\frac{1+\#\{D_{perm}\ge D_{obs}\}}{1000}.
\]

验证 group size、PCG64 seed reproducibility、`0<p<=1`、无 p=0、999个 permutation evaluation、
observed evaluation 恰为1、候选工件 hash 不变及 permutation 内无 refit。对 feature-degenerate
prompt，observed/permuted unbiased 与 biased MMD 均须在 `1e-12` 内为0，p 必须为1。

## 3. 回归 gate

必须验证：MMD symmetry、unequal-size implementation 与直接定义一致、unbiased negative value
保留、所有 candidate file/payload hash、artifact binding 与 fail-closed tamper detection。任何
non-finite、数值 invariant、reproducibility、hash 或 loader gate FAIL，均不得提升 candidate。

Structured prompt 的既有 stability q05/q95 已 PASS；本阶段只记录其较低 subsample ratio 和
pseudo-MMD min/max，不允许修改 gate。

## 4. PASS 后冻结边界

仅当所有 gate PASS，生成 `H8_MMD_PRECALIBRATION_FINAL_REPORT.json/.md`，将 M0-F、M0-G/H、
M0-I/J 协议及通过哈希校验的 measurement artifacts 精确复制到新的 frozen archive。生成
`MMD_FROZEN_MANIFEST.json`，必须写：

- `measurement_layer_status=frozen`
- `detector_status=not_frozen`

Frozen manifest 绑定所有归档文件 SHA256、MMD implementation commit、输入 candidate payload
hash 和 M0-I/J seed manifest hash。冻结复制不得改变 candidate 文件字节。

完成后立即停止。禁止 D→S calibration、Top-r/Max/Energy、global permutation、正式 Reference、
held-out、fresh attack 或任何32B采样。
