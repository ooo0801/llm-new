# H8 F1-B：Qwen2.5-32B 正式 Final Confirmation 采样协议

## 1. 阶段边界

F1-B 仅生成并审计正式 Final Confirmation 响应，不计算 feature、MMD、Score、global permutation、FPR 或攻击检出率。冻结检测器 `r60_q10_top2` 只作为身份依赖加载；性能确认必须等待独立的 F1-C 授权。

## 2. 冻结数据设计

- `final_reference_only`：每个 fingerprint 60 条，共 720 条；只生成一套，并由全部 100 个 evaluation units 共享。
- `final_heldout_intact_target_only`：60 个 intact units，每 unit、每 fingerprint 10 条，共 7,200 条。
- `final_heldout_attack_only`：Gaussian、Pruning、LoRA、Quantization 各 10 个 frozen fresh endpoints，每 endpoint、每 fingerprint 10 条，共 4,800 条。
- 总计严格为 12,720 条，`batch_size=1`。

唯一合法输入是 F1-A 冻结的 generation、attack、permutation、freshness 和 historical-identity 工件。F1-A 六条 tiny smoke 永远不得进入正式数据。

## 3. Fail-closed 门

第一条正式回答前，runner 必须验证：

1. F1-A preflight 和 6/6 smoke 均为 PASS，且 smoke 的正式响应计数为 0；
2. 当前分支及提交继承 F1-A smoke 封存提交，runner/module 哈希与授权文件一致，tracked worktree 干净；
3. Frozen Detector manifest、所选 detector payload、F1-A 所有 manifests/audits、H6 MCC12、generation config、runtime authorization 和 LoRA 隔离训练集的 SHA256 一致；
4. 12,720 个 response IDs/seeds 唯一，并与历史、smoke、endpoint materialization/training seeds 以及 CPU permutation seeds 无交叉；
5. `HF_HUB_CACHE=/root/autodl-tmp/huggingface`，同时启用 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`；不得回退默认缓存；
6. 本地 32B snapshot、tokenizer、chat template、EOS、generation config、runtime identity 和 GPU 清理状态一致。

任何门失败均在生成下一条回答前停止。

## 4. 执行与恢复

执行被划分为一个 intact base-model partition（7,920 条）和 40 个 attack endpoint partitions（每个 120 条）。每个 partition 在第一条回答前必须物化并封存身份；攻击 endpoint 必须使用 F1-A 的配置、materialization seed、LoRA training seed 和 planned identity，不得根据结果替换。

每条成功记录以 fsync 方式追加。恢复时只接受与 schedule 的 `response_id/schedule_position/seed/role/prompt/endpoint` 完全一致的成功前缀；成功记录不可重生成或覆盖。技术失败最多按授权重试，所有 attempt 必须使用同一 generation seed。合法 first-token EOS 空响应按有效样本保留，不得补采。

每条响应生成前检查冻结文件 metadata 和 live tokenizer/runtime identity；每个 12-prompt unit 开始时检查模型参数 state monitor。LoRA adapter 与量化/剪枝/Gaussian 物化证据均写入独立封存文件并哈希。

## 5. 完整性审计与停止条件

采样完成后只验证：12,720 总量、720/7,200/4,800 角色计数、12 prompts 覆盖、60 intact units、40 attack endpoints、ID/seed 唯一、角色与 endpoint membership、same-seed retry、stop reason、空响应、token 长度、物化身份和 GPU 清理。

报告必须明确：

- `formal_detector_statistics_computed=false`
- `final_features_computed=false`
- `final_mmd_computed=false`
- `final_scores_computed=false`
- `final_global_permutation_computed=false`
- `final_fpr_computed=false`
- `final_attack_detection_rate_computed=false`

生成 `H8_F1B_FORMAL_FINAL_CONFIRMATION_SAMPLING_REPORT.json` 和 `F1B_FORMAL_SAMPLING_SHA256_INDEX.json` 后立即停止，等待 F1-C 明确批准。
