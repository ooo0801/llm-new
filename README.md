# LLM Integrity Fingerprint

基于敏感提示词指纹的云端开源大语言模型完整性验证研究代码。项目已完成两个相互独立、均可复现的闭环版本：V1 在 16 条候选的并集上选择 MCC12；V2 用 336 条独立广覆盖校准提示词估计全局可观察组件集合，并在同一 16 条候选上执行全局 MCC12 和全新的下游验证。

## 当前基线

- 基础模型：`Qwen/Qwen2.5-7B-Instruct`
- 内层方法：离散 HotFlip 候选搜索，联合小参数块微观代理与五家族宏观代理
- 微观代理：Hutchinson/Rademacher 探针，轮换 `q_proj`、`v_proj`、`down_proj`
- 宏观家族：非结构化剪枝、结构化剪枝、量化、参数噪声、微调
- 约束：离散修改、正代理增益、PPL、编辑比例、数字保护、任务验收和家族不退化门控
- V6 基线标签：`v6-sensitive-prompt-gate-passed`

V6 正式实验于 2026-07-29 完成：

| 阶段 | 关键结果 |
|---|---|
| 内层 60 条 | 60/60 技术完成，32 条代理接受，42 次离散提交 |
| 开发集外层 | 54 个组合候选，31 个严格通过，冻结选择 22 条，门槛 12，PASS |
| 冻结测试 | 22 条全部通过任务约束，16 条严格通过，门槛 12，PASS |

冻结测试严格通过提示词的平均联合增益为 `1.9419868665`，最小值 `0.2266957546`，最大值 `6.2356713900`。可审计的精简证据位于 [`reproducibility/v6_20260729`](reproducibility/v6_20260729)。

## Fingerprint V1 闭环结果

V1 固定使用上述 16 条严格通过提示词，不在本版本补类别。两次激活提取的组件集合完全一致，MCC12 覆盖 `12619/13138 = 96.05%` 的稳定组件，并冻结 120 条参考响应。端点使用 12 条提示词、每条 10 次响应、999 次置换，共 840 次目标查询。

| 模型状态 | 分层 MMD p | pooled MMD p | V1 判断 |
|---|---:|---:|---|
| 完整模型 | 0.842 | 1.000 | 未修改，正确 |
| 30% 非结构化剪枝 | 0.001 | 0.160 | 已修改，正确 |
| 20% 结构化 FFN 剪枝 | 0.001 | 0.001 | 已修改，正确 |
| 高斯噪声 0.001 | 0.710 | 1.000 | 未检出，假阴性 |
| INT8 | 0.001 | 0.931 | 已修改，正确 |
| NF4 | 0.001 | 0.684 | 已修改，正确 |
| LoRA r8/25 steps | 0.001 | 0.319 | 已修改，正确 |

按模型状态计，V1 的 Accuracy 为 `85.71%`、Precision 为 `100%`、Recall 为 `83.33%`、F1 为 `90.91%`、FPR 为 `0%`。分层 MMD 检出 5/6 个修改状态；pooled MMD 只检出结构化剪枝，再次证明跨提示词混合会产生假阴性。精简证据位于 [`reproducibility/fingerprint_v1_20260730`](reproducibility/fingerprint_v1_20260730)。

## Fingerprint V2 全局校准闭环结果

V2 不覆盖 V1。初始的 224 条校准集在任何端点实验开始前未通过预先冻结的饱和度和审计新颖度门禁，因此按协议扩充为 14 个任务族、每族 20 条构建提示词和 4 条独立审计提示词，共 336 条；门禁阈值、组件定义、攻击强度和统计方法均保持不变。扩充后得到 36,561 个全局可观察组件，最后三批相对新增率为 `0.44%/0.60%/0.44%`，审计新颖度为 `3.44%`，分别通过 `2%` 和 `8%` 的冻结门禁。

V2 将覆盖率拆分为：候选可达覆盖率 `27.78%`、MCC12 选择效率 `97.11%`、最终全局覆盖率 `26.98%`，且三者满足乘法分解。按组件类型，最终覆盖率为注意力 `98.12%`、FFN `25.48%`、attention residual `100%`、MLP residual `100%`。旧口径 `12619/13138 = 96.05%` 仍保留，但只表示 16 条候选并集内覆盖率，不能解释为全局覆盖率。

| 模型状态 | 分层 MMD p | pooled MMD p | V2 判断 |
|---|---:|---:|---|
| 完整模型 | 0.640 | 1.000 | 未修改，正确 |
| 30% 非结构化剪枝 | 0.001 | 0.036 | 已修改，正确 |
| 20% 结构化 FFN 剪枝 | 0.001 | 0.001 | 已修改，正确 |
| 高斯噪声 0.001 | 0.298 | 1.000 | 未检出，假阴性 |
| INT8 | 0.001 | 1.000 | 已修改，正确 |
| NF4 | 0.001 | 0.899 | 已修改，正确 |
| LoRA r8/25 steps | 0.001 | 0.354 | 已修改，正确 |

V2 同样完成 7 个状态、840 次目标查询，Accuracy 为 `85.71%`、Precision 为 `100%`、Recall 为 `83.33%`、F1 为 `90.91%`、FPR 为 `0%`。分层 MMD 检出 5/6 个修改状态，pooled MMD 检出 2/6 个。全局目标选出的 12 条与 V1 的集合相同，但顺序改变；V2 使用新的参考/目标随机种子重新生成全部响应。精简证据位于 [`reproducibility/fingerprint_v2_global_20260731`](reproducibility/fingerprint_v2_global_20260731)。

## 目录

```text
configs/          实验配置；含冻结的 Fingerprint V1/V2 及历史方法演进记录
data/             提示词与攻击训练数据（不含模型权重）
docs/             公式映射、实验协议、运行与调试说明
reproducibility/  已冻结实验的精简报告、清单和校验信息
scripts/          数据、校准、内层生成、外层验收和指纹流程入口
src/llm_integrity 核心 Python 实现
tests/            不依赖模型下载的回归测试
results/          本地实验产物，Git 忽略
backups/          历史备份，Git 忽略
```

## 环境

生产实验使用的服务器环境：Python 虚拟环境 `/root/autodl-tmp/venvs/llm-integrity`、NVIDIA RTX 4080 SUPER 32 GiB、Hugging Face 缓存 `/root/autodl-tmp/huggingface`。模型权重不进入 Git。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
python -m compileall src scripts tests
```

在 AutoDL 上可参考 [`docs/autodl_runbook.md`](docs/autodl_runbook.md)。

## V6 正式流程入口

以下脚本分别对应候选生成、开发集外层重排和冻结测试：

```bash
bash scripts/run_discrete_v6_formal_60_generate.sh
bash scripts/run_discrete_v6_formal_60_dev_rerank.sh
bash scripts/run_discrete_v6_frozen_test.sh
```

这些脚本计算成本高，并会写入 `results/`。复现实验前应先核对脚本中的输出目录、缓存路径和现有阶段标记，避免覆盖已冻结结果。

V1 完整闭环入口：

```bash
bash scripts/run_fingerprint_v1_all.sh
```

V2 全局校准闭环入口：

```bash
bash scripts/run_fingerprint_v2_all.sh
```

## 科学边界

V2 可以声明：在冻结的 336 条独立校准集和 V1 组件定义下，经验全局可观察组件集合为 36,561 个，MCC12 最终覆盖其中 26.98%，并完成七个模型状态的独立闭环。它不能声明获得数学意义上的全部关键组件，也仍不是高功效攻击分布评估：每个状态只有一个端点实例，完整模型也只有一组随机重复，不能把 `0%` FPR 或单实例 `83.33%` Recall 解释为总体概率。高斯噪声 0.001 的假阴性需要在后续版本通过预注册强度曲线、更多攻击实例和分层 bootstrap 继续研究。

详细状态见 [`docs/project_status.md`](docs/project_status.md)，版本管理规则见 [`docs/version_control.md`](docs/version_control.md)。

## 许可证

当前仓库暂未添加开源许可证。除非仓库所有者另行声明，公开可见不代表授予复制、修改或再分发许可。
