# LLM Integrity Fingerprint

基于敏感提示词指纹的云端开源大语言模型完整性验证研究代码。项目当前完成的是建立指纹之前的“敏感提示词生成与硬验收”阶段；指纹组件选择、参考特征固化和完整检测率实验仍是后续工作。

## 当前基线

- 基础模型：`Qwen/Qwen2.5-7B-Instruct`
- 内层方法：离散 HotFlip 候选搜索，联合小参数块微观代理与五家族宏观代理
- 微观代理：Hutchinson/Rademacher 探针，轮换 `q_proj`、`v_proj`、`down_proj`
- 宏观家族：非结构化剪枝、结构化剪枝、量化、参数噪声、微调
- 约束：离散修改、正代理增益、PPL、编辑比例、数字保护、任务验收和家族不退化门控
- 当前标签：`v6-sensitive-prompt-gate-passed`

V6 正式实验于 2026-07-29 完成：

| 阶段 | 关键结果 |
|---|---|
| 内层 60 条 | 60/60 技术完成，32 条代理接受，42 次离散提交 |
| 开发集外层 | 54 个组合候选，31 个严格通过，冻结选择 22 条，门槛 12，PASS |
| 冻结测试 | 22 条全部通过任务约束，16 条严格通过，门槛 12，PASS |

冻结测试严格通过提示词的平均联合增益为 `1.9419868665`，最小值 `0.2266957546`，最大值 `6.2356713900`。可审计的精简证据位于 [`reproducibility/v6_20260729`](reproducibility/v6_20260729)。

## 目录

```text
configs/          实验配置；V2-V5 保留为方法演进记录
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

## 科学边界

当前结果证明 V6 敏感提示词生成与严格外层验收流程达到了预注册数量门槛，但不能表述为完整指纹已经建立，也不能直接推断未知攻击上的最终检测率。下一阶段应从 16 条冻结测试严格通过提示词中提取激活，使用 MCC/最大新增覆盖选择最终 12 条，冻结参考响应与特征，再运行 intact、已见修改和未见修改的统计检测实验。

详细状态见 [`docs/project_status.md`](docs/project_status.md)，版本管理规则见 [`docs/version_control.md`](docs/version_control.md)。

## 许可证

当前仓库暂未添加开源许可证。除非仓库所有者另行声明，公开可见不代表授予复制、修改或再分发许可。
