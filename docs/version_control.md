# 版本管理规范

## 仓库角色

- 服务器 `/root/autodl-tmp/llm`：GPU 实验与日常开发工作区。
- GitHub 私有仓库：代码、配置、测试和精简科研证据的长期中心仓库。
- Windows 本地克隆：可选的离线副本，不单独初始化为无关仓库。

## 跟踪边界

Git 跟踪源代码、脚本、配置、测试、提示词数据、文档和 `reproducibility/` 下人工冻结的小型证据。`results/`、模型权重、适配器、缓存、日志、PID、IDE 状态和历史备份不进入普通 Git。

在使用 `git add .` 前必须先检查：

```bash
git status --short
git diff --stat
```

提交前检查暂存内容：

```bash
git diff --cached --stat
git diff --cached --check
```

## 分支与提交

- `main` 始终表示已验证、可复现的基线。
- 实验性修改使用 `experiment/<short-name>` 分支。
- 修复使用 `fix/<short-name>` 分支。
- 一次提交只表达一个完整意图，不能混入模型权重或原始结果。

建议提交信息：

```text
feat: add activation-based fingerprint selection
fix: correct structured FFN pruning assignment
test: cover portfolio provenance without round
docs: record V6 frozen-test baseline
experiment: add unseen-attack evaluation manifest
```

## 实验可追溯性

每次正式实验至少记录：

- Git commit SHA 和未提交修改状态；
- 完整配置与命令；
- 数据、攻击清单、模型 revision 和 adapter 校验和；
- 随机种子、软件版本、GPU 型号；
- 阶段完成标记和最终报告；
- 是否使用开发集或冻结测试，以及是否发生测试集反馈。

正式结论通过后，把小型报告复制到 `reproducibility/<version>/`，提交并创建带说明的 Git tag。原始大结果继续保存在实验存储中。

## 日常流程

```bash
git switch main
git pull --ff-only
git switch -c experiment/fingerprint-mcc

# 修改与测试
pytest
git status --short
git add <明确文件>
git commit -m "feat: build MCC fingerprint from V6 prompts"
git push -u origin experiment/fingerprint-mcc
```
