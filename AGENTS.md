# 阅读与维护本研究仓库

- 先读 `README.md`、`docs/CURRENT_STATUS.md` 和 `docs/EXPERIMENT_INDEX.md`，再按研究问题查对应证据。机器可读进度入口为 `project-status.json`。
- 区分历史文本特征/MMD路线与当前首token E/P路线。旧32B结果存在，不代表当前首token方法已完成32B验证。
- `docs/history/`、旧实验目录和历史日志里的“当前”“正在运行”“关机”等表述仅对文档当时有效。服务器运行状态需要实时核查。
- 报告具体结论时注明模型、检测方法、预算、样本数及局限；不能把40/40或0/20观察比例写成总体保证，不能把本次证据迁入写成新实验。
- 新实验记录代码提交、配置、数据/模型身份、运行编号和结果位置。精简证据进 `reproducibility/`，原始大数据按 `docs/DATA_LOCATIONS.md` 管理。
- 历史冻结实验的执行代码以其 `CODE_MANIFEST.json` 与外部归档中的 `code_snapshot` 为准。当前工作区代码与发布包相同，不等同于重新完成GPU复现。
