# 小模型研究快照，2026-09-08

`SOURCE_PROVENANCE.json` 保存 81 个导入文件的原始 SHA256：
67 个源码/配置/协议文件和 14 个未修改的服务器证据文件。
当前目录的 README、验证脚本和顶层分支导航为本次发布新增，不冒充原运行产物。

## 证据

- `evidence/stage2_attack_utility_calibration_v2_20260906/`：独立效用校准设计、效用集、汇总及历史验证清单。
- `evidence/stage2_three_proxy_ceiling_v3_20260906/`：冻结搜索计划、预检查、代理尺度、v2 复用清单、最终状态及晋级决定。
- `evidence/stage2_three_proxy_ceiling_v3_20260906/pilot_behavior/evaluation/`：行为计划和连续效应矩阵。

历史 `VERIFICATION.json` 引用了外部运行文件；它不表示这些文件全部随 GitHub 发布。
本次仅对实际导入的文件验证 SHA256。完整响应和模型权重留在服务器/独立备份中。

## 本次本机验证

- Windows / Python 3.13.5：小模型目标测试 81 passed。
- 全仓 `python -m pytest`：381 passed，1 failed。
- 唯一失败：`tests/test_h8_m0ij.py::test_candidate_global_mask_is_bound_to_every_scaler`。
  它要求未发布到 Git 的旧 32B `results/h8_qwen32b_mmd_precalibration/m0gh/`
  中的全局掩码等运行产物。未修改的 H9 `a3e2f59` 上单独运行该测试亦失败。
  没有删除测试、伪造文件或改变统计算法来掩盖这一缺失。
- 未启动模型下载、GPU 采样、LoRA 训练或自动关机脚本。

发布后的代码或文档继续修改时，Git commit 是新版本身份；本清单仍只用于核验这次冻结快照。
