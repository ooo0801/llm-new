# AutoDL 操作清单

1. 租用按量计费实例。0.5B/1.5B 使用 24GB GPU；7B 权重梯度建议 48GB。
2. 把项目上传到 `/root/autodl-tmp/llm-integrity-fingerprint`。
3. 执行 `bash scripts/setup_autodl.sh`。
4. 执行 `source /root/autodl-tmp/venvs/llm-integrity/bin/activate`。
5. 先运行 `python scripts/smoke_test.py --config configs/debug_qwen_0.5b.yaml`。
6. 下载模型时可执行 `source /etc/network_turbo`，完成后按平台说明关闭代理。
7. 长实验必须放在 tmux/screen 中。
8. 每次只加载一个修改模型；`score_prompts.py` 已按此设计。
9. 结果及时复制到本地或 AutoDL 文件存储。本地数据盘不作为唯一备份。
10. GPU 不使用时及时关机；关机前确认结果已落盘。
