# 新服务器、重新生成提示的RESF式pilot

本方案取代2026-09-17早期“复用旧提示、完整文本对照、下载BGE”的草案。用户已明确只要新生成提示和新检测效果。

## 当前范围

固定Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306，BF16/eager。新服务器独立模型和adapter目录，不用旧响应或旧候选；不下载BGE。

本地分支 experiment/resf-token-small-20260917 由GitHub已核对的e295ad1及本地bac0108修复派生。新增代码不表示GitHub已发布。

按阶段执行 scripts/run_resf_small.py PHASE --root /root/autodl-tmp/token-integrity/runs/fresh-v1，PHASE依次为prepare/train/utility/preflight/calibrate/search/reference/evaluate。环境安装、下载、CPU测试完成后才prepare。

## 新生成与攻击

- 从GitHub source pool构造器重新取得extraction/classification各4个来源；原模型预检后各选第一个合格来源。
- 两来源×JS/Top-K continuous/raw-logit L2三个独立搜索臂，最多6个搜索任务、每任务5轮、micro K8。七模块、四代表层均衡调度，不声称5轮覆盖全部组合。
- micro/macro尺度和梯度平衡在新服务器重新计算，不继承旧CALIBRATION。
- 保留micro+macro+HotFlip，五攻击族均须内部不退化；语义/任务保持约束关闭，结果是机制探索，不宣称自然任务保持。
- 搜索使用Gaussian σ=.006（每张参数张量标准差的比例）、40步LoRA、20%非结构剪枝、0.5%FFN结构剪枝、INT8阈值8；所有端点先过32题独立效用筛查。效用集与来源/训练集精确不重叠，模板泛化未认证。
- 新训练4个LoRA seed，每个40个optimizer step，rank8/alpha16/lr1e-4/constant scheduler/dropout0；一个seed参与搜索，三个仅用于后续检测。微型32样本SFT语料，不是原论文5K Alpaca。
- Gaussian同样一个搜索seed、三个未参与搜索的检测seed。单一强度，不能宣称跨强度或跨配置held-out。
- 参数变化实际记录BF16变化计数和ΔW范数；changed_parameters历史字段仍是尝试修改数量，实际值在details.realized_audit。

## 修正版检测

参考直接读取原模型logits，但待测判定只读取每次请求的第一个原始token ID。待测侧generate(max_new_tokens=1)，独立请求100次，不从一次生成取100个token，不用概率采样模拟代替真实待测调用。

声明温度候选集合{0.5,0.7,0.9}，实际检测温度0.7；每个参考候选均应用真实temperature/top-k50/top-p0.9后处理。暂不宣称覆盖连续温度区间或任意服务变更。

- Rule E：各候选真实采样支持集的并集，任何集合外token立即报警。该并集在声明的null族下尾部质量为0；数值/后处理不匹配可能破坏此前提，所以与generate处理后的概率直接对照。
- Rule P：保留所有支持token类别，稀疏计数采用多项分布deviance，在有限温度网格上拟合最小deviance；对每个null网格点模拟10,000组，取各自Monte Carlo p值的最大值，拒绝整个复合null族。
- 预定查看点30/60/100；面板α=.05按去重提示数分配，再按E20%/P80%和查看次数分配。有限样本MC rank用(1+超过次数)/(B+1)，不直接套稀疏χ²。
- 这属于RESF-inspired修正版，不是原论文连续β拟合或原始统计公式的精确复现；选择的离散温度族及联合支持可能降低功效，必须如实报告。
- 两个独立真实intact-null bank检查实际误报；仅两个面板不足以认证真实FPR≤5%。本地小规模模拟单元测试也不是FPR认证。

## 预算和停止

最多6提示×8bank×100次=4800条首token响应；12模型状态×32效用题=384条，预检32条，参考一致性烟测最多6条，总生成请求上限5222。效用/预检最长64新token；其余1。LoRA总160步，搜索30轮，micro/macro校准前后向和CPU模拟另计。实际accepted提示少于6会减少预算，零候选直接停，不用旧候选或原source冒充新候选。

未通过效用、非有限梯度、参考分布对不上实际解码、无accepted候选等均停止。不自动提高攻击强度、换seed、扩提示池、调整α、执行旧launcher或关机。

## 复现和状态

prepare冻结代码hash和模型manifest hash。模型文件hash由下载阶段记录；adapter文件hash绑定训练报告。阶段锁防重复进程，响应逐条保存，续跑检查身份和计划；修改实现后创建新run，不静默在冻结run继续。

技术完成只记TECHNICAL_COMPLETE，最后RESULTS.json是EXPLORATORY_COMPLETE，各Gaussian/LoRA端点分报。没有基于检测结果再选面板，全部accepted去重候选组成固定面板；保证“新生成”但不宣称最优面板。

当前脚本尚需在新服务器执行集成测试。通过CPU测试不等于GPU搜索或科学结果通过。
