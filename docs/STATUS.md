# 当前状态：2026-10-26

**完整训练已结束，正式全量评估已启动。** 用户仓库`origin/main`；环境、固定权重、下载/编译缓存和实验产物均位于multiview workspace，数据和相邻baseline未修改。

## 训练结果

主实验：`20261022_l1_full10ep_4h100_v3`。累计**10epochs / 34,870 optimizer updates**，覆盖固定111,582 train queries/epoch、effective batch32。最终checkpoint：`training/train_20261025T162706453348/final`，complete=true，epoch10；最后训练步骤exit0。adapter及4个rank的优化器/RNG状态SHA256、源码、环境匹配均已核验。先后在93041、96075、101076续跑；前两个allocation到时退出，旧日志保留。

| 检查 | 结果 |
|---|---:|
| 有效轨迹训练loss全部有限 | 是 |
| 前100步训练loss均值 | 0.70268 |
| 最后100步训练loss均值 | 0.45043 |
| 固定8,100-query初始val flow MSE | 0.78240 |
| 最终val flow MSE | 0.68832 |
| 最低val代理MSE / step | 0.67053 / 13,000 |

存在一定后期过拟合迹象，不能把final称为质量最优checkpoint；flow MSE也不能直接替代生成指标。本次正式评估按预先约定使用预算结束的final，没有根据test GT挑模型。

本机结果（不进入Git）：[训练结果报告](../artifacts/mvwd/experiments/20261022_l1_full10ep_4h100_v3/reports/TRAINING_FINAL_REVIEW.md) · [训练曲线](../artifacts/mvwd/experiments/20261022_l1_full10ep_4h100_v3/reports/training_curves.png)。

## 最终模型出图检查

固定3个val场景、18views、seed42、50steps/CFG4，完整生成后再独立导出GT/BEV与评估。小样本preview已完成：

| Metric | Val preview |
|---|---:|
| PSNR | 11.38290 dB |
| LPIPS | 0.69686 |
| FID | 235.94009 |
| MV-GTWarp-DINO | 0.61674 |

这些**不是正式test成绩**；18张FID尤其不适合排名。查看图像可见地面相机视角、机器人颜色和结构已学到，场景材质、遮挡、具体相机位置仍有明显误差，几何遵从尚不能宣称已解决。旧smoke与本次caption可能不同，不能把它们当作严格同prompt受控提升幅度。

[最终模型GT/BEV浏览页](../artifacts/mvwd/experiments/20261022_l1_full10ep_4h100_v3/predictions/final_val3_seed42_20261026/index.html) · [Preview指标详情](../artifacts/mvwd/experiments/20261022_l1_full10ep_4h100_v3/evaluations/final_val3_metrics_20261026/METRICS.md)。

## 当前正式评估

已按用户授权复用**job101076 / hkn0907 / 四H100**，登录节点`hk2n1989`的tmux会话：

```bash
tmux attach -t flux3-eval-101076
```

完整test 20,028views × seeds42/43/44=60,084预测，原生50steps、CFG4。按100episodes分为34个分片/seed，完整分片支持跨allocation复用。当前先生成GT-free test captions，随后自动推理、合并和四指标；**正式指标尚未完成**。该allocation可能不足以完成全部3seed，后续需要新的四卡节点接力；不会自动提交新Slurm作业。

在新allocation重复以下命令即可续跑：

```bash
cd ~/multiview_baselines/flux3_klein
./slurm/evaluate_existing_allocation.sh NEW_JOB_ID 20261022_l1_full10ep_4h100_v3
```

分片是完整test的行政分区，联合支持集必须等于全test；不将子集分数伪装成正式成绩。每个预测的seed依赖episode/frame/robot，不随分片编号变化。完整分片保留；部分失败分片重新命名attempt重算。合并逐图检查checksum、prompt/条件、模型/环境及seed，并保留原始分片来源和生成sidecar；图像像素保持原样。三个完整seed就绪后才读test GT并计算PSNR/LPIPS/FID/MV-GTWarp-DINO。

本机：[评估计划](../artifacts/mvwd/experiments/20261022_l1_full10ep_4h100_v3/plans/formal_evaluation.json) · [当前进度](../artifacts/mvwd/experiments/20261022_l1_full10ep_4h100_v3/reports/formal_evaluation_progress.json) · [启动记录](../artifacts/mvwd/experiments/20261022_l1_full10ep_4h100_v3/reports/evaluation_start_101076.json)。冻结的原生训练/推理源码保持不变；评估启动器另有哈希和实验内快照。

## 实现与验证

**45项CPU tests通过**，其中新增完整split分片/合并、幂等恢复、seed/像素/条件保持和拒绝缺片/重复/模型混用/像素修改的回归。另将真实18张val预测通过同一合并器处理，再调用原始指标代码，PSNR与合并前**逐值相同**。这验证了接口和像素保持，不代表正式test已完成。

当前H100训练配方：每卡batch2、累积4、关闭gradient checkpointing；原生norm/RoPE不变，fused AdamW、BF16冻结骨干、FP32 LoRA。实际完整训练更新约5秒，有效轨迹累计墙钟吞吐约5.7queries/s。vLLM四replica、Triton attention/GDN，批量16；超长/空caption显式重试一次，仍失败时保留完整原始任务约束降级，所有尝试和终止原因保存。GPU故障注入已覆盖retry/fallback。更快的norm融合候选未通过梯度对照，未采用。

历史实验`20261021_l1_smoke`、`20261022_l1_gpu_smoke_v2`和旧full_v1/v2均保留。不得用变更后的核心源码追加旧冻结实验；查看旧source快照或对应Git checkout。数据、checkpoint与完整缓存均严格校验；不放宽恢复或正式评估覆盖协议。
