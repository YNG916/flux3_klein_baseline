# 当前状态：2026-10-22

代码、独立 Python 环境、固定权重、缓存与实验均在 multiview workspace；`origin` 是用户仓库 `YNG916/flux3_klein_baseline`。已使用用户提供的 job **93041 / hkn0909 / 4×H100 94GiB** 做小规模真实GPU验证。修复后完整流水线已在同一allocation恢复，未提交新作业；目前尚处于提示词预处理阶段，尚未产生正式训练更新。

本机推荐实验：

- `20261022_l1_gpu_smoke_v2`：覆盖全部40个训练场景的240 queries、全部3个验证场景的18 queries；原生模型、camera-view-v2 prompt、vLLM、关闭 gradient checkpointing。
- `20261022_l1_full10ep_4h100_v3`：修复超长输出后的推荐全量实验，已冻结新策略/源码，迁入2,528条已校验的v2成功提示词。111,582 queries/epoch，有效batch32，3,487 updates/epoch，10 epochs共34,870 updates。
- `20261022_l1_full10ep_4h100_v2`：用户启动后在2,528条提示词处失败，原因是个别输出未正常结束并触发2048token保护；尚无训练更新。原实验与失败日志保留，不在原冻结实验上改策略。
- `20261021_l1_smoke` 与 `20261021_l1_full10ep_4h100_v1` 保留历史源码和产物。旧实验不能用变更后的当前源码追加 GPU 步骤；查看其 source 快照，或使用原 Git commit `55ad4de` 的独立 checkout。

## 已验证内容

**此前36项 CPU tests 全通过**，包含实际共享指标代码集成；另外已有两进程 CPU/gloo 原生小 DiT DDP 对照。测试覆盖 GT-free 输入、相机干预、固定采样、跨样本 attention 隔离、LoRA 梯度、checkpointing、缓存恢复、完整状态恢复、验证 RNG 隔离、指标和对照导出，以及 vLLM 进程隔离/失败检测。CPU 测试报告在本机 `verification/gpu_update_cpu_tests_20261022.xml`。

真实 GPU 已完成：固定 Base / VAE / text encoder 加载、四个独立 vLLM replica、240+18条完整提示词、VAE latent 缓存、四卡DDP 2步训练及完整恢复至4步。第2步结束与恢复开始的固定18-query val flow MSE **逐值相同（0.8359332432349523）**；恢复后游标继续到16、累计128次 query 访问。checkpoint 包含 adapter、每rank AdamW/RNG/采样进度与完整性校验。

当前 v2 的240条训练提示词为2,246～2,632个native text tokens，没有截断几何、没有读取GT。实际 train latent/prompt 缓存143.58MB，约598.25KB/query；full plan另保存去重前tensor容量上界，不能把小样本均摊量当作全量精确容量。

固定验证集的50步CFG4生成、相同seed零样本参照、独立GT/BEV浏览页和四指标已全部完成，13个步骤均exit0。源码、上游、checkpoint与预测像素哈希审计通过。生成期间不加载GT；完整生成后review/metrics才有独立GT访问记录。

本机浏览：[GPU验证报告](../artifacts/mvwd/experiments/20261022_l1_gpu_smoke_v2/reports/GPU_VALIDATION.md) · [4步LoRA对照](../artifacts/mvwd/experiments/20261022_l1_gpu_smoke_v2/predictions/val_s4_seed42/index.html) · [同条件零样本对照](../artifacts/mvwd/experiments/20261022_l1_gpu_smoke_v2/predictions/val_zeroshot_seed42/index.html)。产物不进入Git。

18张val、seed42的工程preview：PSNR **10.6273dB**，LPIPS **0.79038**，FID **378.936**，MV-GTWarp-DINO **0.71427**（6个有效相机对）。这不是正式test，也不是收敛模型成绩；18张的FID尤其不适合模型排名。

## 资源测量与当前配方

单卡小缓存、完整optimizer更新（不含累积）的原生算子测量：

| 配置 | queries/s | 峰值allocated GiB | 结论 |
|---|---:|---:|---|
| checkpointing开启，batch2 | 1.275 | 23.47 | 可用于较小显存 |
| checkpointing开启，batch4 / 8 / 16 | 1.187 / 1.231 / 1.271 | 28.07 / 36.67 / 51.94 | 增大batch未提高单卡吞吐 |
| checkpointing关闭，batch2 | 1.667 | 65.53 | 当前H100配方 |
| checkpointing关闭，batch4 / 8 | OOM | — | 不采用 |

当前四卡DDP每次更新含2×4×4=32 queries：短测更新耗时约**5.14～5.60秒**（首步6.66秒），合计约**5.72～6.22 queries/s**；主rank峰值allocated约64.8GiB、reserved最高71.4GiB。这些是4次更新的工程测量，不代表长程稳态或所有rank最坏显存。vLLM每卡16条的后续热batch约4.8～5.3秒，首批约19秒，另有模型初始化时间。

norm/RoPE融合候选没有通过梯度对照，**未采用**。正式代码保留原生算术，不能以候选更快的数字宣传正式训练吞吐。旧smoke实验保留候选源码、失败日志和测量记录。

按上述小样本吞吐外推，10epochs仅计算约50～54小时，尚不含全量缓存、验证、I/O与checkpoint；48小时allocation预计需要续节点。当前节点足够完成工程检查，无需立即再排。周期checkpoint和`full --stage train`支持下一个同规格四卡allocation继续。


## 超长输出修复与恢复状态

新增回归后**44项CPU测试通过**。真实H100故障注入覆盖“1token首次输出→512token精简重试成功”及“两次1token输出→使用完整原始任务约束降级”两条路径；所有精确相机约束完整保留。原失败的16-query批次复测全部正常结束，未发现固定样本损坏。

新策略只处理异常输出，最多重试一次；正常首次请求不变。失败尝试原文、终止原因、token数和最终`caption_status`全部记录，训练质量报告须披露降级比例。v2的2,528条结果经过源码/配置/环境/输入和哈希校验迁入v3，每条保留来源，旧缓存未改动。恢复不会复用不同源码下的训练checkpoint。

本机故障注入：[原始日志](../artifacts/mvwd/experiments/20261022_l1_full10ep_4h100_v3/logs/forced_length_probe.log)；迁移记录在v3的`captions/recover_v2_train/import_receipt.json`。用户已明确授权恢复完整流水线，现通过登录节点`hk2n1989.localdomain`的tmux会话`flux3-v3-93041`复用job93041运行v3的`full --stage all --epochs 10`。提示词/特征完成后自动进入训练；未申请新节点。启动记录：v3的`reports/restart_93041.json`，驱动日志：`logs/driver-93041-*.log`。

## 本机正式启动

v3已经启动，勿重复启动并行写入同一缓存。当前会话可在登录节点查看：

```bash
tmux attach -t flux3-v3-93041
```

将来从新allocation恢复时，在对应四卡节点/tmux中：

```bash
cd ~/multiview_baselines/flux3_klein
./mvwd doctor
./mvwd full 20261022_l1_full10ep_4h100_v3 --stage all --epochs 10
```

若从登录节点复用当前 allocation，在命令前使用：

```bash
srun --jobid=93041 --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 --gres=gpu:4 \
  --chdir="$PWD" ./mvwd full 20261022_l1_full10ep_4h100_v3 --stage all --epochs 10
```

缓存完整后，后续allocation使用`--stage train`；缓存未完成则重跑`--stage all`，逐query校验续写。配置、代码、依赖、world size不应在同一次实验中改变。需要重新排队时由用户显式使用`./mvwd batch ...`。

## 质量限制

四步训练只证明链路、数值和恢复可用，**没有完成任务适配，也没有正式test指标**。旧v1的四步训练集出图仍偏向俯视房屋，12张preview PSNR约9.49dB。v2强化了同一允许输入下的机器人相机视角请求，但检查其四步验证集对照图后，仍可见明显俯视/高视点偏差，与GT相机和机器人外观不匹配。因此目前是工程验证通过、生成质量未通过；精确视点、场景与机器人一致性仍需训练后验证，不能从flow MSE或prompt文本推导。

正式test要求固定全量20,028个queries、seeds42/43/44和原有四指标；小规模preview的FID/几何分数只能用于工程排查，不能作为baseline正式成绩。此RGB监督方法与含depth监督的Top2Pano需要明确区分。
