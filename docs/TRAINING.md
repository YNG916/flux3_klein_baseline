# 原生训练与资源使用

上游 commit 由 `UPSTREAM.lock.json` 固定并校验文件内容。使用原生 `flux3.dit.DiT(flow_map=False)`、Qwen3-VL 文本编码器、128 通道 VAE token、4-axis RoPE 和多参考图的负 l-axis ID。不以 FLUX 1/2 或其他 Diffusers 架构替代。

每个 attention 的 q/k/v/attn_out 注入 rank=alpha=32 LoRA，骨干 BF16 冻结，adapter 与 AdamW master state 为 FP32。初始化 B=0，A=Kaiming。默认 LR=5e-5、weight decay=.01、全程恒定 LR、gradient norm clip=1；H100 配方关闭 gradient checkpointing（每卡batch2实测可运行），原始通用模板保留开启选项；这是一套起点配置，尚未做任务超参选优。

目标采用确定性 VAE mean 与官方 BN 标准化。只对目标加噪：`x_t=(1-t)z+t ε`，监督 velocity `ε-z`；条件 latent、text timestep 保持 0。t 先由 logit-normal(mean0,std1) 采样，再用 `16u/(1+15u)` shift。CFG text dropout=.1，unconditional 保留相同参考图；Base 推理 50 Euler steps、CFG4、同样 shift16。该时间分布是本 baseline 明确的训练选择，上游只发布推理代码，不声称复现其未公开预训练配方。

`Packed` 支持不同文本长度的样本，跨样本 attention 由边界隔离，不要求 Qwen baseline 的 exact-layout 分桶。等长使用 batched SDPA；CUDA 变长 pack 使用 PyTorch 2.11 的 varlen Flash Attention；可设 `attention_backend=flash/cudnn/math` 做独立实验。非 reentrant block checkpointing 保持 frozen-input LoRA 梯度。DDP 只同步 adapter，累积中不重复 all-reduce；使用 fused AdamW。

默认 `cache.text_mode=online`：prepare 只加载 VAE，缓存 target/reference latent 和已扩写 prompt。训练阶段只加载 DiT、冻结文本编码器和小缓存，不加载 VAE/upsampler；每个 microbatch 在线获得 BF16 文本特征，并在 inference_mode 外 clone 后送入训练图。它减少数 TB 级 embedding 存储与读取，代价是文本前向和约 4B 编码器权重显存。`cached` 则在 prepare 编码完整 BF16 文本、训练不加载文本编码器；缓存与 config 严格绑定。两种方式使用同一个编码器与精确 prompt，按实际硬件/磁盘 profile 选择。

参考 latent 在每个编码 rank 内按原始 RGB 内容去重，避免跨进程写同一 mmap 对象；query 文件采用 PyTorch zip mmap。DataLoader 使用持久 workers、pin memory、有限预取，每个 worker 对每个文件首读校验 SHA256；不会把全集特征放入 RAM。manifest 报告独立文件总字节数和均摊容量。caption 和 feature 都在每 query 成功后原子提交 row，恢复先校验已有文件，不重算已完成 query。

默认四卡 H100：2 samples/GPU × accumulation4 × 4 GPUs = effective32。每 epoch 打乱全部 query，尾部重复前缀补齐至完整有效 batch；111,582 补齐到 111,584，含 2 次重复访问。每 epoch 3,487 次更新，10 epochs 34,870。日志同时记录 step/epoch/cursor/global_samples_seen、全部 ranks 与累积 microbatches 平均 loss、吞吐、allocated/reserved 峰值。吞吐是实测累计样本/墙钟时间，包含训练过程中验证和 checkpoint 时间。

checkpoint 保存 adapter safetensors、每 rank AdamW、Python/NumPy/Torch/CUDA/noise-generator RNG、完整采样游标、config/cache/source/world-size/validation contract 及全部文件 SHA256。仅在完整 optimizer update 边界发布 complete metadata；只有 adapter 或不完整 checkpoint 不可恢复。换配置、数据、进程数或采样协议须新实验。恢复的 `--max-steps` 是累计总目标，不能小于已完成更新。

固定 val 的每 query timestep/noise 独立派生，并保存/恢复训练 RNG。validation 使用全局平均 target-only flow MSE；每 1,000 updates 和最终状态运行。它是代理指标，不自动解释为几何准确或最优 checkpoint。用同一固定 val 图像生成另行选模；test 不参与选择。

Slurm 中断后用 `./mvwd full EXP --stage train` 找最近 complete checkpoint。训练 worker 能在收到 TERM/USR1 后于下一 optimizer 边界保存 interrupted 状态，但 launcher/硬超时不保证留足写入时间；周期 checkpoint 是可靠恢复边界。未实现自动提前停止，也没有将达到预算标成任务收敛。

实际最优配置须在目标 GPU 实测：先测每卡 microbatch 和 checkpointing，再测四卡完整更新、编码吞吐、DataLoader 等待与磁盘容量。CUDA 13 构建需要计算节点驱动支持；`doctor` 报告实际 CUDA 可用状态。CPU 数值验证不替代 CUDA 内核、DDP 速度或 GPU 显存测试。

## 2026-10-22 GPU 适配

新增 vLLM0.21 上游批处理后端：父进程负责固定 query 与缓存，子进程只暴露本 rank 的一张GPU并清理 Accelerate rendezvous 环境，避免四个TP=1 engine混入父进程组。通过专用响应pipe返回caption，日志仍归入同一experiment。GDN prefill选Triton，以兼容没有nvcc的H100节点。每个batch的生成时间、输出token和实际caption保留。

单卡原生完整更新的初测：开启重算batch2约1.27 queries/s、峰值23.47GiB；关闭重算batch2约1.67 queries/s、峰值65.53GiB。batch4/8在关闭重算时OOM。以上是小样本热缓存测量，不能直接当作完整数据/四卡ETA。

测试过的norm/RoPE融合候选虽然更快，但没有通过固定输入与初始adapter的梯度对照，因此未纳入正式代码或配置；正式训练保留原生算术。失败与候选源码只在旧smoke实验的source/gpu_probes及诊断日志中追溯。

`loss.jsonl`另记录update_compute_seconds与update_samples_per_second：从每次累积更新的首个batch已就绪起计时，包含后续累积数据等待和GPU计算，不含之前的checkpoint/验证或首个batch获取时间。原elapsed_seconds与samples_per_second仍保留完整累计墙钟口径；首步编译/初始化不能作为稳态吞吐。
