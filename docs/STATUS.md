# 当前状态：2026-10-21

代码、环境和固定权重均位于 multiview workspace。Base 已在用户完成 Hugging Face 登录后下载，约 20.30 GB；prompt upsampler 约 18.84 GB。GPU allocation 尚未提供，未启动长训或提交 Slurm 作业。

本机已创建 `20261021_l1_smoke`（`cpu_inputs` 和 `camera_inputs` 均完成且未读取 GT）及 `20261021_l1_full10ep_4h100_v1`（冻结配置/源码和全量训练计划）。无需重复 init；先 `./mvwd status` 查看。所有 GPU 阶段尚未运行。

全量计划是 111,582 queries/epoch、effective batch32、3,487 updates/epoch、10 epochs。在线文本方案的 latent tensor payload 在去重前约 train75.75 GB / val5.50 GB；这不含 prompt、zip、manifest 等开销，实际容量以小样本 manifest 为准。

已完成：

- 33 项 CPU tests：GT-free 数据/相机条件、固定采样与 seed、split/asset 防篡改、原生 DiT LoRA 梯度、Packed 样本隔离、非 reentrant checkpointing、在线文本的 inference-tensor 转换、caption token 预算、缓存恢复、完整训练恢复、验证 RNG 隔离、PSNR 评估适配和 GT/BEV 导出。
- 原生小 DiT 的两进程 CPU/gloo DDP：两个 optimizer updates 后，所有 adapter 参数在两个 rank 逐位相同；启用 block gradient checkpointing。
- 实际 Base 357 个权重张量、VAE 251 个张量的 key 和 shape 与固定官方代码逐项匹配。原生 DiT 为 5,538,458,368 参数；这些是文件/结构检查，不是预训练模型 GPU 前向。
- 实际 upsampler processor 的单样本与多样本两图输入检查，模型配置可构建其 9B meta 模型；未执行该预训练模型的生成。
- 核对 train / val / test 为 111,582 / 8,100 / 20,028 个固定 query。正式数据和相邻基线未修改。

下一步在用户提供的 tmux GPU 节点中，先运行 `./mvwd doctor` 检查 CUDA 13 驱动兼容，再按 README 的 smoke 路径验证实际 caption、latent、2→4 步训练恢复、单/四卡吞吐与显存、固定 val 图像和评估。吞吐最优 microbatch、是否关闭 checkpointing，以及在线/缓存文本的取舍，都需要真实硬件测量。

尚无预训练 FLUX 的实际训练、生成质量或正式 test 指标；不能从 CPU 小模型或完整代码推导模型已收敛、几何遵从或配置已最优。长训由用户明确启动。
