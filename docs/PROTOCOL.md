# 数据与公平性协议

本实现对齐本机 Top2Pano、Qwen-Image 2.1 和用户指定的 Qwen 基线规划聊天的 Level 1 约定。

输入为 before clean environment RGB BEV、BEV 标定、当前物理帧全部机器人的身份/base pose/相机高度、实际 RGB modality 相机 K/T，以及共享 `MVWD_RobotAssets_v1_gt_aligned` 的机器人资产元数据。第二张参考图在同一 BEV 上绘制米制网格、机器人身份/朝向和目标相机 frustum；矩阵、资产部件和身份颜色保留在精确文本中。

该条件版本不渲染 GLB、不添加局部 BEV、scene mesh、SAM、GT depth、GT semantic、GT visibility、after 状态或 world BEV。共享资产已移除 native top_body；机器人条件是元数据与标记，不能宣称显式 mesh 或三维几何约束。相机提示只是投影方向，不是遮挡或可见区域。

训练监督仅为 before perspective RGB；与 Top2Pano 使用 RGB/depth 的监督表达不同，应在结果表披露。训练 loss 是目标 latent 的 flow MSE，不新增 depth/几何/跨视图/时间 loss。Independent 模式每次生成一个相机视角，不保证严格跨视图或时序一致性。

读取器直接解析正式 JSONL/JSON/NPZ，不导入 simulator、producer、consumer_tools 或 Top2Pano adapter；只在评估时调用既有 Top2Pano metrics/GT 注册代码。数据集和源资产只读，split 文件按 dataset.json 的 SHA256 校验，资产 metadata 按 manifest 校验。

采样与现有基线相同：每个 episode 的前后半段各由 SHA256(seed42, episode, stratum) 选一帧，0/1/2 三个 observer，跨 epoch 重复固定两帧。train 为 18,597 episodes / 111,582 queries，val 为 1,350 / 8,100，test 为 3,338 / 20,028。数据按原始官方 split，不重划或按 GT 内容挑选。

目标 256×448 保持完整视野，不裁剪；K 随 resize 更新。BEV 等比例 letterbox 并更新标定。世界坐标米制 Z-up，CV 相机右/下/前。所有条件在训练、推理和导出时使用同一 adapter。

Prompt upsampling 使用固定官方 9B 权重、system prompt、reference JPEG/resize 策略与 i2i scaffold，提供旧版 Transformers greedy decoding 与当前四卡 vLLM 批处理后端；vLLM使用 temperature=0、i2i presence penalty=0 及官方 scaffold。当前相机视图v2在原始请求前明确要求指定机器人相机的地面视角，禁止将地图本身当作输出视角，不新增条件或监督。不是绕过 upsampler 的原始短 prompt。然后原样附加精确条件文本，避免 VLM 改写 K/T/机器人数值。这是 MVWD 任务适配，不能描述成逐位复现官方 vLLM 输出。原始条件/完整 upsampler caption/实际模型 prompt/策略和 hash 都保留。按真实文本 tokenizer 的 4,096-token 上限显式缩短派生 caption，receipt 记录是否缩短及实际长度；精确条件不截断。若精确条件本身超限则拒绝。

推理仅访问条件和 GT-free captions，不使用 train/val feature 缓存，更不构建 test target cache。全部预测写完并标记 complete 后，独立 review 才能读 GT；生成 metadata 永远记录 `ground_truth_loaded=false`，review 另存 receipt。预测像素和既有导出不允许静默覆写。

评估复用已有 PSNR、LPIPS、FID、MV-GTWarp-DINO 实现与权重。正式结果要求完整 test、固定两帧、三个机器人、seeds 42/43/44、256×448 和四项完整指标；子集或单 seed 必须 `--preview`。先 infer 保存图像，再 eval 读取预测和 GT；eval 不重新运行生成模型。

正式测试例子（对每个 seed 使用同一 checkpoint、同一 captions，分别命名）：

```bash
./mvwd caption "$exp" test --processes 4 -- --split test
for seed in 42 43 44; do
  ./mvwd infer "$exp" "test_s${seed}" --processes 4 -- --split test \
    --captions "$root/captions/test" --lora "$checkpoint" --seed "$seed"
done
./mvwd eval "$exp" test_metrics -- \
  --predictions "$root/predictions/test_s42" "$root/predictions/test_s43" "$root/predictions/test_s44" \
  --metrics-repo ../top2pano --weights-dir ../top2pano/models/metrics
```

小模型工程测试、训练子集拟合或 PSNR 高于随机参照均不能证明布局、相机遵从或机器人形体正确；需检查固定 val 生成和逐项指标。当前未报告正式生成效果。Level 2 的编辑标签和训练路径尚未实现。
