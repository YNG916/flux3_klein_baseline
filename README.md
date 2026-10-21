# FLUX 3 Image Klein Base × MVWD

全局 BEV → 指定机器人相机透视图的 **Level 1 baseline**。日常只用 `./mvwd`，所有环境、权重与产物放在 multiview workspace；不改动 Top2Pano、Qwen 或数据集。

当前已实现数据适配、原生多参考图 DiT LoRA 训练、完整续训、推理、原有四项指标和 GT/BEV 浏览页。CPU 验证已通过；真实 H100 吞吐、显存和生成质量尚待计算节点实测，见 [当前状态](docs/STATUS.md)。Level 2 不属于当前实现。

## 开始使用

```bash
cd ~/multiview_baselines/flux3_klein
./mvwd --help
./mvwd doctor
./mvwd status
```

已有环境自动选择，无需 activate。默认路径：

```text
workspace/
├── baselines/flux3_klein/          本仓库
├── envs/flux3-klein/              独立 Python 3.11 / PyTorch 2.11 / CUDA 13
├── software/python/flux3-python311/  本机独立基础 Python，不依赖 Qwen 环境
├── models/flux3_klein/            固定 revision 的 Base / prompt upsampler
├── datasets/MultiViewWorldDataset_v1/
└── cache/                        下载与编译缓存
```

新机器先设置 `MULTIVIEW_STORAGE_ROOT`，提供数据与共享机器人资产，运行 `git submodule update --init --recursive`，再运行 `PYTHON_BOOTSTRAP=python3.11 ./scripts/install.sh`。环境实际完整版本见 `requirements.lock.txt`。可用 `FLUX_PYTHON`、`MVWD_METRICS_PYTHON` 覆盖解释器。

首次下载权重：

```bash
# Base 要先在 Hugging Face 接受许可证，再在终端登录；不要将 token 写进配置。
workspace_root="${MULTIVIEW_STORAGE_ROOT:-$(cd ../.. && pwd -P)}"
HF_HOME="$workspace_root/cache/huggingface" \
  "$workspace_root/envs/flux3-klein/bin/hf" auth login
HF_HUB_OFFLINE=0 ./mvwd download
```

默认离线读取本地权重。上游代码、Base 和 upsampler 的 commit 都已固定；下载避免重复存储 Diffusers 格式和量化变体。

## 先做小规模验证

先创建实验、检查输入（CPU），再在**已有 GPU allocation** 内执行 GPU 命令：

```bash
exp=20261021_l1_smoke
./mvwd init "$exp" --config configs/level1.local.json --purpose 'L1 原生 FLUX 工程验证'
./mvwd inspect "$exp" inputs -- --split train --max-episodes 2 --preview-samples 3
./mvwd diagnose "$exp" cameras -- --split train --max-episodes 1 --max-groups 1

root="artifacts/mvwd/experiments/$exp"
./mvwd caption "$exp" train12 --processes 4 -- \
  --split train --max-episodes 2 --max-samples 12
./mvwd prepare "$exp" train12 --processes 4 -- \
  --split train --max-episodes 2 --max-samples 12 --captions "$root/captions/train12"
./mvwd profile "$exp" batch -- --cache "$root/features/train12" --batches 1 2 4 8
./mvwd train "$exp" smoke2 --processes 4 -- --cache "$root/features/train12" --max-steps 2
./mvwd train "$exp" resume4 --processes 4 -- \
  --cache "$root/features/train12" --max-steps 4 --resume-from "$root/training/smoke2/final"
```

`profile` 是显式 GPU 测量：包含前向、反向与 AdamW 更新，报告吞吐、峰值 allocated/reserved，逐个记录 OOM。它使用独立临时模型，不改实验 checkpoint；单卡结果不代替 DDP 吞吐测量。先实测再确定正式配置，不能据显存空闲量宣称最优 batch。

验证集推理、自动对照和评估：

```bash
./mvwd caption "$exp" val_one -- --split val --max-episodes 1
./mvwd infer "$exp" val4 -- --split val --max-episodes 1 \
  --captions "$root/captions/val_one" --lora "$root/training/resume4/final" --seed 42
./mvwd eval "$exp" val4_eval -- --predictions "$root/predictions/val4" \
  --metrics-repo ../top2pano --weights-dir ../top2pano/models/metrics \
  --preview --metrics psnr --device cpu
./mvwd status "$exp"
```

完整生成结束后，CPU review 才读取 GT，原预测旁保存 GT、条件 BEV、逐帧及三相机对照、`index.html`；生成 metadata 的 `ground_truth_loaded=false` 保持不变。`--no-review` 可关闭自动对照。`--zero-shot` 可代替 `--lora`，仍保存到同一训练 experiment。

## 公平全量训练

正式训练创建独立实验。默认四卡、每卡 microbatch 2、累积 4，有效 batch 32；覆盖全部 **111,582** train queries / epoch，10 epochs，共 **34,870** optimizer updates。默认配置是待实测的起点，不是已证明的最优超参数。

```bash
exp=20261021_l1_full10ep_4h100_v1
./mvwd init "$exp" --config configs/level1.local.json --purpose 'L1 全量两帧公平训练'
./mvwd full "$exp" --stage plan

# 已有四卡交互节点内运行；先通过上面的小规模检查。
./mvwd full "$exp" --stage all --epochs 10

# 若已有全部缓存，只训练；新 allocation 重复此命令会找最近完整 checkpoint。
./mvwd full "$exp" --stage train --epochs 10
```

`all` 依次生成 train/val captions、编码 latent、训练。caption/latent 缓存逐 query 原子提交、校验后续写；失败/重试各有独立日志。全量特征缓存默认保存 latent 和文本，冻结文本编码器在线运行，避免海量 20,480 维 embedding I/O。`cache.text_mode=cached` 可缓存完整 BF16 文本特征，用磁盘换取训练计算；需要新实验并先测容量。参见 [训练实现](docs/TRAINING.md)。

需要后台排队时，显式提交：

```bash
./mvwd batch "$exp" --stage all --dry-run
./mvwd batch "$exp" --stage all
./mvwd batch "$exp" --stage train --after JOB_ID
```

Slurm 模板在 `slurm/`：4×H100、64 CPU、384G、48h，账号与分区可按集群调整。仅 `batch` 或手工 `sbatch` 会提交；普通命令不申请 GPU。完整 checkpoint 每 1,000 updates、每 epoch 和正常结束保存；异常中断回退到最近已完整写入的 checkpoint。

## 命令与产物

| 命令 | 作用 |
|---|---|
| `init EXP --config FILE` | 一次独立训练轨迹，冻结配置和源码 |
| `select EXP NAME -- …` | 不读取 GT 的场景平衡子集 |
| `inspect / diagnose EXP NAME -- …` | CPU 条件预览与相机干预计划 |
| `caption / prepare EXP NAME --processes N -- …` | GT-free prompt upsampling / train-val latent 编码 |
| `profile / train EXP NAME -- …` | 显式 GPU 性能测量 / 训练与完整恢复 |
| `infer EXP NAME -- …` | GT-free 生成，再自动导出对照 |
| `eval / review EXP NAME -- …` | 评估已有预测 / 重建或校验浏览页 |
| `full EXP --stage plan/caption/prepare/train/all` | 全量计划与可恢复流水线 |
| `status [EXP]` | 自动实验索引与报告 |

单次步骤的任务参数放在 `--` 后；步骤支持 `--dry-run`。每个步骤名称唯一，失败日志也保留。

```text
artifacts/mvwd/experiments/EXP/
├── config.json / manifest.json / source/   冻结配置、源码、上游与依赖版本
├── plans/ / selections/                   全量预算、固定子集
├── captions/ / features/                  可恢复缓存及校验清单
├── training/NAME/                         loss、固定 val、完整 checkpoints
├── predictions/NAME/                      原始预测、生成条件记录、GT/BEV 对照、浏览页
├── evaluations/NAME/                      metrics.json、逐图/逐对指标
└── steps/ / logs/ / reports/               argv、环境、状态、原始日志、OVERVIEW
```

同一次训练的续训、不同 checkpoint/seed/split 的推理、zero-shot 参照及评估均归入同一 experiment。独立训练或改变配置要新建 experiment。权重、缓存、实验和本地配置不纳入 Git。

输入、监督、正式 test 及限制见 [任务协议](docs/PROTOCOL.md)。开发检查：

```bash
PYTHONPATH="$PWD/vendor/flux3/src:$PWD" MVWD_METRICS_REPO=../top2pano \
  ../../envs/flux3-klein/bin/python -m pytest -q
```

本仓库使用[官方 FLUX 3 Image Klein](https://github.com/black-forest-labs/flux-3-image-klein)原生模型。上游代码为 Apache-2.0；模型权重使用其各自许可证。
