# 按可用额度进行 sbatch 续跑

入口：`slurm/cw_resume.sbatch`，配套运行控制为 `tools/cw_batch_runtime.py`。这是独立于 SSH/tmux 的批任务，不再依赖交互式 `salloc` 控制会话。

按用户最新要求，优先申请当前额度能容纳的最长单次任务，资源保持 **4 张 H100、64 CPU、384 GiB 内存**，账户保持 `hk-project-p0029489`。模板上限为分区允许的 **48 小时**；提交时可用 `--time` 指定更低的实际可调度时长。不自动提交任务链。账户额度不足时仍可能排队；成功提交不等于资源已经分配。

```bash
sbatch --parsable --chdir="$PWD" \
  --output="$PWD/artifacts/cw/experiments/20261117_cw_full10ep_4h100_v1/logs/slurm-%j.log" \
  slurm/cw_resume.sbatch 20261117_cw_full10ep_4h100_v1
```

## 恢复与停止

1. 获取同一实验的 batch 写锁，防止两个批任务同时续训。
2. 从原实验中选择最新的完整 checkpoint，校验冻结训练合约以及模型、optimizer、四个 rank RNG 的所有校验和。缺失、损坏、配置或环境不匹配时退出，保留错误日志。
3. 保持原有模型、数据顺序、累积次数、学习率及 **23,250 次更新的完整 10 epoch 目标**。缓存直接复用。
4. 继续训练到 allocation 的提前退出通知或完整训练结束；不会在 step 2000 提前结束，原有每 500 步的完整 checkpoint 继续保存。没有修改 `--max-updates` 或另建训练预算。
5. 若接近墙钟时限，Slurm 提前 45 分钟只通知 batch shell，为可能正在进行的整轮验证及完整 checkpoint 留出余量。shell 写 STOP 文件；四卡在原有 optimizer 边界多做一次停止标志 all-reduce，再一致调用原处理器。不会向 DataLoader、torchrun 或训练进程树发送终止信号来请求正常保存。
6. 真实子进程失败则返回其非零退出码；只有验证最终完整 checkpoint 后才生成 batch 的 `completion.json`。batch shell 的 `wait` 若被提前通知中断，会继续等待子进程真实结束。

训练源码及历史 checkpoint/缓存合约不修改，也不放宽 hash 校验。运行控制通过明确记录的启动层安装两个临时回调：optimizer 边界的停止共识，以及原生完整 checkpoint 保存后的停止请求；不替换模型/损失、梯度、optimizer、采样器或 RNG 算法。控制脚本、batch 脚本各自 hash 写入本次 plan。源码如果在排队期间发生改变，预检会拒绝启动。

每次 batch 使用 `joint_sbatch_JOBID_r0` 独立 attempt；运行计划和完成报告在 `EXPERIMENT/reports/ATTEMPT/`，原生训练日志和 checkpoint 仍在 `logs/`、`training/`。未完成的保存目录不会被当作可恢复状态，不覆盖原尝试。

## 验证范围与可靠性

2026-11-18 全仓库 CPU 回归 **76 passed**；四进程 CPU 测试验证了只有一个 rank 收到停止请求时四个 rank 一致停止，以及原生模型、optimizer、各 rank RNG 的完整保存与恢复。报告在 `artifacts/cw/verification/batch_runtime_20261118/`。

原模型已通过单卡/四卡 H100 前后向验收及约 1,886 次真实更新。这次新增 batch 控制层的真实 H100 调度和提前停止，需要等任务获得资源后再确认，不能把 CPU 测试称为已经通过完整 GPU batch 验证。

不能保证任务绝不失败。`sbatch` 可以隔离终端断开；提前保存可以降低正常超时的损失，但硬件故障、文件系统故障、强制取消、长时间挂起或额度限制仍可能中断任务。下一次提交会继续选择同一轨迹的最新完整状态。

Slurm 参数依据：[sbatch 官方文档](https://slurm.schedmd.com/sbatch.html)，特别是 `--signal=B:USR1@2700`（只通知 batch shell）和 `--test-only`（只预检，不提交）。项目授予量与已用量通过集群官方 `kit_project_usage` 查询；有共享额度、已运行任务预留或不可见上级限制时，名义余额不等于当前可调度余额。
