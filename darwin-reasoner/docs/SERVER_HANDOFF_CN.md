> **默认零 Key 模式：** 不需要 HF_TOKEN 或 WANDB_API_KEY。先执行 `bash scripts/bootstrap_no_key.sh` 自动准备公开 MATH-500 Pilot 数据。

# 借用 8×A100 服务器交接说明

目标是让服务器提供者不需要理解论文，只需 clone、配置环境、执行命令。

## 服务器提供者需要做的最少操作

```bash
git clone git@github.com:YOUR_USERNAME/darwin-reasoner.git
cd darwin-reasoner
cp .env.example .env
# 在 .env 填 HF_TOKEN；W&B 可选
DARWIN_GPU_INSTALL=1 bash scripts/setup.sh
source .venv/bin/activate
bash scripts/check_server.sh
bash scripts/smoke.sh
```

Smoke PASS 后再运行真实任务。

## 数据与模型不进 GitHub

建议服务器目录：

```text
/data/darwin_reasoner/
  hf_cache/
  datasets/
  checkpoints/
  rollouts/
```

`.env` 中设置：

```bash
DARWIN_DATA=/data/darwin_reasoner
HF_HOME=/data/darwin_reasoner/hf_cache
HF_TOKEN=...
```

## 中断/续跑

每完成一个实验单元，结果立即 append + fsync，同时写入 `completed_keys.txt`。相同 `--run-dir` 重启时跳过已完成 key。

不要删除 `runs/<experiment>/completed_keys.txt`，除非明确想重跑全部样本。

## tmux

如果没有 Slurm：

```bash
tmux new -s darwin
bash scripts/run_iclr.sh configs/pipeline_iclr.yaml configs/full_sweep.yaml
```

detach：`Ctrl+B` 后按 `D`。

恢复：

```bash
tmux attach -t darwin
```

## Slurm

如果服务器使用 Slurm：

```bash
sbatch scripts/slurm_iclr.sh
```

先根据集群 policy 修改 partition/account/time 等字段。

## 不要做的事

不要把 Hugging Face token、W&B key、SSH private key、模型权重、checkpoint、几百 GB rollout commit 到 GitHub。不要在共享服务器宿主机直接运行未经隔离的 LLM 生成代码。
