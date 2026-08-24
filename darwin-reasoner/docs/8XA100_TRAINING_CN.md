# 8×A100 训练说明

本仓库把「推理实验」和「参数训练」分开：

1. **推理/反事实轨迹生成**：`vLLM`，可用 `tensor_parallel_size: 8` 占用 8 张 A100。
2. **Architecture Policy 参数训练**：Qwen3-8B + BF16 LoRA + PyTorch DDP，一卡一个进程，使用 `torchrun --nproc_per_node=8`。
3. **Reasoning World Model**：默认是轻量 ensemble，可先单卡训练；它不是 8 卡瓶颈。

默认训练配置 `configs/training/policy_qwen3_8b_a100.yaml` 是为 A100 40GB/80GB 保守设置的：

- Qwen/Qwen3-8B
- BF16
- LoRA，仅更新 adapter
- per-device batch size = 1
- sequence length = 1024
- gradient checkpointing = true
- gradient accumulation = 8
- 8 GPU DDP

这比在 8 张卡上做 8B 全参数微调稳得多，也更符合本项目的目标：基础 reasoner 冻结，训练“如何选择/合成推理架构”的策略，而不是重新训练基础知识。

## 上卡前必须通过

```bash
source .venv/bin/activate
bash scripts/check_8xa100.sh
```

它会做两层检查：

- Python 检查 8 张可见 GPU 都是 A100 且显存至少约 40GB；
- `torchrun` 启动 8 个 NCCL 进程并做一次 all-reduce。

只有看到：

```text
PASS: 8xA100 hardware visibility check
PASS: NCCL all_reduce across 8 GPUs
```

再开始正式训练。

## 数据准备

先运行反事实轨迹采集，得到例如：

```text
runs/pilot_counterfactual/raw_results.jsonl
```

再转换为 Architecture Policy SFT 数据：

```bash
python scripts/build_policy_dataset.py \
  --input runs/pilot_counterfactual/raw_results.jsonl \
  --train data/policy_train.jsonl \
  --valid data/policy_valid.jsonl
```

## 8 卡训练

```bash
bash scripts/train_policy_8xa100.sh
```

LoRA adapter 默认保存在：

```text
checkpoints/policy_qwen3_8b_lora/
```

## 重要边界

本仓库的默认 8 卡参数训练路径是 **Qwen3-8B LoRA**。不要把 `Qwen3-32B` 的全参数训练理解为已保证可行。32B 在本项目中优先用于 vLLM 多卡推理/rollout；若要做 32B 参数训练，应另外采用 ZeRO-3/FSDP、LoRA，并根据 A100 是 40GB 还是 80GB重新调参。

## 一条链跑 Pilot → 8卡训练

完成环境安装后，可以直接：

```bash
bash scripts/pilot_to_train_8xa100.sh
```

其逻辑是：

1. 先做 8×A100 + NCCL 检查；
2. 若缺 MATH-500 pilot，自动下载 200 题；
3. 把 200 题切成 8 份，每张 A100 启动一个独立 Qwen3-8B vLLM worker，生成 matched counterfactual rollouts；
4. 合并 8 个 worker 的 `raw_results.jsonl`；
5. 从每个相同 reasoning state 的多种 architecture outcome 中选择高 reward、低 compute 的 supervision；
6. 构建 `data/policy_train.jsonl`；
7. 用 `torchrun --nproc_per_node=8` 进行 Qwen3-8B LoRA DDP 参数训练。

这样 **轨迹生成阶段和参数训练阶段都会真正使用 8 张卡**，而不是“配置里写了 8 GPU，实际只有一张卡在工作”。
