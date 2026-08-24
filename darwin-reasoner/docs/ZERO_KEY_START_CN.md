# 零 API Key 启动指南

本仓库默认可以在 **没有 Hugging Face Token、没有 W&B API Key、手头没有数据集** 的情况下启动。

## 默认选择

- 模型：`Qwen/Qwen3-8B`（公开 Hugging Face 模型）
- Pilot 数据：`HuggingFaceH4/MATH-500`（公开数据集）
- 实验跟踪：本地 `runs/`，W&B 默认关闭
- API：0 个

公开 Hugging Face 仓库下载时认证是可选的；只有 gated/private 资源才需要 token。

## 服务器第一次运行

```bash
git clone git@github.com:YOUR_USERNAME/darwin-reasoner.git
cd darwin-reasoner

bash scripts/bootstrap_no_key.sh
bash scripts/smoke.sh
bash scripts/no_key_preflight.sh
```

然后运行真实 8B Pilot：

```bash
source .venv/bin/activate
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export WANDB_MODE=disabled

python -m darwin_reasoner.cli collect \
  --config configs/pilot_8b.yaml \
  --run-dir runs/pilot_counterfactual
```

第一次运行时 vLLM/Transformers 会从 Hugging Face 下载公开 Qwen3-8B 并缓存。以后不会重复完整下载。

## 什么时候才需要 HF_TOKEN？

只有之后改用 gated/private 模型或数据集时才需要。当前默认实验不需要。

## 什么时候才需要 WANDB_API_KEY？

只有你决定把实验曲线上传 W&B 云端时才需要。当前所有关键结果都会写入 `runs/`，完全可以不用 W&B。

## 如果 GPU 服务器无法访问 Hugging Face

在一台能联网的机器上先下载公开模型和数据，然后把缓存/数据复制到 GPU 服务器。不要把 16GB 以上模型权重提交到 GitHub。`data/math500_pilot.jsonl` 只有 200 条时很小，可以在确认许可与协作策略后作为实验输入单独传给服务器。
