# Validation status

Validated in the packaging environment:

- Python source compilation: PASS
- Repository unit tests: **7 passed**
- Editable package import (without downloading optional GPU dependencies): PASS
- Policy-dataset builder on mock counterfactual output: PASS
- Existing mock end-to-end reasoning pipeline: PASS in the prior harness
- 8-GPU hardware/NCCL smoke test: included as `scripts/check_8xa100.sh`; it must be executed on the target server because this packaging environment has no A100 GPUs

The default parameter-training path is deliberately conservative:

- Qwen/Qwen3-8B
- BF16
- LoRA only
- sequence length 1024
- per-device batch size 1
- gradient checkpointing
- 8-process PyTorch DDP launched with `torchrun`

Not claimed:

- A completed 8×A100 training run in this packaging environment.
- Qwen3-32B full-parameter training. 32B is primarily an 8-GPU vLLM rollout/inference profile here.
