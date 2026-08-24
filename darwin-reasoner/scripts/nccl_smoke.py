from __future__ import annotations

import os
import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = world * (world + 1) / 2
    if abs(value.item() - expected) > 1e-4:
        raise RuntimeError(f"NCCL all_reduce mismatch: {value.item()} != {expected}")
    dist.barrier()
    if rank == 0:
        print(f"PASS: NCCL all_reduce across {world} GPUs")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
