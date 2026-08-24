from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .policy_data import ArchitecturePolicyDataset, CausalSFTCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="8-GPU LoRA training for DarwinReasoner policy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Optional LoRA adapter checkpoint directory")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for policy training")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    seed = int(seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def save_adapter(model: torch.nn.Module, tokenizer, output_dir: str | Path, rank: int) -> None:
    if rank != 0:
        return
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = unwrap(model)
    base.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    rank, local_rank, world_size = setup_distributed()
    seed_everything(int(cfg.get("seed", 42)), rank)
    device = torch.device("cuda", local_rank)

    # Imported lazily so CPU-only CI can still import the repository.
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if bool(cfg.get("bf16", True)) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if args.resume:
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
    else:
        lora_cfg = cfg["lora"]
        peft_cfg = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=list(lora_cfg["target_modules"]),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_cfg)

    if bool(cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.to(device)
    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[DarwinReasoner] world_size={world_size}, trainable={trainable:,}/{total:,}")

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    train_ds = ArchitecturePolicyDataset(
        cfg["train_file"], tokenizer, max_seq_len=int(cfg.get("max_seq_len", 1024))
    )
    valid_path = Path(cfg.get("valid_file", ""))
    valid_ds = (
        ArchitecturePolicyDataset(valid_path, tokenizer, max_seq_len=int(cfg.get("max_seq_len", 1024)))
        if str(valid_path) and valid_path.exists()
        else None
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    collator = CausalSFTCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("per_device_batch_size", 1)),
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=True,
        collate_fn=collator,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )

    epochs = int(cfg.get("epochs", 2))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 8))
    total_updates = max(1, math.ceil(len(train_loader) * epochs / grad_accum))
    warmup_steps = int(total_updates * float(cfg.get("warmup_ratio", 0.03)))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        return max(0.1, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = None  # A100 BF16 does not require GradScaler.
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            loss.backward()
            running_loss += float(loss.detach()) * grad_accum

            should_step = micro_step % grad_accum == 0 or micro_step == len(train_loader)
            if not should_step:
                continue
            torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if rank == 0 and global_step % int(cfg.get("log_every", 5)) == 0:
                print(
                    f"epoch={epoch + 1}/{epochs} step={global_step}/{total_updates} "
                    f"loss={running_loss / max(1, micro_step):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e}"
                )
            if global_step % int(cfg.get("save_every", 100)) == 0:
                save_adapter(model, tokenizer, Path(cfg["output_dir"]) / f"step-{global_step}", rank)

        if dist.is_initialized():
            dist.barrier()
        save_adapter(model, tokenizer, Path(cfg["output_dir"]) / f"epoch-{epoch + 1}", rank)

    save_adapter(model, tokenizer, cfg["output_dir"], rank)
    if rank == 0:
        print(f"[DarwinReasoner] training complete: {cfg['output_dir']}")
    cleanup()


if __name__ == "__main__":
    main()
