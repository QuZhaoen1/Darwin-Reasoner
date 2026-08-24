#!/usr/bin/env python3
from __future__ import annotations

import sys
import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: torch.cuda.is_available() is False")
    count = torch.cuda.device_count()
    print(f"CUDA devices visible: {count}")
    if count < 8:
        raise SystemExit(f"FAIL: need at least 8 visible GPUs, found {count}")
    failures = []
    for i in range(8):
        props = torch.cuda.get_device_properties(i)
        gib = props.total_memory / 1024**3
        name = props.name
        capability = f"{props.major}.{props.minor}"
        print(f"GPU {i}: {name}, {gib:.1f} GiB, compute capability {capability}")
        if "A100" not in name.upper():
            failures.append(f"GPU {i} is not reported as A100: {name}")
        if gib < 38.0:
            failures.append(f"GPU {i} has only {gib:.1f} GiB; default profile assumes A100 40GB+")
    if failures:
        print("\n".join("FAIL: " + x for x in failures))
        raise SystemExit(2)
    print("PASS: 8xA100 hardware visibility check")


if __name__ == "__main__":
    main()
