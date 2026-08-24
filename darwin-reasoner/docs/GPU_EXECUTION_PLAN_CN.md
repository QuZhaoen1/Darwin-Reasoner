# 8×A100 执行计划

## Phase 0：本地/CPU

只跑 `bash scripts/smoke.sh`。任何 GPU 任务前必须通过。

## Phase 1：1 张或 2 张 A100 Pilot

8B，100~200 problems。目标只验证：

- matched intervention 数据有明显 action-value spread；
- world model 可以排序；
- planner 不比固定 baseline 差。

没有这三个信号，不进入 full sweep。

## Phase 2：8 张 A100 Counterfactual Atlas

优先把 GPU 用于并行 architecture intervention，而不是直接上最大模型。大量 states × architectures × repeats 是这篇论文真正的数据资产。

## Phase 3：Main SOTA

在固定 budgets 下跑 8B/14B/32B。主比较至少三 seeds。先完成 strongest matched-compute comparisons，再扩更多 benchmark。

## Phase 4：OOD / Cross-model / Evolution

主表稳定后跑 OOD architecture discovery、8B->32B transfer、macro admission、experience scaling。

## 服务器临时被收回怎么办

所有阶段使用固定 `--run-dir`。`completed_keys.txt` 保证断点续跑。不要换 run_dir，否则会重复烧计算。

## 多 GPU 布局

对于大模型，可用一个 vLLM 实例做 tensor parallel。对于较小模型，推荐把 seeds / dataset shards 分给多个独立进程，每个进程绑定不同 `CUDA_VISIBLE_DEVICES`，实现更高吞吐。具体 TP/DP 划分应根据模型尺寸、context length、KV cache 和服务器互联实测，不要提前写死一个“理论最佳”配置。
