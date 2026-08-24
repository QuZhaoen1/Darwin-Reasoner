# ICLR 实验协议（冻结版骨架）

## E0：CPU/Mock Smoke Test

目的：不占 GPU 验证整个工程链路。必须通过：配置解析、DAG validation、执行、counterfactual collection、world-model training、macro mining、resume、结果表生成。

命令：`bash scripts/smoke.sh`

## E1：真实模型 Baseline Sanity

先用 8B + 100~200 题验证 evaluation pipeline。至少跑 CoT、verify/refine、branch-search family。此阶段不声称 SOTA。

硬规则：相同 model revision、answer parser、temperature policy、max token budget。

## E2：Matched Counterfactual Collection

单位不是“题”，而是 `(problem, reasoning_state, seed_block)`。

每个 state 产生 K 个 architecture candidates。至少包含：continue/solve、verify-heavy、branch-heavy、repair-heavy、tool-heavy、generated/evolved program。

对每个 candidate 做 R 次 repeated matched rollouts。记录：reward、tokens、latency、calls、architecture graph、state prefix、seed block、model revision。

### 随机化要求

相同 matched block 内 architecture 顺序随机；所有 candidate 使用同一 seed family。避免把 GPU 时间、缓存或随机状态系统性绑定到某一方法。

## E3：World Model

训练集按 problem/state 分组切分，不能让同一 state 的不同 architecture 泄漏到 train/test 两边。

指标：

- reward prediction Brier / NLL；
- Spearman rank correlation across architectures within a state；
- Top-1 best-architecture accuracy；
- Top-k recall；
- expected regret of selected architecture；
- calibration error；
- OOD domain and OOD base-model degradation。

必须比较：

- no world model / random ranking；
- scalar value model；
- no uncertainty ensemble；
- no architecture features；
- no state features；
- full model。

## E4：Main DarwinReasoner

对每题：candidate generation -> world model -> risk/Pareto filtering -> execute -> observe -> optional replan。

不能用 reference answer 来选择候选。reference 只在 evaluation 最后用于 correctness。

主预算：2K / 4K / 8K / 16K / 32K generated+prompt token accounting，并额外记录 wall-clock latency 和 search overhead。

## E5：SOTA Battle

优先官方实现/官方 checkpoint：AutoMR、MetaFlow、HyEvo、BayesFlow，以及当前提交前检索到的新 strongest baselines。

主表需要：Accuracy、avg tokens、p95 tokens、latency、search overhead、total model calls、GPU-hours。

任何 baseline 如果因接口差异无法做到严格同预算，要单独说明，不可以偷偷放宽 Ours 预算。

## E6：Ablations

冻结以下实验，不根据结果删：

- Full;
- w/o matched interventions;
- w/o world model;
- w/o uncertainty / active design;
- w/o online replanning;
- fixed operator grammar;
- w/o macro admission test;
- w/o architecture prior/distillation;
- random architecture;
- static best architecture;
- oracle architecture (只作 upper bound，不是合法部署方法)。

## E7：OOD Architecture Discovery

提前固定 domain split，例如：

- train: Math + Logic, test: Code;
- train: Math + Code, test: Logic。

测试时不允许改 task-specific prompt，不允许人工新增 workflow。记录系统生成的 canonical DAG，并做 topology similarity / operator-frequency 分析。

## E8：Cross-Model Transfer

A：8B collection -> 8B world model -> 14B/32B inference；
B：8B+14B collection -> 32B held-out；
C：model-specific world model upper bound。

验证 reasoning-architecture value 是否有可迁移结构，而不是每个模型完全重学。

## E9：Grammar Evolution

先 mine motif，再做 macro admission A/B。新 macro 必须满足至少一个条件：

- reward 显著提高；或
- reward non-inferior 且 token/latency 显著降低。

报告被拒绝的 macro 数，防止只展示成功案例。

## E10：Experience-Time Scaling

随着累计 matched-intervention states 数量增长：0 / 1K / 5K / 10K / 50K / 100K，冻结一组 future tasks。

画：

- accuracy vs accumulated experience；
- search rollouts vs accumulated experience；
- total test tokens vs accumulated experience；
- world-model ranking regret vs accumulated experience。

如果 search cost 明显下降而 accuracy 不降，这是论文最重要的 secondary result 之一。

## 统计规范

主 accuracy 用 problem-level paired bootstrap 95% CI。architecture intervention effect 用 matched state/seed paired bootstrap。多 benchmark 可报告 macro-average，并保留每个 benchmark 原始结果。所有随机方法至少三 seeds；高成本 32B sweep 如果实在无法三 seeds，必须把限制写清楚并优先保证主比较三 seeds。

## 防止借卡场景的算力浪费

所有 run_dir deterministic + completed key resume。服务器被中断后重复同一命令会跳过已落盘结果。Pilot gate 低于阈值时自动停止 full sweep。大型 sweep 前必须 `bash scripts/check_server.sh` 和 `bash scripts/smoke.sh`。
