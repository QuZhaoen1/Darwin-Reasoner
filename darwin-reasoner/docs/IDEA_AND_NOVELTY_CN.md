# DarwinReasoner：Idea 与 Novelty 审计（中文）

## 1. 我们真正研究什么

这篇工作的中心问题不再是“给一道难题多少 token”，而是：

> **在当前推理状态和计算预算下，应该构造并执行什么样的推理架构？模型能否先预测不同推理架构的后果，再决定怎么想，并在长期经验中扩展自己的推理搜索空间？**

令当前内部推理状态为 `s`，候选推理架构为 `A`，计算预算为 `B`，底座模型为 `M`。我们的基本对象不是单步 action，而是一个可执行的 typed DAG：

`A = (G, O, P, R)`

其中 `G` 是拓扑，`O` 是节点算子类型，`P` 是节点参数/提示词，`R` 是路由与停止规则。

目标是在预算约束下最大化：

`J(A | s, B, M) = E[reward] - lambda_c * compute - lambda_l * latency - lambda_u * uncertainty`。

更重要的是，候选集合本身不是固定的。经过跨题经验积累，系统能够把经常出现且经过验证的局部程序压缩成新 macro，从而让搜索空间从 `S_0 -> S_1 -> ...` 演化。

## 2. 为什么不能只做 Dynamic Test-Time Compute

2026 年的相邻工作已经覆盖了很多“明显的第一步”：

- AutoMR 已经做 query-aware DAG/meta-reasoning skeleton 的动态搜索；
- MetaFlow 已经学习在给定 task + operator set 时一次性生成 workflow，并强调 OOD operator/task generalization；
- HyEvo 已经进化 LLM + deterministic code 的 hybrid workflow，并同时优化效果、成本和延迟；
- BayesFlow 已经把 workflow generation 表述成 Bayesian posterior sampling，并通过 parallel look-ahead 做价值估计；
- FlowEvo 已经能把成功 workflow 编译成持久 reusable skills，再反馈给后续任务；
- UniScale 等工作已经把不同 test-time scaling 模式的自适应组合推向统一调度。

因此，“我们也搜索 workflow”“我们也会 evolution”“我们也会动态 budget”都不足以作为核心 novelty。

## 3. 我们把核心差异钉在哪里

### Innovation A：Matched Reasoning Interventions

普通 trajectory data 是：

`s -> A -> y`。

我们构造：

`s -> {A1, A2, ..., Ak} -> {y1, y2, ..., yk}`。

对于同一个 `s`，固定 problem、prefix、base model、budget family、verifier，并使用 matched seed block，对多个 reasoning architecture 做并行 intervention。

于是可以直接估计：

`Delta(s; Ai, Aj) = E[R | s, Ai] - E[R | s, Aj]`。

这比“从历史日志相关性判断哪个 workflow 好”更强。论文中不要轻率写严格 causal identification，除非随机化、匹配和干预定义完全成立；稳妥表述是 *matched interventional supervision / paired intervention effect*。

### Innovation B：Counterfactual Architecture World Model

训练一个 `W_phi`：

`W_phi(s, A, B, M) -> (reward_mean, reward_uncertainty, token_cost, latency)`。

它不是模拟外部环境，而是模拟“**如果我用这种推理架构思考，会发生什么**”。

关键实验不是 regression loss，而是：

1. 对同一个 state 的候选 architecture 排序 Spearman；
2. Top-1 / Top-k 最佳架构召回率；
3. calibration；
4. world-model-guided selection 的真实 downstream regret；
5. OOD task / OOD model 上是否仍能排序。

### Innovation C：Active Counterfactual Experimental Design

8 张 A100 的价值不是无脑同时跑 8 个固定 action，而是让系统选择“最值得实验的反事实”。

采集函数同时考虑：

`acquisition = predicted_value + beta * epistemic_uncertainty - gamma * predicted_cost`。

因此昂贵 GPU rollouts 同时承担两件事：解决当前问题，以及主动学习 reasoning dynamics。

需要一个 uniform-random intervention control，证明 active design 在相同数据采集预算下学到更准的 world model。

### Innovation D：Self-Expanding Typed Graph Grammar

与“固定 operator set 内搜索”不同，系统会从成功架构中挖 motif。例如：

`branch -> critique -> backtrack -> verify`

可以生成候选 macro：`critical_repair`。

但不能看到 motif 常出现就直接纳入 grammar。必须做 matched A/B：

- A：执行原展开程序；
- B：执行压缩 macro；
- 相同 state + seed block；
- reward 做 non-inferiority；
- cost 必须显著下降或成功率显著提升。

只有通过 admission test 才进入下一代 grammar。

这让“search space evolution”有实验定义，而不是一句宣传口号。

### Innovation E：Budget-Conditioned Pareto Compiler

不是只优化 accuracy，而是让 architecture compiler 接收预算：

`Compiler(s, B_token, B_latency, risk) -> A*`。

同一个题在 2K、8K、32K budget 下应该产生不同 architecture，而不是同一个 workflow 截断到不同长度。

主图必须是 Accuracy-Compute Pareto Frontier，而不是只放最高预算 accuracy。

### Innovation F：Counterfactual-Regret Distillation

昂贵 search 的最优选择被蒸馏成一个 cheap architecture prior：

`pi(A_family | s)`。

下一批题先用 prior 缩小搜索，再用 world model 重排，再在高 uncertainty 时做真实 intervention。

这形成：

`Search -> Experience -> World Model -> Distill -> Cheaper Search`。

### Innovation G：Experience-Time Scaling

提出并实证一个新的 scaling 轴：历史 architecture experience。

研究：

`Performance = f(model_size, test_compute, accumulated_architecture_experience)`。

更关键的是验证：

`search_compute_needed(n experience) ↓`

在 accuracy 不下降时，历史推理经验是否能持续降低未来题目的 search cost。

这个结果如果成立，比单 benchmark +1% 更具有论文记忆点。

## 4. 我认为最强的 paper story

不要写成“我们有七个模块”。统一写成三层因果链：

1. **Observe alternatives:** matched architecture interventions 产生高价值监督；
2. **Learn consequences:** architecture world model 学习“怎样思考会怎样”；
3. **Change the algorithm:** planner 利用 world model 合成架构，并把被验证的 motifs 纳入 evolving grammar，随后 distill 经验。

一句话：

> **DarwinReasoner learns a model of the consequences of computation, then uses it to search and expand the space of inference algorithms.**

## 5. 最需要打出来的 SOTA 结果

至少三条战线中的两条必须强：

1. **Matched-compute accuracy SOTA**：同模型、同最大 token/FLOP budget，Accuracy 高于最强 search/workflow baseline；
2. **Matched-accuracy efficiency SOTA**：达到同一 accuracy 所需 tokens/latency 更低；
3. **OOD architecture discovery SOTA**：训练 Math+Logic，测试 Code 等未见 domain，不给人工 workflow，仍能形成有效新 topology；
4. **Search amortization**：随着 experience 增加，同等性能下 architecture-search rollout 数持续下降；
5. **Cross-model transfer**：8B 学到的 world model / grammar 可以迁移到 14B/32B，并证明哪些结构规律具有 model-invariant 性。

## 6. 最危险的 reviewer 攻击与提前封堵

**“只是模块堆砌。”** 解决：所有模块围绕 paired intervention -> world model -> architecture synthesis 这条链，其他模块必须有明确 ablation。

**“你只是烧更多 inference compute。”** 解决：matched token/FLOPs/latency 主表 + Pareto curve + search overhead 单独计费。

**“world model 并没有预测 reasoning dynamics。”** 解决：对 candidate architecture 的 held-out rank correlation、top-1 action regret、OOD calibration。

**“operator invention 就是 skill memory。”** 解决：强调 typed graph grammar expansion、matched macro admission、结构化作用位置，并与 FlowEvo-style trace-to-skill reuse 做直接对比。

**“counterfactual 是滥用因果术语。”** 解决：论文把主数据称为 matched interventions；只有随机化/paired estimator 支撑的比较才称 intervention effect。

**“OOD 是挑任务。”** 解决：在实验注册阶段固定 train-domain / held-out-domain split，不根据结果挑。

## 7. 当前代码包的边界

代码已经实现研究 harness：typed DAG、vLLM backend、matched collection、world-model ensemble、active design、risk-aware planner、motif miner、macro admission test、resume、安全日志、pipeline gate、sweep、表格聚合。

真正投稿前还需要实现/接入：

- 强 transformer-based state/architecture encoder world model；
- 官方 AutoMR / MetaFlow / HyEvo / BayesFlow 等基线；
- 严格 FLOP accounting；
- sandboxed code verifier；
- 更强 graph grammar mutation / learned program generator；
- OOD and cross-model protocol；
- 大规模结果与统计检验。

这不是缺陷隐藏，而是明确区分“实验系统已经可跑”和“论文结果尚未产生”。
