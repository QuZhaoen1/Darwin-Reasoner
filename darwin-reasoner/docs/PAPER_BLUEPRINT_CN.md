# ICLR Paper Blueprint

## 暂定标题

**DarwinReasoner: Counterfactual World Models for Self-Expanding Inference Architecture Search**

更抓眼球的备选标题：

**Can Language Models Learn How to Invent Their Own Inference Algorithms?**

副标题仍建议保留 Counterfactual World Models，避免标题只有概念没有方法锚点。

## 核心论点

现有 test-time scaling 通常扩展某个预设推理过程的长度、宽度或采样数；workflow-search 方法则在预定义 operator/search space 内找结构。DarwinReasoner 研究更高一层的问题：模型能否学习不同 computation architecture 的后果，并利用这种预测来合成、修改和扩展自己的 inference algorithm search space。

## 三个核心贡献，论文正文只围绕它们展开

### C1. Matched Interventional Reasoning Atlas

从相同 reasoning state 出发，对多个 inference architectures 做 matched randomized rollouts，形成包含 success、tokens、latency、model、budget、topology 的 paired supervision。

### C2. Counterfactual Architecture World Model + Budgeted Compiler

学习 architecture consequence model，并在 token/latency/risk budget 下进行 architecture selection / online replanning。核心不是预测答案，而是预测 computation 的后果。

### C3. Validated Search-Space Evolution

从经验中挖 operator motif，新 macro 必须经过 matched non-inferiority / efficiency admission 才能进入 grammar，再用 regret distillation 把昂贵 search 经验内化成 cheap prior。

Active design、cross-model transfer、experience-time scaling 都作为这三项贡献的自然推论和实验，而不是额外平铺贡献。

## Introduction 的逻辑

1. Test-time compute 有效，但 compute 的位置和形式同样重要。
2. “难题多想”不等于“额外思考有价值”。
3. 最近方法可以搜索 DAG/workflow，但通常依赖固定 search grammar、真实执行每个候选或无法建模不同 inference architecture 的 paired consequence。
4. 因此提出新问题：**Can a model predict the consequence of an inference algorithm before spending the compute, and can that experience expand the algorithm space itself?**
5. DarwinReasoner 用 matched interventions 学 world model，用 world model 编译 architecture，用通过验证的 motifs 扩展 grammar。

## Figure 1

左侧：一个固定 reasoning state。

中间：8 个平行 architecture intervention，展示 Deep Chain / Wide Branch / Verify-heavy / Repair / Tool / Debate / Generated Graph / Evolved Macro。

中央上：Counterfactual Architecture World Model 预测 reward/cost/uncertainty。

右侧：Budgeted Compiler 选择 architecture，执行后 observation 回流。

下方：Experience Archive -> Motif Miner -> Matched Admission -> Grammar v(t+1) -> Distillation。

整个图必须一眼看出“search space 也会长”。

## Main Table

行：

- CoT
- Self-Consistency
- strong tree/search baseline
- AutoMR
- BayesFlow
- HyEvo
- MetaFlow
- strongest submission-time concurrent baseline
- DarwinReasoner

列：

- MATH500 / AIME / OlympiadBench / LiveCodeBench / Logic benchmark accuracy
- Avg Tokens
- p95 Tokens
- Model Calls
- Search Overhead
- Latency
- GPU-hours

不要只给平均分。

## Figure 2: Accuracy-Compute Pareto

横轴 tokens/FLOPs，纵轴 accuracy。不同预算 2K/4K/8K/16K/32K 形成完整曲线。

主 claim 应该是 frontier dominance，而不是只报最高预算一个点。

## Figure 3: World Model Predicts the Value of Thinking

每个 held-out reasoning state 有 K 个真实 architecture outcome。横轴 predicted value，纵轴 realized value，附 Spearman / calibration。

再做一张 regret CDF：world model 选的 architecture 距 oracle 差多少。

## Figure 4: Grammar Evolution

横轴 accumulated experience，左 y 轴新 macro 数/存活率，右 y 轴 future-task search cost。展示一些真正被 admission 接受和拒绝的 macro。

## Figure 5: OOD Architecture Discovery

训练时没见 Code。测试 Code 时展示几个自动生成 DAG，与训练期常见 Math/Logic DAG 做结构对比。

## Figure 6: Experience-Time Scaling

固定 future-test set，随着历史 paired states 从 0 增长到 N：

- accuracy；
- search rollouts；
- tokens；
- world-model regret。

目标结果：accuracy 上升或持平，同时 search cost 下降。

## 理论部分可以做什么

不建议硬造一个“全局最优”定理。更稳的理论线有两条：

1. 在有限 architecture candidate set 下，将 planner 写为 constrained decision problem，给出 world-model uniform error epsilon 时 selected architecture 的 regret upper bound；
2. Macro admission 的 paired non-inferiority decision 给出 bootstrap / concentration-based acceptance guarantee。

如果理论证明做不漂亮，宁可把正文理论控制在一个 proposition + appendix proof，避免为理论而理论。

## 必须预注册的失败条件

如果 pilot 中：

- world-model within-state Spearman < 0.2；或
- selected-architecture regret 接近 random；或
- matched-compute accuracy 不优于 strongest baseline；

则不要直接全量 sweep。先诊断 state representation、architecture encoding、intervention diversity。

如果 macro evolution 没贡献，不要把它硬写成主结果，可以降为 analysis；但 paired-intervention + architecture-world-model 主线必须成立。

## 投稿前 novelty audit

至少在投稿前 2 周重新检索：test-time scaling、workflow search、meta-reasoning DAG、self-evolving agents、world models for reasoning、automatic algorithm discovery。2026 年这个方向更新太快，当前 novelty matrix 不能直接沿用到投稿日。
