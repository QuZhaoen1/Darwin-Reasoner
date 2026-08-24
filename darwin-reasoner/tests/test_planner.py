from darwin_reasoner.config import SearchConfig
from darwin_reasoner.dsl import chain_architecture
from darwin_reasoner.schema import OperatorKind, WorldModelPrediction
from darwin_reasoner.search.planner import BudgetConditionedPlanner


def test_planner_prefers_higher_reward_at_same_cost():
    planner = BudgetConditionedPlanner(SearchConfig(risk_beta=0.0, token_penalty=0.0, latency_penalty=0.0))
    a = chain_architecture("a", [OperatorKind.SOLVE])
    b = chain_architecture("b", [OperatorKind.VERIFY])
    preds = [
        WorldModelPrediction(reward_mean=0.6, reward_std=0.1, tokens_mean=100, latency_mean=1, success_prob=0.6),
        WorldModelPrediction(reward_mean=0.8, reward_std=0.1, tokens_mean=100, latency_mean=1, success_prob=0.8),
    ]
    ranked = planner.rank([a, b], preds)
    assert ranked[0].architecture.name == "b"
