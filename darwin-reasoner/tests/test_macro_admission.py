from darwin_reasoner.evolution.admission import evaluate_macro_admission


def test_macro_admission_accepts_equal_reward_lower_cost():
    result = evaluate_macro_admission(
        macro_rewards=[1, 1, 0, 1] * 10,
        expanded_rewards=[1, 1, 0, 1] * 10,
        macro_tokens=[50] * 40,
        expanded_tokens=[100] * 40,
        noninferiority_margin=0.01,
        min_token_saving=0.1,
        bootstrap_draws=500,
    )
    assert result.admitted
