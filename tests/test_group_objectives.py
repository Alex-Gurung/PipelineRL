from pipelinerl.finetune.rl.group_objectives import (
    GroupObjectivesConfig,
    GroupRoleConfig,
    LOORewardConfig,
    apply_group_objectives,
)


def _make_generator_entry(group_id: str, index: int) -> dict:
    return {
        "group_id": group_id,
        "metadata": {"role": "generator", "generator_index": index},
        "reward": 0.0,
    }


def _make_aggregator_entry(group_id: str, leave_out: int, score: float) -> dict:
    metadata = {
        "role": "aggregator",
        "leave_out_index": leave_out,
        "aggregator_logprob": score,
    }
    return {
        "group_id": group_id,
        "metadata": metadata,
        "reward": score,
    }


def test_expected_delta_rewards():
    config = GroupObjectivesConfig(
        enabled=True,
        loo=LOORewardConfig(score_field="metadata.aggregator_logprob"),
    )
    dataset: list[dict] = []
    group_id = "math"
    for idx in range(3):
        dataset.append(_make_generator_entry(group_id, idx))
    scores = {-1: 0.0, 0: -5.0, 1: -2.0, 2: -1.0}
    for leave_out in (0, 1, 2):
        dataset.append(_make_aggregator_entry(group_id, leave_out, scores[leave_out]))

    updated = apply_group_objectives(dataset, config)
    rewards = {
        entry["metadata"]["generator_index"]: entry["reward"]
        for entry in updated
        if entry["metadata"]["role"] == "generator"
    }

    assert rewards[0] == (-2.0 + -1.0) / 2 - (-5.0)
    assert rewards[1] == (-5.0 + -1.0) / 2 - (-2.0)
    assert rewards[2] == (-5.0 + -2.0) / 2 - (-1.0)


def test_skip_value_ignores_full_set_samples():
    role_cfg = GroupRoleConfig(aggregator_skip_value=-1)
    config = GroupObjectivesConfig(
        enabled=True,
        roles=role_cfg,
        loo=LOORewardConfig(score_field="metadata.aggregator_logprob"),
    )
    dataset: list[dict] = []
    group_id = "math"
    for idx in range(2):
        dataset.append(_make_generator_entry(group_id, idx))
    dataset.append(_make_aggregator_entry(group_id, 0, -4.0))
    dataset.append(_make_aggregator_entry(group_id, 1, -2.0))
    dataset.append(_make_aggregator_entry(group_id, -1, 42.0))  # should be ignored

    updated = apply_group_objectives(dataset, config)
    rewards = {
        entry["metadata"]["generator_index"]: entry["reward"]
        for entry in updated
        if entry["metadata"]["role"] == "generator"
    }

    assert rewards[0] == (-2.0) - (-4.0)
    assert rewards[1] == (-4.0) - (-2.0)
