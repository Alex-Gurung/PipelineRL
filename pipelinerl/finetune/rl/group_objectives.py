"""Utilities for computing generator rewards from grouped generator/aggregator rollouts."""

import logging
from collections import defaultdict
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _resolve_path(entry: dict[str, Any], path: str | None) -> Any:
    """
    Navigate a dotted path (e.g. ``metadata.role``) inside a nested dict.

    Returns ``None`` if any intermediary key is missing so callers can treat the
    entire field as optional.
    """
    if not path:
        return entry
    current: Any = entry
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _to_int(value: Any) -> int | None:
    """Best-effort int conversion that tolerates strings/booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    """Best-effort float conversion that tolerates strings."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    try:
        return float(value)
    except Exception:
        return None


class GroupRoleConfig(BaseModel):
    role_path: str = Field(
        default="metadata.role",
        description="Dotted path to the field describing whether an entry belongs to the generator or aggregator.",
    )
    generator_value: str = Field(default="generator", description="Value stored in role_path for generator samples.")
    aggregator_value: str = Field(default="aggregator", description="Value stored in role_path for aggregator samples.")
    generator_index_path: str = Field(
        default="metadata.generator_index",
        description="Dotted path to the generator index inside the group.",
    )
    aggregator_subset_path: str = Field(
        default="metadata.leave_out_index",
        description="Dotted path to the generator index removed for an aggregator subset evaluation.",
    )
    aggregator_skip_value: int | str | None = Field(
        default=None,
        description="Optional value that identifies aggregator rows that should be skipped (e.g. full-set runs).",
    )


class LOORewardConfig(BaseModel):
    version: Literal["expected_delta"] = Field(
        default="expected_delta",
        description="Which leave-one-out estimator to use. Only 'expected_delta' is currently supported.",
    )
    score_field: str = Field(
        default="reward",
        description="Dotted path to the metric used as aggregator score. Use 'reward' to reuse the RL reward.",
    )
    min_positive_samples: int = Field(
        default=1,
        description="Minimum number of positive (with-current-sample) subsets required to compute a generator reward.",
    )


class GroupObjectivesConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable generator/aggregator grouped objectives.")
    group_field: str = Field(default="group_id", description="Field containing the rollout group identifier.")
    roles: GroupRoleConfig = Field(default_factory=GroupRoleConfig)
    loo: LOORewardConfig = Field(default_factory=LOORewardConfig)
    debug_metadata_key: str = Field(
        default="group_objective",
        description="Metadata key used to store diagnostic information about the computed rewards.",
    )


def _score_entry(entry: dict[str, Any], score_field: str) -> float | None:
    """
    Fetch the scalar score used for LOO (either top-level reward or a metadata path).
    """
    if score_field == "reward":
        return _to_float(entry.get("reward"))
    return _to_float(_resolve_path(entry, score_field))


def apply_group_objectives(dataset: list[dict[str, Any]], config: GroupObjectivesConfig) -> list[dict[str, Any]]:
    """
    Rewrite generator rewards in-place using leave-one-out (LOO) scores.

    Args:
        dataset: List of rollout samples produced by the actor / preprocessor. Each
            entry is expected to contain ``group_id`` and metadata describing
            whether it belongs to the generator or aggregator.
        config: Grouped-objective configuration declared in the finetune config.
    """
    if not config.enabled:
        return dataset

    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for entry in dataset:
        group_id = entry.get(config.group_field)
        if group_id is None:
            logger.debug("Skipping entry without group id for grouped objectives")
            continue
        groups[group_id].append(entry)

    for group_id, entries in groups.items():
        _apply_group(entries, config, group_id)

    return dataset


def _apply_group(entries: list[dict[str, Any]], config: GroupObjectivesConfig, group_id: Any) -> None:
    """
    Compute generator rewards for a single group of entries.

    ``entries`` contain both generator and aggregator samples for the same
    rollout group id. Aggregator entries encode which generator index was removed
    and their score (reward/logprob). This helper collects those scores and
    assigns the expected-delta reward to the generator entries.
    """
    generator_entries: list[dict[str, Any]] = []
    aggregator_entries: list[dict[str, Any]] = []

    for entry in entries:
        role = _resolve_path(entry, config.roles.role_path)
        if role == config.roles.generator_value:
            generator_entries.append(entry)
        elif role == config.roles.aggregator_value:
            aggregator_entries.append(entry)

    assert generator_entries, f"Grouped objectives: no generator entries found for group {group_id}"
    assert aggregator_entries, f"Grouped objectives: no aggregator entries found for group {group_id}"

    # Collect scores indexed by which generator response was removed.
    subset_scores: dict[int, list[float]] = defaultdict(list)
    for entry in aggregator_entries:
        subset_idx_raw = _resolve_path(entry, config.roles.aggregator_subset_path)
        assert subset_idx_raw is not None, (
            f"Grouped objectives: aggregator entry missing subset index in group {group_id}"
        )
        if config.roles.aggregator_skip_value is not None and subset_idx_raw == config.roles.aggregator_skip_value:
            continue
        subset_idx = _to_int(subset_idx_raw)
        assert subset_idx is not None, (
            f"Grouped objectives: unable to parse subset index '{subset_idx_raw}' in group {group_id}"
        )
        score = _score_entry(entry, config.loo.score_field)
        assert score is not None, (
            f"Grouped objectives: missing score '{config.loo.score_field}' for aggregator sample in group {group_id}"
        )
        subset_scores[subset_idx].append(score)

    assert subset_scores, f"Grouped objectives: no usable aggregator subsets found for group {group_id}"

    # Reduce scores for each subset by taking the simple mean (expectation).
    reduced_subset_scores = {
        idx: float(mean(scores)) for idx, scores in subset_scores.items() if scores
    }

    if len(reduced_subset_scores) < 2:
        raise AssertionError(
            f"Grouped objectives: insufficient subset coverage for group {group_id} "
            f"(found {len(reduced_subset_scores)} subsets)"
        )

    for entry in generator_entries:
        idx_raw = _resolve_path(entry, config.roles.generator_index_path)
        gen_idx = _to_int(idx_raw)
        assert gen_idx is not None, (
            f"Grouped objectives: missing generator index for sample in group {group_id} (value={idx_raw})"
        )

        # ``expected_without`` corresponds to the aggregator run that omitted this generator sample.
        if gen_idx not in reduced_subset_scores:
            raise AssertionError(
                f"Grouped objectives: no aggregator subset excluding generator {gen_idx} in group {group_id}"
            )
        expected_without = reduced_subset_scores[gen_idx]

        # Any other subset implicitly includes this generator sample, so averaging
        # their scores gives ``E[s | included]``.
        with_scores = [score for subset_idx, score in reduced_subset_scores.items() if subset_idx != gen_idx]
        if len(with_scores) < config.loo.min_positive_samples:
            raise AssertionError(
                f"Grouped objectives: not enough positive subsets for generator {gen_idx} in group {group_id}"
            )

        expected_with = sum(with_scores) / len(with_scores)
        reward_delta = expected_with - expected_without

        entry["reward"] = reward_delta
        # Record diagnostics so downstream analysis can inspect LOO behavior.
        metadata = entry.setdefault("metadata", {})
        debug = metadata.setdefault(config.debug_metadata_key, {})
        debug.update(
            {
                "expected_with": expected_with,
                "expected_without": expected_without,
                "num_positive_subsets": len(with_scores),
                "score_field": config.loo.score_field,
            }
        )
