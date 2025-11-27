# Grouped Generator + Aggregator Objectives

This repository now supports training *two* cooperating policies with a single
rollout: a **generator** that emits a set of diverse answers and an
**aggregator** that reads that set (or a subset of it) and produces the final
task answer. The trainer uses the same RL algorithm (DAPO/GRPO-compatible) for
both policies while keeping their rewards separate.

The workflow is intentionally two-stage:

1. **Generator stage** – sample `K` responses for every question. Each response
   keeps its own logprobs/value estimates so it can later receive an
   advantage/reward update.
2. **Aggregator stage** – run `K` aggregation passes. For pass `j`, feed the
   question plus all generator responses *except* response `j`. The aggregated
   answer is judged (direct reward or an LLM-as-a-judge score), and that score
   is stored on the aggregator sample. The aggregator policy is trained via the
   usual RL loss on these judged samples.
3. **Generator LOO reward** – for generator response `i`, compute:

   \[
   r_i = \mathbb{E}[s \mid i \text{ included}] - \mathbb{E}[s \mid i \text{ left out}]
   \]

   where `s` is the aggregator score. The *negative* expectation is the single
   aggregator pass in which response `i` was removed. The *positive*
   expectation averages the `K-1` passes where `i` was still present (all other
   leave-one-out subsets). This directly captures the marginal utility of each
   generator sample without introducing custom reward heads.

## Implementation Notes

- Generator/aggregator identity is carried through `TrainingText.metadata`.
  Required metadata fields (names are configurable):
  - `metadata.role`: `"generator"` or `"aggregator"`.
  - `metadata.generator_index`: index of the generator response inside the group.
  - `metadata.leave_out_index`: index of the generator response removed for an
    aggregator subset run (set to `None`/`-1` for full-set runs).
- Aggregator subset scores can reuse the normal RL reward or any scalar stored
  in metadata (e.g., a judge logprob). These scores drive both the aggregator
  loss and the LOO delta for the generator.
- The preprocessor injects diagnostic metadata (`metadata.group_objective` by
  default) so you can inspect `expected_with`, `expected_without`, and how many
  positive subsets contributed to each generator reward.

## Configuration

`conf/finetune/base.yaml` now exposes a `rl.group_objectives` block:

```yaml
rl:
  group_objectives:
    enabled: true
    group_field: group_id                # how samples are grouped
    roles:
      role_path: metadata.role           # where generator/aggregator labels live
      generator_value: generator
      aggregator_value: aggregator
      generator_index_path: metadata.generator_index
      aggregator_subset_path: metadata.leave_out_index
      aggregator_skip_value: null        # skip value for full-set aggregator runs
    loo:
      version: expected_delta            # currently the only supported variant
      score_field: reward                # or e.g. metadata.judge_logprob
      min_positive_samples: 1            # require at least this many “with” subsets
    debug_metadata_key: group_objective
```

Key knobs:

- **`score_field`** – switch between using the standard reward (`reward`) or
  any dotted metadata path (e.g., `metadata.aggregator_logprob`) to measure how
  useful an aggregator pass was. This makes it easy to train with logprob-based
  metrics without extra reward heads.
- **`min_positive_samples`** – control how many positive (response-including)
  subsets must exist before we trust the reward; if not satisfied, the
  generator sample is left untouched.
- **`aggregator_skip_value`** – mark aggregator rows that should *not* be used
  for generator rewards (e.g., set to `-1` for the full-set pass that is only
  used to train the aggregator itself).

Once these fields are populated the preprocessing stage automatically rewrites
generator rewards based on the configured LOO estimator, and the training loop
proceeds without additional changes.

## Ready-Made Config

Use `conf/finetune/group_objectives.yaml` to enable all grouped-objective logic
without touching the base config. It inherits from `conf/finetune/base.yaml`,
turns on `rl.group_objectives.enabled`, skips aggregator rows marked with
`metadata.leave_out_index = -1`, and points the LOO score at
`metadata.aggregator_logprob` (so you can reward generator samples using the
aggregator’s own answer likelihood instead of the global reward).

## Math Domain Example

`conf/math_grouped.yaml` wires the grouped rollout logic into the math domain:
it swaps the rollout policy to
`pipelinerl.domains.math.grouped_rollouts.generate_grouped_math_rollout`,
defines generator/aggregator prompts, and chooses a group size of three.
Because it also imports `finetune: group_objectives`, running with this config
produces generator samples whose rewards are automatically overwritten using
the aggregator’s log-likelihood deltas.

### Rollout Workflow

1. **Scheduler** (`actor.schedule_rollouts`) launches the configured rollout
   policy for each problem. For `math_grouped.yaml` that policy is
   `generate_grouped_math_rollout`.
2. **Generator phase**:
   - The policy calls `_build_generator_prompt` `K` times (one per generator
     attempt) and issues `K` LLM calls through `llm_async_generate`.
   - Each `TrainingText` is annotated with `metadata.role=generator`,
     `metadata.generator_index`, and placeholder reward `0.0`.
3. **Aggregator phase**:
   - For each leave-one-out index `j`, `_format_generator_responses` removes
     response `j`, `_build_aggregator_prompt` constructs a prompt that only
     references the remaining responses, and the aggregator LLM call runs.
   - The verifier RPC scores the aggregator answer. These samples are tagged
     with `metadata.role=aggregator`, `metadata.leave_out_index=j`,
     `metadata.aggregator_logprob`, and receive their actual RL reward.
4. **Result assembly**:
   - The policy returns a single `RolloutResult` containing the entire
     generator+aggregator set (group id is filled downstream in `actor.py`).
5. **Preprocessing**:
   - `preprocess_dataset` applies `prepare_rl_fields`, then
     `apply_group_objectives` which reads the stored metadata, computes the
     expected-delta rewards for generator samples, and finally
     `populate_rl_data` packs everything into token-level tensors.
6. **Trainer**:
   - The RL trainer consumes the batches and optimizes the generator and
     aggregator policies with the same PPO/DAPO/GRPO loss, each using its own
     reward signal (LOO deltas vs. task scores).

See `tests/test_group_objectives.py` for unit tests that exercise the expected
delta reward computation on synthetic math groups.
