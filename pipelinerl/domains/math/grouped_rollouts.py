"""
Multi-response math rollout that produces generator/aggregator samples for LOO training.
"""

import random
import time
from typing import Sequence

from omegaconf import DictConfig
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.domains.math.rollouts import RewardTable, reward_from_status
from pipelinerl.domains.math.verifier_api import verify_answer_rpc
from pipelinerl.rollouts import BaseMetrics, RolloutResult, TrainingText
from pipelinerl.world import Job


def _build_generator_prompt(cfg: DictConfig, task: str, attempt: int) -> Prompt:
    """
    Construct the user/system prompts for generator attempt ``attempt``.

    Each attempt receives the base task template plus an optional suffix that
    encourages diverse reasoning.
    """
    system_prompt = getattr(cfg.actor, "system_prompt", None)
    task_template = cfg.actor.task_template
    attempt_prompt = getattr(cfg.actor, "generator_attempt_prompt", None)

    if "{task}" not in task_template:
        raise ValueError("actor.task_template must contain '{task}' for grouped math rollouts.")

    user_content = task_template.format(task=task)
    if attempt_prompt:
        user_content = f"{user_content}\n\n{attempt_prompt.format(attempt=attempt + 1)}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return Prompt(messages=messages)


def _format_generator_responses(outputs: Sequence[str], skip_index: int) -> str:
    """
    Concatenate generator responses except the one indexed by ``skip_index``.

    This builds the aggregator context for a specific leave-one-out evaluation.
    """
    sections = []
    for idx, content in enumerate(outputs):
        if idx == skip_index:
            continue
        cleaned = (content or "").strip()
        sections.append(f"Response #{idx + 1}:\n{cleaned}")

    combined = "\n\n".join(sections)
    if not combined:
        raise AssertionError("Aggregator prompt cannot be built without generator responses.")
    return combined


def _build_aggregator_prompt(cfg: DictConfig, task: str, responses: str) -> Prompt:
    """
    Construct the aggregator prompt given the task text and the visible responses.
    """
    template = getattr(cfg.actor, "aggregator_task_template", None)
    system_prompt = getattr(cfg.actor, "aggregator_system_prompt", None)
    if not template:
        raise ValueError("actor.aggregator_task_template must be configured for grouped math rollouts.")
    if "{task}" not in template or "{responses}" not in template:
        raise ValueError("actor.aggregator_task_template must contain '{task}' and '{responses}' placeholders.")

    user_content = template.format(
        task=task,
        responses=responses,
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return Prompt(messages=messages)


def _ensure_logprobs(sample: TrainingText, role: str) -> None:
    """Guardrail: grouped objectives require per-token logprobs for both policies."""
    if not sample.logprobs:
        raise AssertionError(
            f"{role} sample is missing logprobs. "
            "Set llm.collect_logprobs=True to enable grouped math rollouts."
        )


def _pick_environment_job(cfg: DictConfig) -> Job:
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    if not env_jobs:
        raise AssertionError("No environment jobs available for math verification.")
    return random.choice(env_jobs)


async def generate_grouped_math_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session,
) -> RolloutResult:
    """
    Generate one grouped math rollout consisting of K generator responses and K
    aggregator evaluations (one per leave-one-out subset).

    Returns:
        RolloutResult: includes ``2K`` TrainingText instances annotated with their
        metadata so the preprocessor can compute LOO rewards.
    """
    time_start = time.time()
    group_size = getattr(cfg.actor, "generator_group_size", None)
    if group_size is None or group_size < 2:
        raise ValueError("actor.generator_group_size must be >= 2 for grouped math rollouts.")

    rewards = RewardTable(**dict(cfg.rewards))
    env_job = _pick_environment_job(cfg)

    # Stage 1: sample generator responses.
    generator_calls = []
    for attempt in range(group_size):
        prompt = _build_generator_prompt(cfg, problem["task"], attempt)
        llm_call = await llm_async_generate(llm, prompt, session)
        generator_calls.append(llm_call)

    generator_texts: list[TrainingText] = []
    for idx, call in enumerate(generator_calls):
        text = make_training_text(llm, call)
        _ensure_logprobs(text, "Generator")
        # Reward placeholders are replaced by ``apply_group_objectives``.
        text.reward = 0.0
        text.metadata["role"] = "generator"
        text.metadata["generator_index"] = idx
        text.metadata["group_size"] = group_size
        generator_texts.append(text)

    generator_outputs = [call.output.content or "" for call in generator_calls]
    aggregator_texts: list[TrainingText] = []
    answer_statuses: list[str] = []

    # Stage 2: aggregate every leave-one-out subset.
    for leave_out in range(group_size):
        responses = _format_generator_responses(generator_outputs, leave_out)
        prompt = _build_aggregator_prompt(cfg, problem["task"], responses)
        llm_call = await llm_async_generate(llm, prompt, session)
        aggregator_text = make_training_text(llm, llm_call)
        _ensure_logprobs(aggregator_text, "Aggregator")

        # Evaluate aggregator output using the same verifier as the base math domain.
        answer_status = await verify_answer_rpc(
            session=session,
            host=env_job.hostname,
            port=env_job.port,
            prediction=llm_call.output.content or "",
            gold=problem["answer"],
            strict=True,
        )
        reward = reward_from_status(rewards, answer_status, aggregator_text.finished)
        aggregator_text.reward = reward
        included_indices = [idx for idx in range(group_size) if idx != leave_out]
        if not included_indices:
            raise AssertionError("Aggregator subset must include at least one generator response.")
        aggregator_text.metadata["role"] = "aggregator"
        aggregator_text.metadata["leave_out_index"] = leave_out
        aggregator_text.metadata["subset_size"] = len(included_indices)
        aggregator_text.metadata["group_size"] = group_size
        aggregator_text.metadata["generator_indices_included"] = included_indices
        aggregator_text.metadata["aggregator_status"] = answer_status
        # ``logprobs`` is per-token; summing produces the total log-likelihood of this answer.
        aggregator_text.metadata["aggregator_logprob"] = float(sum(aggregator_text.logprobs))
        aggregator_texts.append(aggregator_text)
        answer_statuses.append(answer_status)

    # Aggregate metrics so monitoring dashboards have a quick view of rollout quality.
    aggregator_rewards = [text.reward for text in aggregator_texts]
    avg_reward = float(sum(aggregator_rewards) / len(aggregator_rewards)) if aggregator_rewards else 0.0
    success_flags = [status == "correct" for status in answer_statuses]
    unparsable_flags = [status == "unparsable" for status in answer_statuses]
    no_answer_flags = [status == "no_answer" for status in answer_statuses]

    metrics = BaseMetrics(
        reward=avg_reward,
        success=any(success_flags),
        no_error=not any(unparsable_flags),
        no_answer=all(no_answer_flags) if no_answer_flags else False,
    )
    latency = time.time() - time_start

    return RolloutResult(
        training_texts=generator_texts + aggregator_texts,
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
    )
