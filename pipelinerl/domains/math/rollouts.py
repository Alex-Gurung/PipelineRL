import logging
import time
import random
from typing import Tuple

import aiohttp
import torch
from omegaconf import DictConfig
from pydantic import BaseModel
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.world import Job
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from .verifier_api import verify_answer_rpc

logger = logging.getLogger(__name__)


class Metrics(BaseMetrics):
    penalty: float
    likelihood_score: float = 0.0
    likelihood_reward: float = 0.0

class RewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float
    likelihood_weight: float = 0.0
    buffer_tokens: int = 0 # 0 means no overlong reward shaping


def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    """
    Compute the overlong penalty
    """
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.


def split_reasoning_and_answer_tokens(
    tokenizer,
    generated_token_ids: list[int],
    output_text: str,
) -> Tuple[list[int], list[int]]:
    """
    Split generated token ids into reasoning tokens and the tokens that correspond to the final boxed answer.
    """
    if not generated_token_ids:
        return [], []

    boxed_index = output_text.rfind("\\boxed{")
    if boxed_index < 0:
        return generated_token_ids, []

    decoded_tokens = [
        tokenizer.decode([token_id], skip_special_tokens=False)
        for token_id in generated_token_ids
    ]
    cumulative_text = ""
    answer_start_idx = len(generated_token_ids)
    for idx, token_text in enumerate(decoded_tokens):
        cumulative_length = len(cumulative_text)
        cumulative_text += token_text
        if boxed_index < len(cumulative_text):
            answer_start_idx = idx
            break

    reasoning_tokens = generated_token_ids[:answer_start_idx]
    answer_tokens = generated_token_ids[answer_start_idx:]
    return reasoning_tokens, answer_tokens


async def generate_math_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": cfg.actor.task_template.format(task=problem["task"])})
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    rewards = RewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor

    # math_verify is a fast environment, no support for environment replicas for now
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    # choose the job randomly
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    answer_status = await verify_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        prediction=llm_call.output.content,
        gold=problem["answer"],
        strict=True,
    )

    trace = make_training_text(llm, llm_call)
    # Determine reward based on answer status and finished state
    match (answer_status, trace.finished):
        case ("wrong", False):
            reward = rewards.wrong_answer_not_finished
        case ("wrong", True):
            reward = rewards.wrong_answer_finished
        case ("no_answer", False):
            reward = rewards.no_answer_not_finished
        case ("no_answer", True):
            reward = rewards.no_answer_finished
        case ("unparsable", False):
            reward = rewards.unparsable_not_finished
        case ("unparsable", True):
            reward = rewards.unparsable_finished
        case ("correct", False):
            reward = rewards.correct_answer_not_finished
        case ("correct", True):
            reward = rewards.correct_answer_finished
        case _:
            raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{trace.finished}")

    # Apply discount factor based on output length
    reward *= discount_factor**llm_call.output_length_tokens
    overlong_penalty = 0
    if rewards.buffer_tokens > 0:
        overlong_penalty = length_penalty(llm.parameters['max_tokens'], llm_call.output_length_tokens, rewards.buffer_tokens)
    reward += overlong_penalty
    trace.reward = reward

    metrics = Metrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
        penalty=overlong_penalty,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency, 
        dataset_name=problem.get("dataset"),
    )


async def generate_math_rollout_with_likelihood(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": cfg.actor.task_template.format(task=problem["task"])})
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    rewards = RewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor

    # math_verify is a fast environment, no support for environment replicas for now
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    # choose the job randomly
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    answer_status = await verify_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        prediction=llm_call.output.content,
        gold=problem["answer"],
        strict=True,
    )

    trace = make_training_text(llm, llm_call)
    # Determine reward based on answer status and finished state
    match (answer_status, trace.finished):
        case ("wrong", False):
            reward = rewards.wrong_answer_not_finished
        case ("wrong", True):
            reward = rewards.wrong_answer_finished
        case ("no_answer", False):
            reward = rewards.no_answer_not_finished
        case ("no_answer", True):
            reward = rewards.no_answer_finished
        case ("unparsable", False):
            reward = rewards.unparsable_not_finished
        case ("unparsable", True):
            reward = rewards.unparsable_finished
        case ("correct", False):
            reward = rewards.correct_answer_not_finished
        case ("correct", True):
            reward = rewards.correct_answer_finished
        case _:
            raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{trace.finished}")

    # Compute VR-like likelihood reward component using 1/perplexity heuristic on the ground-truth answer
    likelihood_score = 0.0
    likelihood_reward = 0.0
    generated_token_ids = trace.input_ids[-len(trace.logprobs) :] if trace.logprobs else []
    prompt_token_ids = trace.input_ids[: -len(trace.logprobs)] if trace.logprobs else trace.input_ids
    reasoning_tokens, _ = split_reasoning_and_answer_tokens(
        llm.tokenizer, generated_token_ids, trace.output_text
    )
    context_token_ids = prompt_token_ids + reasoning_tokens

    gold_answer = problem.get("answer", "")
    try:
        gold_completion_token_ids = llm.tokenizer.encode(gold_answer.strip(), add_special_tokens=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to tokenize gold answer '{gold_answer}'") from exc

    gold_logprobs: list[float] | None = None
    raw_gold_logprob_response = None
    if gold_completion_token_ids:
        try:
            raw_gold_logprob_response = llm.get_batch_logprobs_token_ids(
                [context_token_ids],
                [gold_completion_token_ids],
            )
            token_entries = []
            if raw_gold_logprob_response:
                first = raw_gold_logprob_response[0]
                if isinstance(first, dict):
                    token_entries = first.get("content", [])
                else:
                    token_entries = first
            gold_logprobs = [
                entry["logprob"] for entry in token_entries if entry and "logprob" in entry
            ]
            if gold_logprobs:
                logprob_tensor = torch.tensor(gold_logprobs, dtype=torch.float32)
                # 1 / perplexity = exp(mean logprob)
                score = torch.exp(logprob_tensor.mean()).clamp_(min=0.0, max=1.0)
                likelihood_score = score.item()
                likelihood_reward = rewards.likelihood_weight * likelihood_score
        except Exception as exc:
            raise RuntimeError("Failed to compute likelihood score from vLLM logprob API") from exc
        if not gold_logprobs:
            raise RuntimeError(
                "No gold logprobs returned from vLLM logprob API. "
                f"Context tokens: {len(context_token_ids)}, gold tokens: {len(gold_completion_token_ids)}, "
                f"raw response: {raw_gold_logprob_response}"
            )

    # Apply discount factor based on output length
    reward *= discount_factor**llm_call.output_length_tokens
    overlong_penalty = 0
    if rewards.buffer_tokens > 0:
        overlong_penalty = length_penalty(llm.parameters['max_tokens'], llm_call.output_length_tokens, rewards.buffer_tokens)
    reward += overlong_penalty
    reward += likelihood_reward
    trace.reward = reward
    if trace.metadata is None:
        trace.metadata = {}
    trace.metadata.update(
        {
            "likelihood_score": likelihood_score,
            "likelihood_reward": likelihood_reward,
            "likelihood_gold_logprobs": gold_logprobs,
        }
    )

    metrics = Metrics(
        reward=reward,
        success=answer_status == "correct",
        no_error=answer_status != "unparsable",
        no_answer=answer_status == "no_answer",
        penalty=overlong_penalty,
        likelihood_score=likelihood_score,
        likelihood_reward=likelihood_reward,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency, 
        dataset_name=problem.get("dataset"),
    )
