import re
import string
import time
from typing import List

import aiohttp
from omegaconf import DictConfig
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import BaseMetrics, RolloutResult

COMPILED_REGEX = re.compile(r"\\boxed\{(.*?)\}")


import torch
import re

COMPILED_REGEX = re.compile(r"\\boxed\{(.*?)\}")

def get_distance(generation, reference):
    try:
        gen_score = float(generation)
    except:
        return 25
    ref_score = float(reference)
    distance = (gen_score-ref_score) ** 2
    if gen_score < 1 or gen_score > 5:
        distance = 25
    return distance


def reward_func(generation: str, reference: str):
    """
    Reward function for calculating rewards of model outputs.

    Args:
        queries (torch.Tensor): Complete text sequences containing prompts and responses
        reference (str): Ground truth answer
    """
    matches = COMPILED_REGEX.findall(generation)
    generation = matches[-1] if matches else ""
    reward = -25 # worst dist is 25
    if len(generation) == 1: # only support 1 character answer
        dist = get_distance(generation, reference)
        reward = -1 * dist
    return reward


async def generate_semeval_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    """One-shot rollout using dataset messages and EM+F1 reward on boxed answer."""
    # Use dataset-provided messages
    messages = list(problem.get("messages", []))

    # Prepend system prompt if configured and not already present
    # if cfg.actor.system_prompt:
    #     if not (messages and messages[0].get("role") == "system"):
    #         messages = [{"role": "system", "content": cfg.actor.system_prompt}] + messages

    prompt = Prompt(messages=messages)
    start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - start

    content = llm_call.output.content or ""

    gold = str(problem.get("answer", ""))
    reward = reward_func(content, gold)

    trace = make_training_text(llm, llm_call)
    trace.reward = reward

    # we count it as a success if the distance is under 1
    # the reward is the negative distance
    success = bool(reward > -1)
    # when we seen an error/no answer we use -25 as reward
    no_error = bool(reward > -25)
    no_answer = bool(reward == -25)
    metrics = BaseMetrics(
        reward=reward,
        success=success,
        no_error=no_error,
        no_answer=no_answer,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
    )
