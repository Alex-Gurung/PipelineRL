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


def _normalize_text(s: str) -> str:
    """Lowercase, remove punctuation/articles/extra spaces, and round numbers if applicable."""

    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    def round_number(text: str) -> str:
        try:
            tmp = float(text)
            return f"{tmp:.3f}"
        except ValueError:
            return text

    return round_number(white_space_fix(remove_articles(remove_punc(lower(s)))))


def _exact_match(generation: str, reference: str) -> float:
    return 1.0 if _normalize_text(generation) == _normalize_text(reference) else 0.0


def _f1_score(generation: str, reference: str) -> float:
    """Token-level F1 without external deps (bag-of-words)."""
    g_tokens: List[str] = _normalize_text(generation).split()
    r_tokens: List[str] = _normalize_text(reference).split()
    if not g_tokens and not r_tokens:
        return 1.0
    if not g_tokens or not r_tokens:
        return 0.0
    # multiset overlap
    from collections import Counter

    g_c = Counter(g_tokens)
    r_c = Counter(r_tokens)
    overlap = sum((g_c & r_c).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(g_c.values()))
    recall = overlap / max(1, sum(r_c.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


async def generate_scitrek_rollout(
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
    matches = COMPILED_REGEX.findall(content)
    answer = matches[-1] if matches else ""

    # Clean spacing like original
    answer = answer.strip()
    answer = " ".join(answer.split())
    answer = ", ".join([tmp.strip() for tmp in answer.split(",")])

    gold = str(problem.get("answer", ""))
    em = _exact_match(answer, gold)
    f1 = _f1_score(answer, gold)
    reward = float(em + f1)

    trace = make_training_text(llm, llm_call)
    trace.reward = reward

    metrics = BaseMetrics(
        reward=reward,
        success=bool(em == 1.0),
        no_error=True,
        no_answer=(answer == ""),
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
    )

