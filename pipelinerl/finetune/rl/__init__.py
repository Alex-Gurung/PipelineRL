import logging
import os
from functools import partial
from typing import Any
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import PreTrainedModel
from pipelinerl.finetune.types import PipelineBatchEncoding

from .utils import (
    sum_sum,
    mean_sum,
    replace_dataset_column,
)

# FIXME: remove a warnings, but might be worth investigating
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)

RL_DATA_COLUMNS = [
    "overflow",
    "group_tokens",
    "num_labels",
    "rewards",
    "advantages",
    "old_logprobs",
    "ref_logprobs",
    "long_prompt_ref_logprobs",
]


class RLConfig(BaseModel):
    policy_loss: str = Field(
        default="ppo",
        description="Policy Loss to use for RL",
        choices=["ppo", "reinforce"],
    )
    use_advantages: bool = Field(
        default=True,
        description="Use advantages instead of rewards to compute the loss",
    )
    epsilon: float = Field(default=0.2, description="Clip parameter for the ration of log probs")
    batch_size: int = Field(default=0, description="Batch size is required for normalization")
    reward_minus_kl_coef: float = Field(
        default=0.0,
        # https://arxiv.org/abs/2402.14740
        description="Implicit KL coefficient similar to the RLOO paper",
    )
    kl_coef: float = Field(
        default=0.1,
        description="KL penalty coefficient with reference policy",
    )
    final_kl_coef: float = Field(
        default=0.1,
        description="Final KL penalty coefficient value",
    )
    entropy_bonus: float = Field(
        default=0.0,
        description="Entropy bonus coefficient",
    )
    final_entropy_bonus: float = Field(
        default=0.0,
        description="Final entropy bonus value",
    )
    relu_log_p_weights: bool = Field(
        default=False,
        description="ReLU the weights before updating the model",
    )
    clamp_log_ratio_ref_new_value: float = Field(
        default=10,
        description="Clamp the log ratio ref new value",
    )
    divide_advantage_by_std: bool = Field(
        default=True,
        description="Normalize the advantage by the standard deviation",
    )
    overlong_filtering: bool = Field(default=False, description="Filter out sequence that do not have eos_token_id")
    group_normalization: bool = Field(
        default=False,
        description="Divide the weight of each sequence by the (average) number of tokens in the group",
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature for the training log probs",
    )
    filter_zero_advantage_groups: bool = Field(
        default=False,
        description="Filter out groups where all advantages are zero during preprocessing",
    )
    # DAPO options
    clip_ratio_high: float | None = Field(
        default=None,
        description="Upper clip bound for asymmetric clipping (DAPO). If None, uses epsilon for symmetric clipping.",
    )
    dynamic_filtering_reward_range: tuple[float, float] | None = Field(
        default=None,
        description="Reward range [low, high] for DAPO-style filtering. If set, filters groups where average reward is outside this range.",
    )
    # Multi-prompt training options
    enable_long_prompt_is: bool = Field(
        default=False,
        description=(
            "Enable importance sampling correction to train on long prompts while sampling from short prompts. "
            "Applies weight π_new(a|long)/π_new(a|short) to correct the distribution mismatch. "
            "Use this when you want a single RL objective but need to correct for the prompt format difference. "
            "When combined with enable_long_prompt_rl, the same correction is applied to both short and long losses. "
            "Requires 'messages_long' field in dataset."
        ),
    )
    enable_long_prompt_rl: bool = Field(
        default=False,
        description=(
            "Add a separate RL objective for long prompts with its own importance sampling correction. "
            "This creates a dual objective: L = (1-w)*L_short + w*L_long where w=long_prompt_rl_weight. "
            "Use this to explicitly optimize performance on both prompt formats. "
            "Can be combined with enable_long_prompt_is for a fully corrected dual objective. "
            "Requires 'messages_long' field in dataset."
        ),
    )
    long_prompt_rl_weight: float = Field(
        default=0.5,
        description=(
            "Weight for the long-prompt RL objective when enable_long_prompt_rl=True. "
            "Final loss = (1-weight)*loss_short + weight*loss_long. "
            "Set to 0.5 for equal weighting, higher values prioritize long-prompt performance."
        ),
    )
    long_prompt_seq_length: int | None = Field(
        default=None,
        description=(
            "Maximum sequence length for long prompts during preprocessing. "
            "If None, uses the same seq_length as short prompts. "
            "Set this to a higher value (e.g., 120000) when long prompts are significantly longer than short prompts "
            "to avoid excessive truncation while keeping short prompts memory efficient. "
            "Only applies when multi-prompt training is enabled."
        ),
    )
    # Knowledge distillation options
    enable_reasoning_distillation: bool = Field(
        default=False,
        description=(
            "Add KL divergence loss to keep reasoning consistent across prompt formats. "
            "Minimizes KL(π_new(·|short) || π_new(·|long)) to ensure the model produces similar "
            "token distributions regardless of whether it sees short or long prompts. "
            "This is a consistency loss, not traditional knowledge distillation. "
            "Requires 'messages_long' field in dataset."
        ),
    )
    distillation_coef: float = Field(
        default=0.1,
        description=(
            "Weight coefficient for the reasoning consistency loss term. "
            "Final loss includes: distillation_coef * KL(π_new(·|short) || π_new(·|long)). "
            "Higher values more strongly enforce consistency between prompt formats."
        ),
    )
    distillation_top_k: int | None = Field(
        default=None,
        description=(
            "If set, only match the top-k logits when computing KL divergence for distillation. "
            "This filters out low-probability tokens that may add noise to the KD loss. "
            "None means use the full distribution over all vocabulary tokens. "
            "Typical values: 10-50 for focused distillation."
        ),
    )
    value_loss_coef: float = Field(
        default=0.0,
        description="Coefficient for the value loss in the final loss",
    )


def make_rl_data_callback(args, current_dir, rl_config, model):
    if rl_config:
        populate_rl_data_ = partial(
            populate_rl_data,
            config=rl_config,
        )
    else:
        populate_rl_data_ = None
    return populate_rl_data_


def linear_decay_coef(current_step: int, max_step: int, initial_coef: float, final_coef: float) -> float:
    """
    Linearly decay the coefficient from initial to final value over the course of training.

    Args:
        current_step (int): Current step in the training
        max_step (int): Maximum number of steps in the training
        initial_coef (float): Initial coefficient value
        final_coef (float): Final coefficient value

    Returns:
        float: Linearly decayed coefficient value

    """
    return initial_coef + (final_coef - initial_coef) * current_step / max_step


def rl_step(
    model: PreTrainedModel,
    batch: PipelineBatchEncoding,
    current_step: int,
    max_step: int,
    config: RLConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Perform a single RL step on the model using the given batch and config.
    Handles both packed and unpacked sequences.

    Args:
        model (PreTrainedModel): The model to train
        batch (PipelineBatchEncoding): Batch of data containing rewards, advantages, masks, input_ids etc.
        current_step (int): Current training step
        max_step (int): Maximum number of training steps
        config (RLConfig): Configuration for the RL training

    Returns:
        tuple[torch.Tensor, dict[str, float]]: Loss tensor and metrics dictionary
    """
    # pre-compute masks
    masks = batch.labels != -100
    masks_shifted = masks[:, 1:]

    has_value_head = hasattr(model, 'value_head')

    # if we have position_ids, we are packing
    if batch.is_packed:
        position_ids = batch.position_ids[0]
        is_sequence_start = position_ids == 0
        # For computing the loss we will consider the first token the beginning of the sequence,
        # even if currently we are in the middle of a sequence.
        is_sequence_start[0] = True 
        sequence_starts = torch.where(is_sequence_start)[0]
        seq_boundaries = torch.cat(
            [
                sequence_starts,
                torch.tensor([position_ids.shape[0]], device=position_ids.device),
            ]
        )
        num_sequences = len(sequence_starts)

        # ensure we have valid sequence boundaries
        assert num_sequences > 0, "No sequences found in packed batch"
        assert seq_boundaries[-1] == position_ids.shape[0], "Sequence boundaries don't match input length"

        # pre-compute segment boundaries
        segments = list(zip(seq_boundaries[:-1], seq_boundaries[1:]))
    else:
        num_sequences = masks.shape[0]
        segments = None

    # Assert labels align with input_ids (ignoring masked positions)
    def _assert_labels_align(input_ids: torch.Tensor, labels: torch.Tensor | None, name: str):
        if labels is None:
            return
        assert input_ids.size() == labels.size(), (
            f"{name}: input_ids shape {tuple(input_ids.size())} != labels shape {tuple(labels.size())}"
        )
        mask = labels != -100
        if torch.any(mask):
            a = labels[mask]
            b = input_ids[mask]
            assert torch.equal(a, b), f"{name}: labels differ from input_ids on unmasked positions"

    _assert_labels_align(batch.input_ids, batch.labels, name="short")
    if getattr(batch, "long_prompt_input_ids", None) is not None and getattr(batch, "long_prompt_labels", None) is not None:
        _assert_labels_align(batch.long_prompt_input_ids, batch.long_prompt_labels, name="long")

    # Do not pass labels to model forward to ensure logits are returned by all backends/kernels
    model_inputs = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
    }
    if batch.is_packed:
        model_inputs["position_ids"] = batch.position_ids
    
    # Add visual features if present (for multimodal models)
    if hasattr(batch, 'pixel_values') and batch.pixel_values is not None:
        model_inputs["pixel_values"] = batch.pixel_values
    if hasattr(batch, 'image_grid_thw') and batch.image_grid_thw is not None:
        model_inputs["image_grid_thw"] = batch.image_grid_thw #torch.tensor(.reshape((1, 3))
    
    outputs = model(**model_inputs)

    # compute log probs and entropy for short prompts
    logits_short = outputs.logits[:, :-1, :] / config.temperature
    logprobs_short = F.log_softmax(logits_short, dim=-1)
    probs_short = F.softmax(logits_short, dim=-1)
    entropy = -(probs_short * logprobs_short).sum(dim=-1)

    # get log probs for actual tokens (short context)
    new_logprobs = torch.gather(logprobs_short, dim=2, index=batch.input_ids[:, 1:].unsqueeze(2)).squeeze(2)
    assert torch.isfinite(new_logprobs).all(), f"new_logprobs is not finite: {new_logprobs}"

    is_weight = None  # Optional IS correction applied to both objectives

    # Clean up tensors we don't need to keep
    if not config.enable_reasoning_distillation:
        del logits_short, logprobs_short
    del probs_short
    # torch.cuda.empty_cache()

    # Long-prompt forward pass if any long-prompt features are enabled
    new_logprobs_long = None
    logits_long = None
    long_prompt_ref_logprobs = None

    if (config.enable_long_prompt_is or config.enable_long_prompt_rl or config.enable_reasoning_distillation):
        if batch.long_prompt_input_ids is not None:
            # Forward pass with long prompts
            model_inputs_long = {
                "input_ids": batch.long_prompt_input_ids,
                "attention_mask": batch.long_prompt_attention_mask,
            }
            if batch.is_packed and batch.long_prompt_position_ids is not None:
                model_inputs_long["position_ids"] = batch.long_prompt_position_ids

            # Add visual features if present
            if hasattr(batch, 'pixel_values') and batch.pixel_values is not None:
                model_inputs_long["pixel_values"] = batch.pixel_values
            if hasattr(batch, 'image_grid_thw') and batch.image_grid_thw is not None:
                model_inputs_long["image_grid_thw"] = batch.image_grid_thw

            # Memory-saving path: no-grad prefill (prompt) + grad on continuation with KV cache
            ids, attn, lbls = batch.long_prompt_input_ids, batch.long_prompt_attention_mask, batch.long_prompt_labels
            # We derive prompt length as the index of the first non-masked label.
            # Do NOT count all -100 values (tail padding may also be -100).
            assert lbls is not None, "long_prompt_labels required to derive prompt length"
            T = new_logprobs.shape[1]
            rows = []  # long-context logprobs aligned with short pass positions

            # Collect logits for KD if enabled (only continuation part, aligned with short)
            logits_rows = [] if config.enable_reasoning_distillation else None

            # Validate batch sizes match
            batch_size = batch.input_ids.size(0)
            long_batch_size = ids.size(0)
            assert batch_size == long_batch_size, (
                f"Batch size mismatch: short prompt batch has {batch_size} examples "
                f"but long prompt batch has {long_batch_size} examples. "
                f"All examples must have both short and long prompts."
            )

            # Log sizes for debugging
            logger.info(
                f"[Long Prompt Forward Pass] "
                f"batch_size={ids.size(0)}, "
                f"short_seq_len={batch.input_ids.size(1)}, "
                f"long_seq_len={ids.size(1)}, "
                f"T={T}"
            )

            for i in range(ids.size(0)):
                non_mask = torch.nonzero(lbls[i] != -100, as_tuple=False)
                assert non_mask.numel() > 0, "no non-masked labels found in long_prompt_labels"
                p = int(non_mask[0].item())
                assert 0 < p < ids.size(1), f"invalid prompt length {p}"
                # Use the short completion tokens/mask as canonical targets
                mask_short = masks_shifted[i]
                mask_indices = torch.nonzero(mask_short, as_tuple=False).view(-1)
                short_targets = batch.input_ids[i, 1:][mask_short].to(ids.device)

                # Log details for first example in batch
                if i == 0:
                    short_prompt_len = (batch.input_ids.size(1) - 1) - mask_short.sum().item()
                    short_completion_len = short_targets.shape[0]
                    long_total_len = ids.size(1)
                    long_completion_len = long_total_len - p
                    logger.info(
                        f"[Example 0 Sizes]\n"
                        f"  Short: prompt={short_prompt_len}, completion={short_completion_len}, total={batch.input_ids.size(1)}\n"
                        f"  Long:  prompt={p}, completion={long_completion_len}, total={long_total_len}\n"
                        f"  Short completion tokens (first 5): {short_targets[:5].tolist()}\n"
                        f"  Long input tokens at completion boundary (5 before, 5 after p={p}): {ids[i, p-5:p+5].tolist()}"
                    )
                # Prefill (prompt only) under no_grad with cache to avoid building a graph
                # Some backends disable cache when gradient checkpointing is on: toggle it just for prefill.
                under = model.module if hasattr(model, "module") else model
                prev_gc = bool(getattr(under, "is_gradient_checkpointing", False))
                prev_use_cache = bool(getattr(under.config, "use_cache", False))
                try:
                    if prev_gc:
                        under.gradient_checkpointing_disable()
                    under.config.use_cache = True
                    with torch.no_grad():
                        pre = model(input_ids=ids[i:i+1, :p], attention_mask=attn[i:i+1, :p], use_cache=True, return_dict=True)
                finally:
                    under.config.use_cache = prev_use_cache
                    if prev_gc:
                        under.gradient_checkpointing_enable()
                pkv = getattr(pre, "past_key_values", None)
                assert pkv is not None, "past_key_values is None after prefill"
                # Backprop only through the continuation using the KV cache from the prefill
                out = model(input_ids=ids[i:i+1, p:], attention_mask=attn[i:i+1, p:], past_key_values=pkv, use_cache=False, return_dict=True)
                del pkv
                logger.info(f"out.logits shape: {out.logits.shape}")

                # Extract logits (before softmax) for KD if needed
                continuation_logits = out.logits[:, :-1, :] / config.temperature  # [1, cont_len, vocab]
                lp = F.log_softmax(continuation_logits, dim=-1)
                cont_len = lp.shape[1]
                L = short_targets.shape[0]
                assert cont_len >= L, (
                    "Long prompt truncated the reasoning trace "
                    f"(long continuation len={cont_len}, short trace len={L})"
                )
                tgt = short_targets.view(1, L, 1)
                row_vals = torch.gather(lp[:, -L:, :], 2, tgt).squeeze(2)

                # Log gathered logprobs for first example
                if i == 0:
                    logger.info(
                        f"[Example 0 Logprobs]\n"
                        f"  Long continuation length: {cont_len}\n"
                        f"  Short target length (L): {L}\n"
                        f"  Using last {L} positions from long continuation\n"
                        f"  Gathered logprobs (first 5): {row_vals[0, :5].tolist()}"
                    )
                assert mask_indices.shape[0] == L, "short mask and targets length mismatch"
                row_full = torch.zeros((1, T), device=lp.device, dtype=lp.dtype)
                row_full[0, mask_indices] = row_vals
                rows.append(row_full)

                # Collect logits for KD if enabled (aligned with masks)
                if logits_rows is not None:
                    # Extract the same last L positions from continuation logits
                    logits_tail = continuation_logits[:, -L:, :]  # [1, L, vocab]
                    # Align with short sequence positions using mask_indices
                    logits_full = torch.zeros((1, T, continuation_logits.size(-1)),
                                             device=continuation_logits.device,
                                             dtype=continuation_logits.dtype)
                    logits_full[0, mask_indices, :] = logits_tail[0]
                    logits_rows.append(logits_full)
            new_logprobs_long = torch.cat(rows, dim=0)
            assert new_logprobs_long.shape == new_logprobs.shape, "long vs short logprobs length mismatch after align"

            # Assemble logits for KD if collected
            if logits_rows is not None:
                logits_long = torch.cat(logits_rows, dim=0)  # [B, T, vocab]
                assert logits_long.shape[:2] == logits_short.shape[:2], (
                    f"Long logits shape {logits_long.shape[:2]} doesn't match short logits {logits_short.shape[:2]}"
                )

            # Get long-prompt ref logprobs if available
            if batch.long_prompt_ref_logprobs is not None:
                ref_rows = []
                for i in range(ids.size(0)):
                    mask_short = masks_shifted[i]
                    mask_indices = torch.nonzero(mask_short, as_tuple=False).view(-1)
                    L = mask_indices.shape[0]
                    # Convert to same device and dtype as new_logprobs
                    vec = batch.long_prompt_ref_logprobs[i].to(device=new_logprobs.device, dtype=new_logprobs.dtype).view(-1)
                    if vec.shape[0] >= L:
                        tail = vec[-L:]
                    else:
                        pad = torch.zeros(L - vec.shape[0], device=new_logprobs.device, dtype=new_logprobs.dtype)
                        tail = torch.cat([pad, vec])
                    row_full = torch.zeros((1, T), device=new_logprobs.device, dtype=new_logprobs.dtype)
                    row_full[0, mask_indices] = tail
                    ref_rows.append(row_full)
                long_prompt_ref_logprobs = torch.cat(ref_rows, dim=0)

            # Clean up if not needed for KD (we computed only continuation logits)
        else:
            logger.warning(
                "Long-prompt features enabled but batch.long_prompt_input_ids is None. "
                "Skipping long-prompt forward pass."
            )

    # get shifted values and compute ratios
    rewards = batch.rewards[:, 1:]
    ref_logprobs = batch.ref_logprobs[:, 1:]
    old_logprobs = batch.old_logprobs[:, 1:]
    group_tokens = batch.group_tokens[:, 1:]
    num_labels_in_seq = batch.num_labels[:, 1:] # sequence dependent normalization
    overflow = batch.overflow[:, 1:]

    if config.group_normalization:
        # Check for zero or invalid group_tokens
        invalid_mask = (group_tokens <= 0) | ~torch.isfinite(group_tokens)
        if invalid_mask.any():
            logger.error(f"Found {invalid_mask.sum().item()} invalid group_tokens values")
            logger.error(f"group_tokens unique values: {torch.unique(group_tokens)}")
            raise ValueError(
                f"group_tokens must be greater than zero for group normalization. "
                f"Found {invalid_mask.sum().item()} invalid values."
            )
        tokens_weights = torch.ones_like(group_tokens) / group_tokens
    else:
        tokens_weights = torch.ones_like(group_tokens) / config.batch_size

    if config.overlong_filtering:
        # filter out sequences that do not have eos_token_id
        overflow = torch.tensor(overflow, device=overflow.device)
        tokens_weights = tokens_weights * (1 - overflow)

    assert new_logprobs.shape == ref_logprobs.shape

    log_ratio_new_old = new_logprobs - old_logprobs
    ratio_new_old = torch.exp(log_ratio_new_old)
    log_ratio_ref_new = ref_logprobs - new_logprobs
    assert torch.isfinite(log_ratio_ref_new).all(), f"log_ratio_ref_new is not finite: {log_ratio_ref_new}"

    if has_value_head:
        # Get value predictions if available
        value_predictions = outputs.value[:, :-1] # no target for the last token 
        # Compute value-based advantages: A(s,a) = MC_return - V(s)
        # where MC_return is the Monte Carlo return (rewards) and V(s) is the value prediction
        #FIXME: if this works better it should be a config
        #advantages = rewards - torch.clamp(value_predictions, 0, 1)
        advantages = rewards - value_predictions
    else:
        advantages = batch.advantages[:, 1:]

    log_p_weights = advantages.detach() if config.use_advantages else rewards
    if config.relu_log_p_weights:
        log_p_weights = torch.clamp(log_p_weights, min=0)

    clamp_log_ratio_ref_new_indicators = torch.abs(log_ratio_ref_new) > config.clamp_log_ratio_ref_new_value

    log_ratio_ref_new_clamp = torch.clamp(
        log_ratio_ref_new,
        min=-config.clamp_log_ratio_ref_new_value,
        max=config.clamp_log_ratio_ref_new_value,
    )

    approx_kl = torch.exp(log_ratio_ref_new_clamp) - log_ratio_ref_new_clamp - 1  # Schulman KL approx

    assert torch.isfinite(approx_kl).all(), f"approx_kl is not finite: {approx_kl}"
    entropy_bonus_coef = linear_decay_coef(current_step, max_step, config.entropy_bonus, config.final_entropy_bonus)
    kl_coef = linear_decay_coef(current_step, max_step, config.kl_coef, config.final_kl_coef)

    # compute algorithm-specific losses
    match config.policy_loss:
        case "ppo":
            surr1 = ratio_new_old * log_p_weights
            # DAPO: Asymmetric clipping if clip_ratio_high is specified
            clip_low = config.epsilon
            clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.epsilon
            clamped_ratio = torch.clamp(ratio_new_old, 1 - clip_low, 1 + clip_high)
            clamp_log_ratio_new_old_indicators = clamped_ratio != ratio_new_old
            surr2 = clamped_ratio * log_p_weights
            policy_loss = torch.min(surr1, surr2)
        case "reinforce":
            surr1 = torch.zeros_like(ratio_new_old)
            surr2 = torch.zeros_like(ratio_new_old)
            clamp_log_ratio_new_old_indicators = ratio_new_old > 1 + config.epsilon
            ratio_new_old = torch.clamp(ratio_new_old, 0, 1 + config.epsilon)
            policy_loss = new_logprobs * log_p_weights * ratio_new_old.detach()
        case _:
            raise ValueError(f"Unknown algorithm {config.policy_loss}")

    # combine loss components
    loss = policy_loss - kl_coef * approx_kl + entropy_bonus_coef * entropy  # 1 x (BxL) x 1
    assert loss.shape == tokens_weights.shape, (
        f"Loss shape {loss.shape} does not match example weights shape {tokens_weights.shape}"
    )

    # Apply importance sampling correction if enabled
    if config.enable_long_prompt_is and new_logprobs_long is not None:
        # IS weight: π_new(a|long) / π_new(a|short)
        log_ratio_long_short = new_logprobs_long - new_logprobs
        is_weight = torch.exp(log_ratio_long_short)
        loss = loss * is_weight

    loss = loss * tokens_weights  # 1 x (BxL) x 1

    policy_loss_total = -sum_sum(loss, masks_shifted, segments)
    # Save short-prompt loss for logging
    policy_loss_short_total = policy_loss_total

    # Add long-prompt RL objective if enabled
    policy_loss_long_total = None
    if config.enable_long_prompt_rl and new_logprobs_long is not None:
        # Compute policy loss for long prompts (using same algorithm)
        log_ratio_new_old_long = new_logprobs_long - old_logprobs
        ratio_new_old_long = torch.exp(log_ratio_new_old_long)

        # KL penalty for long prompts
        if long_prompt_ref_logprobs is not None:
            log_ratio_ref_new_long = long_prompt_ref_logprobs - new_logprobs_long
            log_ratio_ref_new_long_clamp = torch.clamp(
                log_ratio_ref_new_long,
                min=-config.clamp_log_ratio_ref_new_value,
                max=config.clamp_log_ratio_ref_new_value,
            )
            approx_kl_long = torch.exp(log_ratio_ref_new_long_clamp) - log_ratio_ref_new_long_clamp - 1
        else:
            approx_kl_long = torch.zeros_like(new_logprobs_long)

        # Policy loss for long prompts
        match config.policy_loss:
            case "ppo":
                surr1_long = ratio_new_old_long * log_p_weights
                clamped_ratio_long = torch.clamp(ratio_new_old_long, 1 - clip_low, 1 + clip_high)
                surr2_long = clamped_ratio_long * log_p_weights
                policy_loss_long = torch.min(surr1_long, surr2_long)
            case "reinforce":
                ratio_new_old_long_clamped = torch.clamp(ratio_new_old_long, 0, 1 + config.epsilon)
                policy_loss_long = new_logprobs_long * log_p_weights * ratio_new_old_long_clamped.detach()
            case _:
                raise ValueError(f"Unknown algorithm {config.policy_loss}")

        # Combine loss components for long prompts
        loss_long = policy_loss_long - kl_coef * approx_kl_long + entropy_bonus_coef * entropy
        if config.enable_long_prompt_is and is_weight is not None:
            loss_long = loss_long * is_weight
        loss_long = loss_long * tokens_weights
        policy_loss_long_total = -sum_sum(loss_long, masks_shifted, segments)

        # Mix: (1-w)*loss_short + w*loss_long
        w = config.long_prompt_rl_weight
        policy_loss_total = (1 - w) * policy_loss_total + w * policy_loss_long_total
    
    if has_value_head:
        # Get the value predictions
        values = outputs.value
        # Use the already extracted and shifted rewards as value labels
        value_labels = rewards  # This is already shifted (from line 216)
        values = values[:, :-1]
        values_labels = value_labels
        assert values.shape == tokens_weights.shape, (
            f"Values shape {values.shape} does not match example weights shape {tokens_weights.shape}"
        )
        value_loss = 0.5 * torch.square(values - values_labels) * tokens_weights
        value_loss = sum_sum(value_loss, masks_shifted, segments) 
        
        # Combine policy loss and value loss
        final_loss = policy_loss_total + config.value_loss_coef * value_loss
    else:
        final_loss = policy_loss_total

    # Add knowledge distillation loss if enabled
    kd_loss_value = None
    if config.enable_reasoning_distillation and new_logprobs_long is not None:
        # Use Schulman approximation by default (fast), allow top-k or full vocab overrides
        top_k_cfg = config.distillation_top_k
        if top_k_cfg is None:
            use_schulman = True
            top_k_value = None
        elif top_k_cfg == 0:
            use_schulman = False
            top_k_value = None  # full vocabulary KL
        else:
            use_schulman = False
            top_k_value = top_k_cfg
        kd_loss = compute_kd_loss(
            logits_short=logits_short,
            logits_long=logits_long,
            logprobs_short=new_logprobs,
            logprobs_long=new_logprobs_long,
            masks=masks_shifted,
            use_schulman=use_schulman,
            top_k=top_k_value,
            segments=segments,
        )
        kd_loss_value = kd_loss.item()
        final_loss = final_loss + config.distillation_coef * kd_loss

    # ensure loss is valid
    assert torch.isfinite(final_loss), f"Non-finite loss detected: {final_loss}"

    if int(masks_shifted.sum().item()) == 0:
        stats_no_labels = {
            "input_size": float(batch.input_ids.numel()),
        }
        return final_loss, stats_no_labels

    # Metric aggregation behavior:
    # 1. loss: pre-multiplied by token_weights, reported as sum
    # 2. min/max values: computed across entire batch
    # 3. other statistics: averaged per sequence, then averaged across batch
    stats = {
        "loss": final_loss.item(),
        "max_loss": final_loss.item(),
        "min_loss": final_loss.item(),
        "reward": sum_sum(rewards / num_labels_in_seq, masks_shifted, segments).item(),
        "max_reward": rewards[masks_shifted].max().item(),
        "min_reward": rewards[masks_shifted].min().item(),
        "entropy": sum_sum(entropy / num_labels_in_seq, masks_shifted, segments).item(),
        "old_logprobs": sum_sum(old_logprobs / num_labels_in_seq, masks_shifted, segments).item(),
        "new_logprobs": sum_sum(new_logprobs / num_labels_in_seq, masks_shifted, segments).item(),
        "ref_logprobs": sum_sum(ref_logprobs / num_labels_in_seq, masks_shifted, segments).item(),
        "advantage": sum_sum(advantages / num_labels_in_seq, masks_shifted, segments).item(),
        "max_advantage": advantages[masks_shifted].max().item(),
        "min_advantage": advantages[masks_shifted].min().item(),
        "kl": sum_sum(approx_kl / num_labels_in_seq, masks_shifted, segments).item(),
        "max_kl": approx_kl[masks_shifted].max().item(),
        "min_kl": approx_kl[masks_shifted].min().item(),
        "policy_loss": sum_sum(policy_loss / num_labels_in_seq, masks_shifted, segments).item(),
        "surr1": sum_sum(surr1 / num_labels_in_seq, masks_shifted, segments).item(),
        "surr2": sum_sum(surr2 / num_labels_in_seq, masks_shifted, segments).item(),
        "ratio_new_old": sum_sum(ratio_new_old / num_labels_in_seq, masks_shifted, segments).item(),
        "ratio_new_old_sum": sum_sum(ratio_new_old, masks_shifted, segments).item(),
        "ratio_new_old_squared_sum": sum_sum(  # useful to estimate the ESS
            ratio_new_old * ratio_new_old, masks_shifted, segments
        ).item(),
        "ratio_ref_new": sum_sum(torch.exp(log_ratio_ref_new) / num_labels_in_seq, masks_shifted, segments).item(),
        "ratio_ref_old": sum_sum(torch.exp(ref_logprobs - old_logprobs) / num_labels_in_seq, masks_shifted, segments).item(),
        "clamp_log_ratio_ref_new_indicator": sum_sum(
            clamp_log_ratio_ref_new_indicators / num_labels_in_seq, masks_shifted, segments
        ).item(),
        "clamp_log_ratio_new_old_indicator": sum_sum(
            clamp_log_ratio_new_old_indicators / num_labels_in_seq, masks_shifted, segments
        ).item(),
        "num_nans": torch.isnan(loss).sum().item(),
        "token_weight": sum_sum(tokens_weights / num_labels_in_seq, masks_shifted, segments).item(),
        "max_token_weight": tokens_weights[masks_shifted].max().item(),
        "min_token_weight": tokens_weights[masks_shifted].min().item(),
        "kl_coef": num_sequences * kl_coef,
        "entropy_bonus_coef": num_sequences * entropy_bonus_coef,
        "num_output_tokens_sum": masks_shifted.sum().item(),
        "input_size": batch.input_ids.numel(), 
    }

    if has_value_head:
        stats["value_mean"] = sum_sum(value_predictions / num_labels_in_seq, masks_shifted, segments).item()
        stats["value_max"] = value_predictions[masks_shifted].max().item() if masks_shifted.any() else 0.0
        stats["value_min"] = value_predictions[masks_shifted].min().item() if masks_shifted.any() else 0.0
        stats["value_loss"] = value_loss.item()
        stats["value_mse"] = sum_sum(
            torch.square(value_predictions - value_labels) / num_labels_in_seq, masks_shifted, segments
        ).item()

    # Add multi-prompt training loss components
    if policy_loss_short_total is not None:
        stats["policy_loss_short"] = policy_loss_short_total.item()
    if policy_loss_long_total is not None:
        stats["policy_loss_long"] = policy_loss_long_total.item()
    if kd_loss_value is not None:
        stats["kd_loss"] = kd_loss_value

    return final_loss, stats


def compute_kd_loss(
    logits_short: torch.Tensor,
    logits_long: torch.Tensor,
    logprobs_short: torch.Tensor,
    logprobs_long: torch.Tensor,
    masks: torch.Tensor,
    use_schulman: bool = True,
    top_k: int | None = None,
    segments: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """
    Compute knowledge distillation loss KL(π_short || π_long) to keep reasoning consistent
    across different prompt formats.

    Supports three modes:
    1. Schulman approximation (default): Fast approximation using selected token log-probs
    2. Top-k KL: Full KL but only over top-k tokens to reduce noise
    3. Full vocab KL: Full KL divergence over entire vocabulary

    Args:
        logits_short: Logits from short context [B, L, vocab_size]
        logits_long: Logits from long context [B, L, vocab_size]
        logprobs_short: Log-probs of selected tokens from short context [B, L]
        logprobs_long: Log-probs of selected tokens from long context [B, L]
        masks: Token masks [B, L]
        use_schulman: If True, use Schulman approximation (fast, default).
                      If False, compute full KL over logits.
        top_k: If set and use_schulman=False, only compute KL over top-k tokens.
               Pass None (with use_schulman=False) to compute full-vocabulary KL.
               Ignored if use_schulman=True.
        segments: Optional sequence boundaries for packed sequences

    Returns:
        Scalar KL divergence loss
    """
    if use_schulman:
        # Schulman approximation: KL ≈ exp(log_ratio) - log_ratio - 1
        # where log_ratio = log(π_short) - log(π_long)
        log_ratio = logprobs_short - logprobs_long
        kl = torch.exp(log_ratio) - log_ratio - 1  # [B, L]
    elif top_k is not None:
        # Top-k KL: Renormalize within top-k tokens from short context
        topk_values_short, topk_indices = torch.topk(logits_short, top_k, dim=-1)  # [B, L, k]
        logits_long_topk = torch.gather(logits_long, dim=-1, index=topk_indices)  # [B, L, k]

        log_p_topk = F.log_softmax(topk_values_short, dim=-1)
        log_q_topk = F.log_softmax(logits_long_topk, dim=-1)

        kl = F.kl_div(log_q_topk, log_p_topk, log_target=True, reduction='none').sum(dim=-1)  # [B, L]
    else:
        # Full vocabulary KL: KL(P || Q) = sum(P * (log_P - log_Q))
        log_p = F.log_softmax(logits_short, dim=-1)
        log_q = F.log_softmax(logits_long, dim=-1)
        # F.kl_div expects log_q as input, log_p as target
        kl = F.kl_div(log_q, log_p, log_target=True, reduction='none').sum(dim=-1)  # [B, L]

    # Apply masks and reduce
    if segments is not None:
        # Packed sequences: use segment-aware reduction
        return sum_sum(kl, masks, segments)
    else:
        # Standard sequences
        kl_masked = kl * masks
        return kl_masked.sum() / masks.sum()


def populate_rl_data(dataset: list[dict[str, Any]], eos_token_id: int, config: RLConfig) -> list[dict[str, Any]]:
    """
    Populates a dataset with reinforcement learning specific data columns including
    rewards, advantages, and token weights.

    Args:
        dataset (Dataset): The input dataset to populate with RL data
        eos_token_id (int): End of sequence token ID
        config (RLConfig): Configuration object containing RL training parameters

    Returns:
        Dataset: The dataset populated with RL-specific columns
    """
    # Convert to pandas for processing
    df_init = pd.DataFrame(dataset)
    assert isinstance(df_init, pd.DataFrame)

    # Step 1: calculate group-level statistics
    df_stats = df_init[["group_id", "rollout_index", "step_index"]].copy()
    df_stats["num_tokens"] = df_init["input_ids"].apply(lambda x: len(x))
    # We assume that rewards for all tokens are the same
    df_stats["rollout_reward"] = df_init["rewards"].apply(lambda x: x[0])
    # Check that the reward is the same for each step in the rollout
    assert df_stats.groupby(["group_id", "rollout_index"])["rollout_reward"].nunique().max() == 1
    # Only keep step_index == 0
    df_stats = df_stats[df_stats["step_index"] == 0].drop(columns=["step_index"])
    df_grouped = (
        df_stats.groupby("group_id")
        .agg(
            rollout_reward_mean=("rollout_reward", "mean"),
            rollout_reward_std=("rollout_reward", "std"),
            group_tokens=("num_tokens", "mean"), 
        )
        .reset_index()
    )
    assert df_grouped.columns.tolist() == ["group_id", "rollout_reward_mean", "rollout_reward_std", "group_tokens"]

    # Step 2: calculate advantages for each sample
    df_advantages = pd.merge(
        df_init[["group_id", "rollout_index", "step_index", "rewards"]],
        df_grouped,
        on="group_id",
        how="left"
    )
    assert len(df_advantages) == len(df_init)
    def calculate_advantages(row):
        rewards = row["rewards"]
        mean = row["rollout_reward_mean"]
        std = row["rollout_reward_std"]
        if config.divide_advantage_by_std:
            advantages = [(reward - mean) / (np.nan_to_num(std) + 1e-4) for reward in rewards]
        else:
            advantages = [(reward - mean) for reward in rewards]
        return advantages
    df_advantages["advantages"] = df_advantages.apply(
        calculate_advantages,
        axis=1,
    )
    df_advantages = df_advantages.drop(columns=["rewards", "rollout_reward_mean", "rollout_reward_std"])
    assert df_advantages.columns.tolist() == ["group_id", "rollout_index", "step_index", "group_tokens", "advantages"]

    # Step 3: bring advantages and group level stats back to the main df
    df = df_init.drop(columns=["advantages", "group_tokens"])
    df = pd.merge(df, df_advantages, on=["group_id", "rollout_index", "step_index"], how="left")
    # Debug print lengths of all dataframes
    assert len(df) == len(df_init)

    # Step 4: make token-level overflow and mean group length information
    df["overflow"] = df.apply(
        lambda row: [0.0] * len(row["overflow"]) if eos_token_id in row["input_ids"] else [1.0] * len(row["overflow"]),
        axis=1,
    )
    df["group_tokens"] = df.apply(lambda row: [row["group_tokens"]] * len(row["input_ids"]), axis=1)
    df["num_labels"] = df.apply(lambda row: [sum(1 for label in row["labels"] if label != -100)] * len(row["input_ids"]), axis=1)

    # Step 5: move the results back to the dataset
    advantages_list = df["advantages"].tolist()
    group_tokens_list = df["group_tokens"].tolist()
    overflow_list = df["overflow"].tolist()
    num_labels_list = df["num_labels"].tolist()
    for i, entry in enumerate(dataset):
        entry["advantages"] = advantages_list[i]
        entry["group_tokens"] = group_tokens_list[i]
        entry["overflow"] = overflow_list[i]
        entry["num_labels"] = num_labels_list[i]
    return dataset


def prepare_rl_fields(
    encoding: dict[str, Any],
    reward: float,
    old_logprobs: list[float],
    ref_logprobs: list[float],
) -> dict[str, Any]:
    """
    Convert reward per agent step to reward per token and add returns and advantages placeholders
    """
    target_tokens = [token for token in encoding["labels"] if token != -100]
    assert len(target_tokens) == len(old_logprobs), (
        f"Target tokens: {len(target_tokens)}, old logprobs: {len(old_logprobs)}"
    )

    encoding["rewards"] = [reward] * len(encoding["labels"])
    encoding["advantages"] = [0.0] * len(encoding["labels"])  # place holder
    encoding["old_logprobs"] = [0] * (len(encoding["labels"]) - len(old_logprobs)) + old_logprobs
    encoding["ref_logprobs"] = [0] * (len(encoding["labels"]) - len(ref_logprobs)) + ref_logprobs
    encoding["overflow"] = [0] * len(encoding["labels"])  # place holder
    encoding["group_tokens"] = [0] * len(encoding["labels"])  # place holder
    encoding["num_labels"] = [1 if label != -100 else 0 for label in encoding["labels"]]  # count only output tokens
    return encoding
