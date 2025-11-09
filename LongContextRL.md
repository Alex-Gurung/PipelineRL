# Long-Context RL Training Design Document

## Overview

This document describes the multi-prompt RL training system that enables training models on both short and long prompts simultaneously. The system addresses the computational challenge of long-context RL: long prompts (100k+ tokens) are too expensive for vLLM rollout generation, but we still want the model to perform well on them.

**Core Idea**: Sample rollouts using cheap short prompts, but train on both short and long prompts using importance sampling corrections and consistency losses.

## Problem Statement

### The Challenge
- **Rollout Generation**: We use vLLM to generate reasoning traces, but long prompts (full articles, 100k+ tokens) are too expensive to run in vLLM at scale
- **Deployment Reality**: At inference time, users may provide either short prompts (metadata only) or long prompts (full articles)
- **Training Goal**: We want the model to perform well on both prompt formats without the cost of generating rollouts with long prompts

### The Solution
1. **Generate rollouts** using short prompts only (cheap, fits in vLLM)
2. **During training**, compute losses for both short and long prompts:
   - Short prompt loss: standard PPO/GRPO loss
   - Long prompt loss: importance-sampling corrected loss
   - Consistency loss: keeps distributions similar across formats

## Architecture

### Data Flow

```
Training Data (from dataset):
  - messages_short: Short prompt (metadata only)
  - messages_long: Long prompt (full article)
  - completion: Reasoning trace (same for both)

Rollout Generation (vLLM):
  - Input: messages_short
  - Output: completion tokens + old_logprobs

Preprocessing:
  - Tokenize short prompt + completion → input_ids, labels
  - Tokenize long prompt + completion → long_prompt_input_ids, long_prompt_labels
  - Compute ref logprobs for both if reference model exists

Training Forward Pass:
  - Short forward: model(input_ids) → logits_short, new_logprobs
  - Long forward: model(long_prompt_input_ids) → logits_long, new_logprobs_long
    (uses KV cache: no-grad prefill + grad continuation)

Training Loss:
  - L_short: PPO loss on short prompts
  - L_long: PPO loss on long prompts (optional)
  - L_consistency: KL(π_new(·|short) || π_new(·|long)) (optional)
  - L_final = (1-w)*L_short + w*L_long + λ*L_consistency
```

## Core Components

### 1. Preprocessing

#### Long Prompt Tokenization
The system tokenizes long prompts by:
1. Extracting the completion text from the rollout trace
2. Building the full conversation (long prompt + assistant response)
3. Applying the chat template to ensure consistent special tokens
4. Tokenizing with `long_prompt_seq_length` (e.g., 124000 tokens)
5. Creating labels that mask the prompt and expose only the completion

**Critical Design Decision - Chat Template Consistency**:
The chat template must be applied to the FULL conversation (including the assistant's response), not just concatenating strings. This ensures the completion tokens include the same special tokens that were present during rollout generation (e.g., `<|im_start|>assistant\n` and `<|im_end|>`). Without this, the completion would be missing 2+ special tokens, causing a mismatch between short and long prompt completions.

#### Reference Logprobs for Long Prompts
When computing reference model logprobs:
- **Completion length** is determined by counting non-masked labels (`sum(label != -100)`)
- **Why not backward indexing?** If truncation occurred during tokenization, indexing backward from the end would capture the wrong tokens. Label-based counting is robust to truncation.
- The reference vLLM is called with: `get_batch_logprobs_token_ids(prompt_tokens, completion_tokens)`

### 2. Training Forward Pass

#### Short Prompt Pass
Standard forward pass through the model with the short prompt + completion. Computes:
- Logits and log-probabilities for all tokens
- Entropy of the policy distribution
- Log-probabilities for the actual generated tokens

#### Long Prompt Pass - Memory Optimization
**Challenge**: Long prompts (100k tokens) cannot fit in memory with a full backward pass.

**Solution - KV Cache Split**:
1. **Prefill Stage** (no gradients):
   - Process the prompt portion only
   - Cache the key/value states from attention layers
   - Run under `torch.no_grad()` to avoid storing activations

2. **Continuation Stage** (with gradients):
   - Process only the completion tokens (~1k tokens)
   - Use the cached KV states from prefill
   - Full gradients computed only for this stage

**Memory Savings**: Instead of computing gradients for 100k+ tokens, we only compute them for ~1k tokens (the completion). This is a ~100x reduction in activation memory.

**Gradient Checkpointing Interaction**: Some model backends disable KV caching when gradient checkpointing is enabled. The implementation temporarily disables gradient checkpointing during prefill, then re-enables it for the continuation.

#### Alignment of Short and Long Completions
The short and long prompts produce the same semantic completion, but may tokenize differently. The system uses the **short completion as canonical**:
- Extract the last L tokens from the long continuation (where L = length of short completion)
- Align these with the masked positions in the short sequence
- This produces `new_logprobs_long` with the same shape as `new_logprobs`

**Why alignment is necessary**: Chat templates, tokenization context, and special tokens can cause the same text to tokenize differently. By using short completion token IDs as ground truth and extracting corresponding logprobs from the long context, we maintain consistency.

### 3. Loss Computation

#### Base PPO/GRPO Loss (Short Prompts)
Standard proximal policy optimization:
- Ratio: `ratio_new_old = exp(new_logprobs - old_logprobs)`
- Clipped surrogate: `loss = min(ratio * advantages, clamp(ratio, 1-ε, 1+ε) * advantages)`
- KL penalty: `approx_kl = exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1` (Schulman approximation)
- Entropy bonus: `-H[π_new]`
- Final: `L_short = loss - β*KL + α*H`
- Weighted by: `tokens_weights` (from GRPO group normalization)

#### Importance Sampling Correction

**Feature**: `enable_long_prompt_is`

**Mathematical Justification**:
Actions were sampled from the distribution `π_old(a|short)` during rollout generation, but we want to optimize the policy for `π_new(a|long)`. The importance sampling weight corrects for this distribution mismatch:

```
IS_weight = π_new(a|long) / π_new(a|short)
```

**Why π_new and not π_ref?**
- The PPO ratio `π_new / π_old` already handles the on-policy → off-policy correction
- This IS correction addresses the **prompt format** mismatch, not the policy staleness
- We want to ask: "According to the current policy, how much more/less likely is this action under long vs short prompts?"
- Using π_new gives us the current policy's assessment of the prompt format effect

**When to use**: Single RL objective optimized for long prompts, when rollouts must come from short prompts for computational reasons.

#### Dual Objective - Long Prompt RL

**Feature**: `enable_long_prompt_rl`

**Mathematical Formulation**:
```
L_final = (1-w)*L_short + w*L_long
```

Where:
- `L_short`: PPO loss computed with short-prompt logprobs
- `L_long`: PPO loss computed with long-prompt logprobs (same advantages, same old_logprobs from rollout)
- `w`: `long_prompt_rl_weight` (e.g., 0.5 for equal weighting)

**Justification**: This creates a direct optimization pressure for long-prompt performance. Instead of only correcting short-prompt loss with IS weights, we explicitly compute gradients for long-prompt behavior.

**Long-Prompt KL Penalty**: If `long_prompt_ref_logprobs` are available, the long-prompt loss includes its own KL penalty:
```
approx_kl_long = exp(long_prompt_ref_logprobs - new_logprobs_long) - ... - 1
```

**Entropy Term**: Currently uses entropy from the short-prompt distribution for both losses. This is a design choice - entropy is a measure of policy uncertainty and using a single entropy value is simpler and more stable.

**When to use**: Explicitly optimize for both prompt formats with controllable weighting. Useful when both formats are important at inference time.

#### Reasoning Consistency Loss

**Feature**: `enable_reasoning_distillation`

**Mathematical Formulation**:
```
L_consistency = KL(π_new(·|short) || π_new(·|long))
```

**Important Clarification**: This is NOT knowledge distillation from old to new policy. It's a **consistency loss** that ensures the current policy produces similar token distributions regardless of prompt format.

**Why this matters**:
- Without this loss, the model might learn very different reasoning styles for short vs long prompts
- Long prompts have more context and might cause the model to take shortcuts or use different reasoning patterns
- This loss enforces that the *way* the model thinks should be similar, even if the context differs

**Three Computation Modes**:

1. **Schulman Approximation** (default, fastest):
   - Uses only the log-probabilities of selected tokens
   - `KL ≈ exp(log_ratio) - log_ratio - 1` where `log_ratio = log(π_short/π_long)`
   - No need for full vocabulary logits

2. **Top-k KL** (when `distillation_top_k` is set):
   - Computes full KL but only over top-k tokens from short context
   - Reduces noise from low-probability tokens
   - Requires full logits for both short and long

3. **Full Vocabulary KL**:
   - Full KL divergence over entire vocabulary
   - Most accurate but most expensive
   - Requires full logits for both short and long

**When to use**: Prevent the model from developing divergent reasoning styles across prompt formats. Particularly important when long prompts might enable shortcuts or different reasoning paths.

## Configuration

### Core Config Options

```yaml
# Multi-prompt RL features
enable_long_prompt_is: false          # IS correction for single objective
enable_long_prompt_rl: true           # Dual objective (short + long)
long_prompt_rl_weight: 0.5            # Weight for long objective (0=short only, 1=long only)
long_prompt_seq_length: 124000        # Max tokens for long prompts

# Reasoning consistency
enable_reasoning_distillation: false  # Consistency loss
distillation_coef: 0.1                # Weight for consistency loss
distillation_top_k: null              # null=Schulman, int=top-k, 0=full vocab
```

### Feature Combinations

| Configuration | Use Case | Mathematical Form |
|---------------|----------|-------------------|
| `enable_long_prompt_rl=True` only | Equal weight on both formats | `L = 0.5*L_short + 0.5*L_long` |
| `enable_long_prompt_is=True` only | Single objective, long-optimized | `L = L_short * π_new(long)/π_new(short)` |
| Both IS + RL | Dual objective, both corrected | `L = (1-w)*[L_short * IS] + w*[L_long * IS]` |
| Add `enable_reasoning_distillation` | Enforce consistency | `L = ... + λ*KL(π_short || π_long)` |

### Example Scenarios

**Scenario 1: Train for both formats equally with consistency**
```yaml
enable_long_prompt_rl: true
long_prompt_rl_weight: 0.5
enable_reasoning_distillation: true
distillation_coef: 0.1
```
→ Balanced training on short and long, with KL penalty preventing divergence

**Scenario 2: Optimize primarily for long, minimal short**
```yaml
enable_long_prompt_is: true
enable_long_prompt_rl: true
long_prompt_rl_weight: 0.8
```
→ 80% weight on long prompts, 20% on short, both with IS correction

**Scenario 3: Long-only optimization (if short performance doesn't matter)**
```yaml
enable_long_prompt_is: true
enable_long_prompt_rl: true
long_prompt_rl_weight: 1.0
```
→ 100% optimization for long prompts, IS-corrected

## Dataset Requirements

### Required Fields

Each training example must have:
- `messages_short`: Chat-formatted short prompt
- `messages_long`: Chat-formatted long prompt (full context)
- `text`: Generated completion from vLLM rollout
- `n_predicted`: Character length of the completion
- `logprobs`: Old policy log-probabilities (from rollout)
- `rewards`: Reward values for this rollout
- `group_id`: Group identifier for GRPO normalization
- Standard RL metadata: `rollout_index`, `step_index`, `model_version`

### Preprocessing Output

After preprocessing, each example contains:
- **Short prompt fields**: `input_ids`, `labels`, `attention_mask`, `logprobs`, `ref_logprobs`
- **Long prompt fields**: `long_prompt_input_ids`, `long_prompt_labels`, `long_prompt_attention_mask`, `long_prompt_ref_logprobs`
- **RL metadata**: `advantages`, `group_tokens`, `overflow`, `num_labels`

The labels mask the prompt tokens (-100) and expose only the completion tokens.

## Implementation Details

### Chat Template Consistency Issue

**Symptom**: Completion token counts differ by exactly 2 between short and long prompts.

**Root Cause**: The rollout generation uses `apply_chat_template()` which adds role-specific special tokens (e.g., `<|im_start|>assistant\n`, `<|im_end|>`). If preprocessing concatenates strings instead of using the chat template on the full conversation, these special tokens are missing.

**Fix**: Apply chat template to the complete conversation including the assistant's response:
```
messages_long + [{"role": "assistant", "content": completion_text}]
```

This matches exactly what vLLM does during rollout generation, ensuring identical tokenization.

### Batch Collation and Padding

**Issue**: The field `group_tokens` is a per-sequence constant (all values identical, e.g., `[781.0, 781.0, 781.0, ...]` representing the mean tokens per group). When sequences in a batch have different lengths, padding was adding 0.0 values, which then caused division-by-zero errors in GRPO normalization.

**Fix**: Pad `group_tokens` and `num_labels` with their own sequence values rather than 0.0. Each sequence uses its first element as the pad value.

**Generalization**: This applies to any per-sequence constant field - pad with the constant value, not with 0.

### Sequence Packing Incompatibility

**Issue**: Multi-prompt training requires aligned pairs of (short prompt, long prompt) for each example. Sequence packing concatenates multiple examples into a single sequence, destroying this alignment.

**Resolution**: Disable both `seq_packing` and `seq_parallel` for multi-prompt training configurations. These features are incompatible by design.

### Reference Model Requirements

The reference vLLM is only needed when:
- `kl_coef > 0` (KL penalty requires reference policy)
- `enable_long_prompt_rl=True` AND `kl_coef > 0` (long-prompt KL penalty)

The reference vLLM is NOT needed when:
- `kl_coef = 0` (no KL penalty; can use old_logprobs as reference)
- `preprocessor_fraction = 0` (no separate reference sampling phase)

In typical multi-prompt configs with GRPO (`kl_coef=0`), the reference model is not loaded, saving significant memory.

## Memory Optimization

### KV Cache Two-Stage Strategy

**Without KV Cache**:
- Forward pass: 100k tokens × hidden_dim × num_layers
- Backward pass: Store activations for all 100k tokens
- Peak memory: ~100k × hidden_dim × num_layers × 2

**With KV Cache Split**:
- Prefill (no grad): 100k tokens, cache KV states only (~10% of activation memory)
- Continuation (with grad): 1k tokens, full backward pass
- Peak memory: ~1k × hidden_dim × num_layers × 2 + KV cache

**Savings**: Approximately 100x reduction in activation memory for gradients.

**Trade-off**: Two forward passes instead of one (prefill + continuation), but prefill has no gradients so it's much faster than a full backward pass.

## Debugging and Validation

### Size Consistency Checks

The implementation logs detailed size information at each stage:

**Preprocessing**: Logs first trace statistics showing prompt/completion/total lengths for both short and long

**Training Forward Pass**: Logs batch dimensions, sequence lengths, and validates that short and long batches have matching sizes

**Completion Alignment**: Logs the token IDs at the prompt/completion boundary for both formats to verify correct alignment

**Ref Logprobs**: Logs completion length computation and first few completion tokens to verify correct extraction

### Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `group_tokens must be greater than zero` | Padding with 0.0 | Pad with sequence's own value |
| `Batch size mismatch: short=1, long=6` | Sequence packing enabled | Disable `seq_packing` and `seq_parallel` |
| `Long prompt truncated the reasoning trace` | `long_prompt_seq_length` too small | Increase limit or reduce prompt size |
| Completion length mismatch (+2 tokens) | String concatenation instead of chat template | Use `apply_chat_template` on full conversation |

## Design Rationale

### Why Not Sample Long Prompts for Rollouts?

**Cost Analysis**:
- vLLM throughput scales inversely with sequence length
- 100k token prompts → ~100x slower than 1k token prompts
- For online RL with thousands of rollouts per iteration, this is prohibitive

**Solution Justification**:
- Sampling from short prompts is cheap and fast
- IS correction and dual objectives provide training signal for long prompts
- Final model learns to handle both formats despite only sampling from one

### Why Use π_new for IS Instead of π_ref?

**Mathematical Reasoning**:
- PPO already includes the ratio `π_new(a|short) / π_old(a|short)` for policy staleness
- The prompt format correction should use current policy: "How does MY current policy differ between formats?"
- Using π_ref would double-count the policy staleness correction

**Practical Reasoning**:
- π_new is always available (just computed in forward pass)
- π_ref may not exist if `kl_coef=0` or reference model is disabled
- Simpler and more stable

### Why Reuse Short-Prompt Entropy for Long-Prompt Loss?

**Design Choice**:
- Entropy is a measure of policy uncertainty/exploration
- Using a single entropy value is simpler and more stable than computing separate entropy for long prompts
- Long-prompt forward pass uses KV cache and only computes continuation logits, making full entropy computation expensive

**Alternative**: Could compute entropy from long continuation logits, but this would:
- Add computational cost
- Only measure entropy over completion, not full sequence
- May not provide significant benefit

### Why KL Consistency Instead of Traditional KD?

**Traditional KD**: `KL(π_old || π_new)` - preserve knowledge from old model
**Our Consistency Loss**: `KL(π_new(·|short) || π_new(·|long))` - keep current model consistent across formats

**Justification**:
- We don't want to preserve old model behavior (that's what PPO does)
- We want to prevent the current model from developing format-specific reasoning styles
- This is a regularization on the current policy, not distillation from a teacher

## Future Directions

### Potential Enhancements
1. **Clipped IS weights**: Prevent extreme importance ratios from destabilizing training
2. **Adaptive weighting**: Adjust `long_prompt_rl_weight` during training based on performance
3. **Multi-length interpolation**: Train on prompts of varying lengths (not just binary short/long)
4. **Shared prefill caching**: Batch examples with identical long prompts to share KV cache
5. **Separate entropy for long**: Optionally compute and use long-prompt entropy in long loss

### Known Limitations
1. **No packing support**: Cannot use sequence packing (architectural constraint)
2. **Memory scaling**: Still limited by longest sequence in batch
3. **Single long format**: Only one long prompt per example (could support multiple variants)
4. **IS weight stability**: Unbounded IS weights can cause high variance

## References

### Key Files
- `pipelinerl/preprocess.py`: Long prompt tokenization, ref logprobs
- `pipelinerl/finetune/rl/__init__.py`: RL loss computation, KV cache logic
- `pipelinerl/finetune/data.py`: Batch collation with correct padding
- `pipelinerl/finetune/types.py`: Data types with long-prompt fields
- `conf/scitrek_multiprompt.yaml`: Example configuration

### Related Concepts
- **GRPO**: Group Relative Policy Optimization - normalizes advantages within groups
- **PPO**: Proximal Policy Optimization - clipped surrogate objective
- **Importance Sampling**: Correct for sampling from one distribution while optimizing for another
- **KV Cache**: Cache attention key/value states to avoid recomputation
- **Schulman KL**: Fast approximation `KL ≈ exp(r) - r - 1` where `r = log(p/q)`

### Recent Fixes (2025-11)
1. Updated KD docstring to clarify consistency loss vs knowledge distillation
2. Fixed `logits_long` collection for KD when using top-k or full-vocab modes
3. Fixed ref logprob indexing to use label-based completion length (robust to truncation)
4. Fixed `group_tokens` padding to use sequence value instead of 0.0
5. Added comprehensive size logging for debugging alignment issues
