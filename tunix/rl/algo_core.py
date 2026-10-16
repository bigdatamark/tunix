# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [MODULE DOC]
# Core numerical pieces of the RL algorithms, registered by name in the
# default function registry at import time (side effect): advantage
# estimators "gae", "grpo", "rloo", "drgrpo"; policy losses "ppo" and "grpo";
# value loss "ppo". Also masked statistics helpers (mean, variance,
# whitening) and token entropy from logits. Loss functions receive the
# model being trained and a TrainExample and return a scalar loss plus an
# aux dict of scalar diagnostics.
"""Algorithm core implementations for RL and Agentic RL learners."""

import functools
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import function_registry


registry = function_registry.default_registry

# ==============================================================================
# Utils
# ==============================================================================

@registry.register("advantage_estimator", "gae")
@jax.jit
def compute_gae_advantages(
    rewards: jax.Array,
    values: jax.Array,
    completion_mask: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
  # [FUNCTION DOC]
  # Registered as advantage estimator "gae"; jitted. Inputs: rewards, values and
  # completion_mask (all batch by T), gamma, gae_lambda. Runs a reverse scan
  # over time carrying (next GAE, next value), both starting at zero. At step
  # t: delta = reward + gamma times next value minus value;
  # gae = delta + gamma times lambda times next GAE; at masked-out positions
  # both carried quantities pass through unchanged (so padding is skipped).
  # Advantages are the per-step GAE values; returns = advantages + values
  # (computed before normalization). Advantages are then whitened over valid
  # tokens with masked_whiten. Returns (advantages, returns), each batch by T.
  """Compute advantages using Generalized Advantage Estimation (GAE).

  Computing GAE is a two-step process:

  First, compute the temporal difference (TF), `δ_t`, for each timestep `t`:

  ```
  δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
  ```

  Then, compute the GAE advantage, `A_t`, by summing the discounted TD
  residuals. It is calculated recursively, starting from the last timestep:

  ```
  A_t = δ_t + (γ * λ) * A_{t+1}
  ```

  where:

  - `A_t` is the GAE advantage at timestep `t`.
  - `δ_t` is the temporal difference at timestep `t`.
  - `γ` is the discount factor.
  - `λ` is the GAE lambda parameter.
  - `V(s_t)` is the value function at timestep `t`.
  - `r_t` is the reward at timestep `t`.

  Args:
    rewards: A 2D array of rewards for each step in the rollout.
    values: A 2D array of value estimates from the critic for each step.
    completion_mask: A 2D mask, which is 0 for padding tokens.
    gamma: The discount factor, `γ`.
    gae_lambda: The GAE lambda parameter, `λ`.

  Returns:
    A tuple of two 2D arrays - advantages and returns for each step.
  """
  batch_size = values.shape[0]

  def gae_step(state_t_plus_1, xs):
    # [FUNCTION DOC]
    # HELPER: Scan body for one time step as described above; returns the new
    # carry (gae_t, next_values) and gae_t as the step output.
    # Unpack state and inputs.
    gae_t_plus_1, next_values = state_t_plus_1
    rewards_t, values_t, mask_t = xs

    # Compute Temporal Difference (TD).
    delta = rewards_t + gamma * next_values - values_t
    # Compute GAE for this time step.
    gae_t = delta + gamma * gae_lambda * gae_t_plus_1

    # Skip values on non-completion tokens.
    next_values = values_t * mask_t + (1 - mask_t) * next_values
    gae_t = gae_t * mask_t + (1 - mask_t) * gae_t_plus_1

    # New state to carry over comprises `gae_t` and `next_values`. Output for
    # this step is `gae_t`.
    return (gae_t, next_values), gae_t

  _, advantages_transposed = jax.lax.scan(
      gae_step,
      init=(jnp.zeros((batch_size,)), jnp.zeros((batch_size,))),
      xs=(
          jnp.transpose(jnp.array(rewards)),
          jnp.transpose(jnp.array(values)),
          jnp.transpose(jnp.array(completion_mask)),
      ),
      reverse=True,
  )
  advantages = jnp.transpose(advantages_transposed)
  returns = advantages + values

  # Normalise advantages.
  advantages = masked_whiten(advantages, completion_mask)
  return advantages, returns


@jax.jit
def masked_whiten(
    x: jax.Array,
    completion_mask: jax.Array,
) -> jax.Array:
  # [FUNCTION DOC]
  # HELPER: Jitted. Computes the masked mean and the Bessel-corrected masked
  # variance of x and returns (x - mean) times reciprocal sqrt(var + 1e-8),
  # applied to every element including masked ones.
  """Normalize the input array."""
  x_mean = masked_mean(x, completion_mask)
  x_var = masked_var(
      x,
      completion_mask,
      x_mean,
  )
  x = (x - x_mean) * jax.lax.rsqrt(x_var + 1e-8)
  return x


@functools.partial(jax.jit, static_argnames=('axis',))
def masked_mean(
    x: jax.Array, mask: jax.Array, axis: int | None = None
) -> jax.Array:
  # [FUNCTION DOC]
  # HELPER: Jitted with static axis. Casts mask to x's dtype and returns
  # sum(x times mask) divided by (sum(mask) + 1e-8) along axis (all elements
  # when None).
  # KNOWN ISSUE: when x is boolean (as passed for clip fractions), the mask is
  # cast to bool and the result relies on implicit integer promotion.
  """Compute the mean of a masked array."""
  cast_mask = mask.astype(x.dtype)
  return jnp.sum(x * cast_mask, axis=axis) / (
      jnp.sum(cast_mask, axis=axis) + 1e-8
  )


@jax.jit
def masked_var(
    x: jax.Array,
    mask: jax.Array,
    mean: jax.Array | None = None,
) -> jax.Array:
  # [FUNCTION DOC]
  # HELPER: Jitted. Masked variance: uses the given mean or computes the masked
  # mean, takes the masked mean of squared deviations, and multiplies by the
  # Bessel correction n / (n - 1) where n is the mask sum. Returns a scalar.
  # KNOWN ISSUE: with exactly one valid element the correction divides by zero
  # (inf/NaN); with zero valid elements it becomes zero times a negative ratio.
  """Compute the variance of a masked array."""
  cast_mask = mask.astype(x.dtype)
  if mean is None:
    mean = masked_mean(x, cast_mask)

  variance = masked_mean(jnp.square(x - mean), cast_mask)

  mask_sum = cast_mask.sum()
  bessel_corr = mask_sum / (mask_sum - 1)
  return variance * bessel_corr


def compute_entropy_from_logits(logits: jax.Array) -> jax.Array:
  # [FUNCTION DOC]
  # HELPER: Inputs: logits (batch, seq, vocab). Computes log-softmax, then
  # probabilities (softmax of the log-probs, equal to softmax of logits), and
  # returns the negative sum of p times log p over the vocab: per-token
  # entropy, batch by seq.
  # KNOWN ISSUE: the docstring calls the last dimension emb_dim; it is vocab.
  """Computes the entropy of a distribution given its logits.

  Args:
    logits: Logits as returned by the model. Of shape `[batch_size, seq_len,
      emb_dim]`.

  Returns:
    A JAX array of shape `[batch_size, seq_len]`, containing the entropy values.
  """
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  probs = jax.nn.softmax(log_probs)
  return -jnp.sum(probs * log_probs, axis=-1)


# ==============================================================================
# PPO Core
# ==============================================================================


@function_registry.register_policy_loss_fn("ppo")
def ppo_policy_loss_fn(
    model,
    train_example,
    algo_config,
    pad_id,
    eos_id,
    **kwargs,
):
  # [FUNCTION DOC]
  # Registered as policy loss "ppo". Inputs: model, train_example (needs
  # completion ids/mask, prompt ids, per-token advantages, old logps; optional
  # packing fields), algo_config (epsilon_low, epsilon_high, entropy_coef,
  # optional epsilon_c, optional kl_coef, kl_clamp_value), pad_id, eos_id,
  # ignored kwargs. Computes current per-token logps and logits with
  # gradients. ratio = exp(logp - old logp). Standard clipped loss per token is
  # the max of -ratio times A and -clip(ratio, 1 - eps_low, 1 + eps_high)
  # times A. Dual clip (when epsilon_c set): for negative advantages the loss
  # is the min of that and -epsilon_c times A; pg_clipfrac_lower is the masked
  # fraction where this lower clip binds. Policy loss is the masked token mean.
  # If entropy_coef is non-zero, subtracts entropy_coef times masked mean
  # entropy (aux "loss/entropy"). If config has kl_coef > 0 and reference
  # logps exist, adds kl_coef times masked mean "kl"-method KL (aux "kl").
  # Returns (loss, aux) where aux also has pg_clipfrac (fraction of tokens
  # where the clipped term is larger) and pg_clipfrac_lower.
  # KNOWN ISSUE: PPOConfig has no kl_coef field, so the KL loss branch is never
  # active for PPO (KL is applied in rewards instead).
  """PPO policy loss function."""
  epsilon_low = algo_config.epsilon_low
  epsilon_high = algo_config.epsilon_high
  entropy_coef = algo_config.entropy_coef

  completion_ids = train_example.completion_ids
  completion_mask = train_example.completion_mask

  graphdef, state = nnx.split(model)
  per_token_logps, logits = common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=train_example.prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
      return_logits=True,
      segment_ids=getattr(train_example, "segment_ids", None),
      segment_positions=getattr(train_example, "segment_positions", None),
  )

  advantages = train_example.advantages
  old_per_token_logps = train_example.old_per_token_logps

  seq_importance_ratio = jnp.exp(per_token_logps - old_per_token_logps)

  # Compute pg_clipfrac
  pg_losses_1 = -seq_importance_ratio * advantages
  pg_losses_2 = (
      -jnp.clip(seq_importance_ratio, 1 - epsilon_low, 1 + epsilon_high)
      * advantages
  )

  per_token_loss = jnp.maximum(pg_losses_1, pg_losses_2)

  # add dual clip logic
  epsilon_c = getattr(algo_config, "epsilon_c", None)
  if epsilon_c is not None:
    pg_loss_3 = -epsilon_c * advantages
  else:
    pg_loss_3 = per_token_loss
  unreduced_pg_clipfrac_lower = (
      (per_token_loss > pg_loss_3) & (advantages < 0.0)
  ).astype(jnp.float32)
  pg_clipfrac_lower = masked_mean(unreduced_pg_clipfrac_lower, completion_mask)

  pg_loss_clipped_dual = jnp.minimum(pg_loss_3, per_token_loss)
  pg_losses = jnp.where(advantages < 0.0, pg_loss_clipped_dual, per_token_loss)

  aux = {
      "pg_clipfrac": masked_mean(
          jnp.greater(pg_losses_2, pg_losses_1), completion_mask
      ),
      "pg_clipfrac_lower": pg_clipfrac_lower,
  }

  policy_loss = masked_mean(pg_losses, completion_mask)
  loss = policy_loss

  if entropy_coef is not None and entropy_coef != 0.0:
    token_entropy = compute_entropy_from_logits(logits)
    entropy_loss = masked_mean(token_entropy, completion_mask)
    loss = loss - entropy_coef * entropy_loss
    aux["loss/entropy"] = entropy_loss

  # kl penalty term logic as before
  kl_coef = getattr(algo_config, "kl_coef", 0.0)
  if kl_coef > 0.0 and train_example.ref_per_token_logps is not None:
    kl = common.compute_kl_divergence(
        per_token_logps,
        train_example.ref_per_token_logps,
        "kl",
        clamp_value=getattr(algo_config, "kl_clamp_value", None),
    )
    kl_loss = masked_mean(kl, completion_mask)
    loss = loss + kl_coef * kl_loss
    aux["kl"] = kl_loss

  return loss, aux


@function_registry.register_value_loss_fn("ppo")
def ppo_value_loss_fn(
    model: nnx.Module,
    train_example,
    clip_range_value: float | None,
    pad_id: int,
    eos_id: int,
):
  # [FUNCTION DOC]
  # Registered as value loss "ppo". Inputs: critic model, train_example (needs
  # old_values and returns), clip_range_value, pad_id, eos_id. Computes value
  # predictions over prompt+completion with gradients and slices them to the
  # positions preceding each completion token (packed mode: full length minus
  # one, then left padded by one zero). Clipped prediction = old values plus
  # the prediction offset clipped to plus/minus clip_range_value. Per-token
  # loss is the max of squared errors of the raw and clipped predictions
  # against returns; vf_loss = 0.5 times masked token mean. Returns (vf_loss,
  # aux) with vf_loss, vpred_mean, vf_clipfrac (fraction where the clipped
  # error is larger) and return_mean.
  # KNOWN ISSUE: clip_range_value is typed optional but None would fail in the
  # arithmetic.
  """Computes the value loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )
  # ====== Loss ======
  values = train_example.old_values
  returns = train_example.returns

  segment_ids = getattr(train_example, "segment_ids", None)
  if segment_ids is not None:
    # For packed sequences, prompt_ids is empty and completion_ids holds the full sequence.
    # We predict values for token t using the model's output at t-1.
    logits_to_keep = completion_ids.shape[1] - 1
  else:
    logits_to_keep = completion_ids.shape[1]

  # Get new values.
  vpreds = common.compute_score(
      model,
      prompt_ids,
      completion_ids,
      pad_id,
      eos_id,
      stop_gradient=False,
      segment_ids=segment_ids,
      segment_positions=getattr(train_example, "segment_positions", None),
  )
  vpreds = vpreds[:, -logits_to_keep - 1 : -1]

  if segment_ids is not None:
    # Pad the first token's value with 0.0, since it has no preceding token to predict it.
    vpreds = jnp.pad(vpreds, ((0, 0), (1, 0)), constant_values=0.0)
  vpred_clipped = jnp.clip(
      vpreds, values - clip_range_value, values + clip_range_value
  )
  vf_losses1 = jnp.square(vpreds - returns)
  vf_losses2 = jnp.square(vpred_clipped - returns)

  clipped_vf_losses = jnp.maximum(vf_losses1, vf_losses2)
  # "token mean" style of normalisation.
  vf_loss = 0.5 * masked_mean(clipped_vf_losses, completion_mask)

  aux = {
      "vf_loss": vf_loss,
      "vpred_mean": masked_mean(vpreds, completion_mask),
      "vf_clipfrac": masked_mean(
          jnp.greater(vf_losses2, vf_losses1), completion_mask
      ),
      "return_mean": masked_mean(returns, completion_mask),
  }

  return vf_loss, aux


# ==============================================================================
# GRPO Core
# ==============================================================================


@function_registry.register_policy_loss_fn("grpo")
def grpo_loss_fn(
    model,
    train_example,
    algo_config,
    pad_id,
    eos_id,
    **kwargs,
):
  # [FUNCTION DOC]
  # Registered as policy loss "grpo" (used by GRPO, DAPO, Dr. GRPO, GSPO).
  # Inputs: model, train_example, algo_config (beta, epsilon, loss_algo,
  # optional epsilon_high and epsilon_c, loss_agg_mode, temperature,
  # kl_loss_mode, kl_clamp_value), pad_id, eos_id, ignored kwargs. Steps:
  # 1) Current per-token logps and logits with gradients at the configured
  # temperature (packing supported); cast to float32. Old logps default to
  # the detached current logps when absent (ratio 1).
  # 2) log-ratio = logp - old logp; ppo_kl diagnostic = masked mean of its
  # negation; clip the log-ratio to [-20, 20].
  # 3) If loss_algo is "gspo-token": replace by the per-sequence masked mean
  # log-ratio (detached) plus logp minus detached logp (so the gradient is
  # per token), then cap at 10.
  # 4) ratio = exp(log-ratio). Advantages broadcast to tokens if 1-D.
  # Per-token loss = max(-A ratio, -A clip(ratio, 1 - epsilon,
  # 1 + epsilon_high)) where epsilon_high defaults to epsilon. Dual clip for
  # negative advantages as in PPO when epsilon_c is set.
  # 5) Optionally multiply per-token loss by sampler_is_weights.
  # 6) Aggregate with common.aggregate_loss and loss_agg_mode.
  # 7) If reference logps exist, compute KL (kl_loss_mode, clamp), log the
  # token-mean KL as "kl", aggregate it as "kl_loss", and add beta times it
  # to the loss when beta is non-zero and not None.
  # 8) Entropy of logits aggregated the same way (logged only).
  # Returns (loss, aux) with kl, kl_loss, pg_loss, pg_clipfrac, ppo_kl,
  # pg_clipfrac_lower, importance-ratio mean/max/min, mean absolute log-ratio,
  # unclipped/clipped pg loss means, advantage abs-mean/max/min and non-zero
  # fraction, sampler IS weight mean/min (1.0 when absent), entropy.
  """GRPO loss function.

  The loss aims to maximize the expected advantage of the chosen actions while
  constraining the policy updates to stay within a certain range of the
  reference policy.

  Args:
    model: The policy model to be trained.
    train_example: A `TrainExample` instance containing the processed input
      data, including prompt IDs, completion IDs, masks, advantages, and
      per-token log probabilities from the reference and policy models.
    algo_config: The algorithm config.
    pad_id: The pad ID from tokenizer.
    eos_id: The eos ID from.

  Returns:
    A tuple containing the loss and an aux dictionary.
  """
  beta = algo_config.beta
  epsilon = algo_config.epsilon
  loss_algo = algo_config.loss_algo
  epsilon_high = (
      algo_config.epsilon_high
      if hasattr(algo_config, "epsilon_high")
      else epsilon
  )
  epsilon_c = getattr(algo_config, "epsilon_c", None)
  loss_aggregation_mode = algo_config.loss_agg_mode

  completion_ids, completion_mask = (
      train_example.completion_ids,
      train_example.completion_mask,
  )

  # TODO(tsbao): split can be avoided with updated peft_trainer model handling.
  graphdef, state = nnx.split(model)
  per_token_logps, logits = common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=train_example.prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
      return_logits=True,
      segment_ids=getattr(train_example, "segment_ids", None),
      segment_positions=getattr(train_example, "segment_positions", None),
      temperature=algo_config.temperature,
  )
  per_token_logps = jnp.astype(per_token_logps, jnp.float32)
  # TODO(tsbao): We should handle token level advantages.
  advantages = jnp.astype(train_example.advantages, jnp.float32)

  if train_example.old_per_token_logps is None:
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
  else:
    old_per_token_logps = jnp.astype(
        train_example.old_per_token_logps, jnp.float32
    )

  seq_importance_ratio = per_token_logps - old_per_token_logps
  # Record KL divergence before clipping.
  ppo_kl = masked_mean(-seq_importance_ratio, completion_mask)

  seq_importance_ratio = jnp.clip(seq_importance_ratio, max=20.0, min=-20.0)

  # TODO(sizhi): Refactor this to a separate function.
  if loss_algo == "gspo-token":
    seq_importance_ratio = (seq_importance_ratio * completion_mask).sum(
        axis=-1
    ) / jnp.clip(completion_mask.sum(-1), min=1)
    seq_importance_ratio = (
        per_token_logps
        - jax.lax.stop_gradient(per_token_logps)
        + jnp.expand_dims(jax.lax.stop_gradient(seq_importance_ratio), axis=-1)
    )
    seq_importance_ratio = jnp.clip(seq_importance_ratio, max=10.0)

  is_ratio = jnp.exp(seq_importance_ratio)

  # Advantages must be broadcast against seq_length.
  # When sequence packing is used, advantages are already 2D [B, seq_length].
  # When unpacked, they are 1D [B].
  adv = advantages if advantages.ndim == 2 else jnp.expand_dims(advantages, 1)

  pg_loss_1 = -adv * is_ratio
  pg_loss_2 = -adv * jnp.clip(is_ratio, 1 - epsilon, 1 + epsilon_high)

  per_token_loss = jnp.maximum(pg_loss_1, pg_loss_2).astype(jnp.float32)

  clipped_fraction = masked_mean(
      jnp.greater(pg_loss_2, pg_loss_1), completion_mask
  )

  # dual-clip ppo loss
  if epsilon_c is not None:
    pg_loss_3 = -epsilon_c * adv
  else:
    pg_loss_3 = per_token_loss

  # pg_clipfrac_lower measures how often dual-clip ppo kicks in.
  # It kicks in when the standard clipped loss is larger than pg_loss_3
  # for instances with negative advantages.
  unreduced_pg_clipfrac_lower = (
      (per_token_loss > pg_loss_3) & (adv < 0.0)
  ).astype(jnp.float32)
  pg_clipfrac_lower = common.aggregate_loss(
      unreduced_pg_clipfrac_lower, completion_mask, loss_aggregation_mode
  )

  pg_loss_clipped_dual = jnp.minimum(pg_loss_3, per_token_loss)
  per_token_loss = jnp.where(adv < 0.0, pg_loss_clipped_dual, per_token_loss)

  # Optional truncated importance-sampling (TIS) correction for the residual
  # sampler-vs-trainer log-probability mismatch. The weights are precomputed
  # upstream (already detached and threshold-clipped) and applied per token
  # BEFORE loss aggregation so they affect the gradient through the loss
  # magnitude only, not as a stop-gradient bias on the ratio.
  sampler_is_weights = getattr(train_example, "sampler_is_weights", None)
  if sampler_is_weights is not None:
    per_token_loss = per_token_loss * sampler_is_weights.astype(jnp.float32)

  loss = common.aggregate_loss(
      per_token_loss, completion_mask, loss_aggregation_mode
  )
  # Per-token diagnostics — log only over assistant tokens (completion_mask).
  is_ratio_mean = masked_mean(is_ratio, completion_mask)
  is_ratio_max = jnp.max(jnp.where(completion_mask > 0, is_ratio, 0.0))
  is_ratio_min = jnp.min(
      jnp.where(completion_mask > 0, is_ratio, jnp.inf)
  )
  log_ratio_abs_mean = masked_mean(
      jnp.abs(seq_importance_ratio), completion_mask
  )
  pg_loss_1_mean = masked_mean(pg_loss_1, completion_mask)
  pg_loss_2_mean = masked_mean(pg_loss_2, completion_mask)
  adv_broadcast = jnp.broadcast_to(adv, completion_mask.shape)
  adv_abs_mean = masked_mean(jnp.abs(adv_broadcast), completion_mask)
  adv_max = jnp.max(jnp.where(completion_mask > 0, adv_broadcast, -jnp.inf))
  adv_min = jnp.min(jnp.where(completion_mask > 0, adv_broadcast, jnp.inf))
  nonzero_adv_frac = masked_mean(
      (jnp.abs(adv_broadcast) > 1e-8).astype(jnp.float32), completion_mask
  )
  aux = {
      "kl": 0.0,
      "kl_loss": 0.0,
      "pg_loss": loss,
      "pg_clipfrac": clipped_fraction,
      "ppo_kl": ppo_kl,
      "pg_clipfrac_lower": pg_clipfrac_lower,
      "is_ratio/mean": is_ratio_mean,
      "is_ratio/max": is_ratio_max,
      "is_ratio/min": is_ratio_min,
      "log_ratio/abs_mean": log_ratio_abs_mean,
      "pg_loss/unclipped_mean": pg_loss_1_mean,
      "pg_loss/clipped_mean": pg_loss_2_mean,
      "advantage/abs_mean": adv_abs_mean,
      "advantage/max": adv_max,
      "advantage/min": adv_min,
      "advantage/nonzero_frac": nonzero_adv_frac,
  }
  if sampler_is_weights is not None:
    sis = sampler_is_weights.astype(jnp.float32)
    aux["sampler_is/weight_mean"] = masked_mean(sis, completion_mask)
    aux["sampler_is/weight_min"] = jnp.min(
        jnp.where(completion_mask > 0, sis, jnp.inf)
    )
  else:
    aux["sampler_is/weight_mean"] = jnp.float32(1.0)
    aux["sampler_is/weight_min"] = jnp.float32(1.0)
  # We do not always compute KL divergence (e.g. when beta is 0.0 unless
  # force_compute_kl is True).
  if train_example.ref_per_token_logps is not None:
    kl = common.compute_kl_divergence(
        per_token_logps,
        train_example.ref_per_token_logps,
        algo_config.kl_loss_mode,
        clamp_value=algo_config.kl_clamp_value,
    )
    # Log mean KL.
    aux["kl"] = jnp.astype(
        (kl * completion_mask).sum() / jnp.clip(completion_mask.sum(), min=1),
        jnp.float32,
    )
    kl_loss = common.aggregate_loss(kl, completion_mask, loss_aggregation_mode)
    aux["kl_loss"] = kl_loss
    if beta is not None and beta != 0.0:
      loss = loss + beta * kl_loss

  token_entropy = compute_entropy_from_logits(logits)
  entropy_loss = common.aggregate_loss(
      token_entropy, completion_mask, loss_aggregation_mode
  )
  aux["entropy"] = entropy_loss

  return loss, aux


@function_registry.register_advantage_estimator("grpo")
def compute_advantages(rewards: np.ndarray, num_generations: int) -> np.ndarray:
  # [FUNCTION DOC]
  # Registered as advantage estimator "grpo". Input rewards (1-D, grouped in
  # consecutive blocks of num_generations) and num_generations. Per group
  # computes mean and sample std (ddof 1), repeats them per member, and returns
  # (reward - mean) / (std + 1e-6). Raises if the length is not divisible by
  # num_generations; with num_generations 1 the std is NaN.
  """Compute group relative advantages.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    Group relative advantages.
  """
  mean_grouped_rewards = rewards.reshape(-1, num_generations).mean(axis=-1)
  std_grouped_rewards = rewards.reshape(-1, num_generations).std(
      axis=-1, ddof=1
  )

  mean_grouped_rewards = mean_grouped_rewards.repeat(num_generations)
  std_grouped_rewards = std_grouped_rewards.repeat(num_generations)
  return (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-6)


@function_registry.register_advantage_estimator("rloo")
def compute_rloo_advantages(
    rewards: jax.Array, num_generations: int
) -> jax.Array:
  # [FUNCTION DOC]
  # Registered as advantage estimator "rloo". Returns zeros if num_generations
  # is below 2. Otherwise, per group, baseline for each member is the mean of
  # the other members' rewards (group sum minus own reward, divided by G - 1);
  # returns reward minus baseline, flattened to 1-D.
  """Compute RLOO (REINFORCE Leave-One-Out) advantages.

  RLOO computes a baseline for each completion by averaging the rewards of all
  other completions to the same prompt.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    RLOO advantages.
  """
  if num_generations < 2:
    # RLOO requires at least 2 samples to calculate a baseline.
    return jnp.zeros_like(rewards)

  reshaped_rewards = rewards.reshape(-1, num_generations)
  loo_mean = (
      reshaped_rewards.sum(axis=-1, keepdims=True) - reshaped_rewards
  ) / (num_generations - 1)
  rloo_advantages = reshaped_rewards - loo_mean

  return rloo_advantages.flatten()


# ==============================================================================
# DrGRPO Core
# ==============================================================================


@function_registry.register_advantage_estimator("drgrpo")
def compute_drgrpo_advantages(
    rewards: jax.Array, num_generations: int
) -> jax.Array:
  # [FUNCTION DOC]
  # Registered as advantage estimator "drgrpo". Returns rewards minus their
  # group mean (repeated per member), without dividing by the std.
  """Group relative advantages -- done right.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    Group relative advantages.
  """
  mean_grouped_rewards = rewards.reshape(-1, num_generations).mean(axis=1)
  return rewards - mean_grouped_rewards.repeat(num_generations)
