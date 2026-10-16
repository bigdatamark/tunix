# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Mostly forked from https://github.com/google-deepmind/gemma/blob/2a892cd9462618ce8bf8f0b5830466498858beb4/gemma/gm/nn/gemma4/_moe.py
# [MODULE DOC]
# Token-choice top-k mixture-of-experts layer for Gemma 4 (MoERagged), forked
# from google-deepmind/gemma. Each token is routed to num_experts_per_tok
# experts; tokens are sorted by expert so all experts run as two
# jax.lax.ragged_dot grouped matmuls (gated-GELU expert FFN), then outputs
# are unsorted and combined with renormalized router probabilities.
# Private helpers handle renormalization, dispatch (sorting) and collection
# (unsorting). No side effects.

from flax import nnx
import jax
import jax.numpy as jnp


def _renormalization_factor(router_probs: jax.Array, choices: jax.Array):
  # [FUNCTION DOC]
  # HELPER: given router_probs [..., E] and chosen expert ids choices [..., k],
  # returns the sum of probabilities of the chosen experts ([..., 1]), or 1.0
  # where that sum is not positive; dividing by it renormalizes the chosen
  # probabilities to sum to 1.
  """Computes the renormalization factor for routing weights."""
  indicator = jax.nn.one_hot(
      choices, router_probs.shape[-1], dtype=router_probs.dtype
  ).sum(axis=-2)
  gate_weights = indicator * router_probs
  renormalization_factor = jnp.sum(gate_weights, axis=-1, keepdims=True)
  return jnp.where(renormalization_factor > 0.0, renormalization_factor, 1.0)


def _expert_dispatch(
    x: jax.Array,
    expert_choices: jax.Array,
    expert_weights: jax.Array,
):
  # [FUNCTION DOC]
  # HELPER: sorts token copies by expert.
  # Inputs: x [G, S, D] (flattened internally to [G*S, D]), expert_choices
  # [G, S, k] int, expert_weights [G, S, E] (renormalized probabilities).
  # Behavior: flattens the choices to G*S*k entries, argsorts them by expert id
  # (xs_order) and computes the inverse permutation; gathers the token row for
  # each sorted entry; counts tokens per expert from one-hot choices; computes
  # combine weights [G, S, k] = probability of each chosen expert.
  # Output tuple: (tokens_per_expert [E] int32, sorted_xs [G*S*k, D],
  # reverse permutation [G*S*k], combine weights [G, S, k]).
  """Sorts tokens by expert for each expert choice."""
  num_groups, group_size, k = expert_choices.shape
  x = x.reshape((-1, x.shape[-1]))
  batch_size = num_groups * group_size
  num_experts = expert_weights.shape[2]

  expert_choices_flat = expert_choices.ravel()
  xs_order = expert_choices_flat.argsort()
  xs_reverse_argsort = xs_order.argsort()
  xs_indices = jnp.repeat(jnp.arange(batch_size), k)[xs_order]
  sorted_xs = x[xs_indices, :]
  expert_choices_oh = jax.nn.one_hot(
      expert_choices, num_classes=num_experts, dtype=jnp.int32
  )
  xs_tokens_per_expert = jnp.sum(expert_choices_oh, axis=(0, 1, 2))
  xs_combine_weights = (
      (
          expert_choices_oh[:, :, :, :num_experts].astype(jnp.float32)
          * jnp.expand_dims(expert_weights, axis=2)
      )
      .sum(axis=-1)
      .astype(expert_weights.dtype)
  )
  return (
      xs_tokens_per_expert,
      sorted_xs,
      xs_reverse_argsort,
      xs_combine_weights,
  )


def _expert_collect(
    sorted_xs: jax.Array,
    xs_reverse_argsort: jax.Array,
    xs_combine_weights: jax.Array,
) -> jax.Array:
  # [FUNCTION DOC]
  # HELPER: applies the inverse permutation to sorted expert outputs
  # [G*S*k, D] to restore (token, choice) order and reshapes to [G, S, k, D]
  # (shape taken from xs_combine_weights). Weights are not applied here.
  """Unshuffles tokens back to their original token order and reshapes."""
  num_groups, group_size, k = xs_combine_weights.shape
  xs = sorted_xs[xs_reverse_argsort]
  xs_reshaped = xs.reshape((num_groups, group_size, k, -1))
  return xs_reshaped


class MoERagged(nnx.Module):
  # [CLASS DOC]
  # Mixture-of-experts FFN. Parameters: router_logits [D, E] (router weight,
  # unsharded), gating_einsum [E, 2, F, D] (stacked gate and up projections,
  # sharding exp_weight_edf), linear [E, F, D] (down projection, sharding
  # exp_weight_efd), per_expert_scale [E] (ones) and router_scale [D] (ones).
  # D = embed_dim, F = expert_dim, E = num_experts.
  """Mixture of Experts using ragged_dot."""

  def __init__(
      self,
      config,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Reads embed_dim, expert_dim, num_experts and num_experts_per_tok from the
    # Gemma 4 config and creates the parameters (normal init in
    # config.param_dtype for the weights).
    self.config = config
    self.features = config.embed_dim
    self.hidden_dim = config.expert_dim
    self.num_experts = config.num_experts
    self.num_experts_per_datapoint = config.num_experts_per_tok

    self.router_logits = nnx.Param(
        nnx.initializers.normal(dtype=config.param_dtype)(
            rngs.params(), (self.features, self.num_experts)
        )
    )
    self.gating_einsum = nnx.Param(
        nnx.initializers.normal(dtype=config.param_dtype)(
            rngs.params(), (self.num_experts, 2, self.hidden_dim, self.features)
        ),
        sharding=config.shd_config.exp_weight_edf,
    )
    self.linear = nnx.Param(
        nnx.initializers.normal(dtype=config.param_dtype)(
            rngs.params(), (self.num_experts, self.hidden_dim, self.features)
        ),
        sharding=config.shd_config.exp_weight_efd,
    )
    self.per_expert_scale = nnx.Param(
        jnp.ones((self.num_experts,), dtype=config.param_dtype)
    )
    self.router_scale = nnx.Param(
        jnp.ones((self.features,), dtype=config.param_dtype)
    )

  def _router(self, router_logits: jax.Array):
    # [FUNCTION DOC]
    # HELPER: in float32, computes softmax probabilities over experts and the
    # top-k expert indices (jax.lax.approx_max_k on the logits), and returns
    # (probabilities renormalized over the chosen experts [G, S, E], choices
    # [G, S, k]).
    # KNOWN ISSUE: approx_max_k may return an approximate top-k on TPU.
    router_logits = router_logits.astype(jnp.float32)
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    _, choices = jax.lax.approx_max_k(
        router_logits,
        k=self.num_experts_per_datapoint,
    )
    weights = router_probs / _renormalization_factor(router_probs, choices)
    return weights, choices

  def _run_ffw_and_routing(
      self,
      x: jax.Array,
      expert_choices: jax.Array,
      expert_weights: jax.Array,
  ):
    # [FUNCTION DOC]
    # HELPER: runs the experts. Dispatches tokens, reshapes gating weights to
    # [E, D, 2F], computes ragged_dot(sorted_xs, gate) in config.dtype, splits
    # into gate and up halves, activation = gelu(gate) * up, down projects with
    # ragged_dot over linear, multiplies each row by its expert's
    # per_expert_scale, unsorts to [G, S, k, D] and sums over k weighted by the
    # combine weights. Returns [G, S, D].
    (
        xs_tokens_per_expert,
        sorted_xs,
        xs_reverse_argsort,
        xs_combine_weights,
    ) = _expert_dispatch(x, expert_choices, expert_weights)

    w_gate = self.gating_einsum.value
    w_gate = jnp.transpose(w_gate, (0, 3, 1, 2))
    w_gate = w_gate.reshape(
        self.num_experts, self.features, 2 * self.hidden_dim
    )

    gate_out = jax.lax.ragged_dot(
        sorted_xs,
        w_gate.astype(self.config.dtype),
        group_sizes=xs_tokens_per_expert,
    )

    gate_out = gate_out.reshape(gate_out.shape[0], 2, self.hidden_dim)
    x1 = gate_out[:, 0, :]
    x2 = gate_out[:, 1, :]
    activation = nnx.gelu(x1) * x2

    expert_outputs = jax.lax.ragged_dot(
        activation,
        self.linear.value.astype(self.config.dtype),
        group_sizes=xs_tokens_per_expert,
    )

    expert_indices = jnp.repeat(
        jnp.arange(self.num_experts),
        xs_tokens_per_expert,
        total_repeat_length=expert_outputs.shape[0],
    )
    per_expert = self.per_expert_scale.value.astype(expert_outputs.dtype)
    expert_outputs = expert_outputs * per_expert[expert_indices, None]

    out = _expert_collect(
        expert_outputs,
        xs_reverse_argsort=xs_reverse_argsort,
        xs_combine_weights=xs_combine_weights,
    )

    out = jnp.einsum(
        'blkd,blk->bld',
        out,
        xs_combine_weights,
        preferred_element_type=out.dtype,
    )
    return out

  def __call__(self, x):
    # [FUNCTION DOC]
    # Input x [G, S, D] (batch, sequence, features). Router input =
    # RMS-normalized x (no learned scale, eps 1e-6) times D^-0.5 times
    # router_scale; logits = router_input @ router_logits [G, S, E]; route with
    # _router; return the expert mixture from _run_ffw_and_routing ([G, S, D]).
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    router_input = x * jax.lax.rsqrt(var + 1e-06).astype(x.dtype)

    root_size = jax.lax.rsqrt(
        jnp.array(self.features, dtype=router_input.dtype)
    )
    router_input = (
        router_input
        * root_size
        * self.router_scale.value.astype(router_input.dtype)
    )
    logits = jnp.einsum(
        'gsd,de->gse',
        router_input,
        self.router_logits.value.astype(router_input.dtype),
    )
    weights, choices = self._router(logits)
    out = self._run_ffw_and_routing(x, choices, weights)
    return out
