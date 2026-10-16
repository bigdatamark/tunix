# Copyright 2025 Google LLC
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

# [MODULE DOC]
# Rollout backend that uses the native Tunix sampler (tunix.generate.sampler)
# running in the same JAX process. Implements the BaseRollout interface:
# generation, per-token log-prob recomputation, in-place weight updates
# (with optional resharding and dtype casting), and tokenizer pad/eos ids.
"""Vanilla rollout worker with Tunix sampler."""

import dataclasses
import functools
import operator
from typing import Any, Optional, Tuple

from flax import nnx
import jax
import jaxtyping
from tunix.generate import sampler
from tunix.rl import common
from tunix.rl import reshard
from tunix.rl import utils
from tunix.rl.rollout import base_rollout


class VanillaRollout(base_rollout.BaseRollout):
  # [CLASS DOC]
  # BaseRollout implementation wrapping a single sampler.Sampler instance built
  # from the given model, tokenizer and KV-cache configuration.
  """Vanilla rollout worker."""

  def __init__(
      self,
      model: nnx.Module,
      tokenizer: Any,
      cache_config_or_size: base_rollout.CacheConfig,
  ):
    # [FUNCTION DOC]
    # Inputs: model (nnx module), tokenizer, cache config (a dataclass whose
    # fields match sampler.CacheConfig). Converts the config to a dict and builds
    # sampler.CacheConfig from it, then constructs and stores a Sampler. Raises
    # if the config is not a dataclass or has incompatible fields.
    self._sampler = sampler.Sampler(
        model,
        tokenizer,
        sampler.CacheConfig(**dataclasses.asdict(cache_config_or_size)),
    )

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    # [FUNCTION DOC]
    # Inputs: list of prompt strings, RolloutConfig, extra kwargs (ignored).
    # Calls the sampler with max generation steps, max prompt length, echo off,
    # temperature, top_p, top_k, seed, unpadded outputs, eos tokens and the
    # return_logprobs flag from the config. Returns RolloutOutput with text,
    # logits, per-sample token lists, left-padded prompt tokens and logprobs from
    # the sampler output.
    """Generates samples from the model."""
    output = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        echo=False,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,
        pad_output=False,
        eos_tokens=rollout_config.eos_tokens,
        return_logprobs=rollout_config.return_logprobs,
    )
    return base_rollout.RolloutOutput(
        text=output.text,
        logits=output.logits,
        tokens=output.tokens,
        left_padded_prompt_tokens=output.padded_prompt_tokens,
        logprobs=output.logprobs,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    # [FUNCTION DOC]
    # Inputs: prompt token array and completion token array. Gets the sampler's
    # current graphdef and state and returns common.compute_per_token_logps with
    # the tokenizer's pad and eos ids, gradients stopped and no logits:
    # per-token log-probs of completions under the rollout model.
    # KNOWN ISSUE: the signature does not accept the extra completion_mask or
    # keyword arguments that other backends/callers may pass, and temperature is
    # not applied.
    """Returns per-token log probabilities from the rollout policy."""
    graphdef, state = self._sampler.model_def_and_state()
    return common.compute_per_token_logps(
        graphdef,
        state,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=self.pad_id(),
        eos_id=self.eos_id(),
        stop_gradient=True,
        return_logits=False,
    )

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    # [FUNCTION DOC]
    # Inputs: params (pytree of new weights), optional filter_types. If
    # filter_types is given, takes the rollout model's state filtered by those
    # variable types as a target and reshards params onto it; otherwise uses
    # params as is. Flattens the new params; if their leaf dtype (taken from the
    # first leaf) differs from the sampler's current parameter dtype, casts all
    # new leaves to the rollout dtype. Flattens the current sampler state, merges
    # the two flat dicts (new values override old keys), rebuilds the tree with
    # the old tree structure, merges with the sampler's transformer graphdef and
    # assigns the resulting nnx.Param variables back to the sampler. Side effect:
    # replaces sampler weights. Returns None.
    # KNOWN ISSUE: if new params contain keys absent from the old state, the
    # merged dict has extra values and unflattening with the old treedef fails;
    # dtype detection only inspects the first leaf.
    if filter_types is not None:
      dst_params = nnx.state(self.model(), filter_types)
      resharded_params = reshard.reshard_pytree(params, dst_params)
    else:
      resharded_params = params
    flat_new_params, _ = utils.to_flat_dict(resharded_params)
    # TODO(linchai): Cast on rollout devices when from lower precision to
    # higher precision.
    new_params_precision = jax.tree.leaves(flat_new_params)[0].dtype
    rollout_precision = jax.tree.leaves(self._sampler.transformer_state)[
        0
    ].dtype
    if new_params_precision != rollout_precision:
      flat_new_params = jax.tree.map(
          lambda x: x.astype(rollout_precision), flat_new_params
      )
    flat_old_params, tree_def = utils.to_flat_dict(
        self._sampler.transformer_state
    )
    merged_params = functools.reduce(
        operator.ior, [flat_old_params, flat_new_params], {}
    )
    merged_params = jax.tree.unflatten(tree_def, merged_params.values())
    new_model = nnx.merge(self._sampler._transformer_graphdef, merged_params)  # pylint: disable=protected-access
    self._sampler.transformer_state = nnx.variables(new_model, nnx.Param)

  def pad_id(self) -> int:
    # [FUNCTION DOC]
    # Returns the sampler tokenizer's pad token id.
    return self._sampler.tokenizer.pad_id()

  def eos_id(self) -> int:
    # [FUNCTION DOC]
    # Returns the sampler tokenizer's EOS token id.
    return self._sampler.tokenizer.eos_id()

  def model(self) -> nnx.Module:
    # [FUNCTION DOC]
    # Returns the sampler's transformer module (the live rollout model).
    return self._sampler.transformer
