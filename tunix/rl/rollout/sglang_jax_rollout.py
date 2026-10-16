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
# Rollout backend that delegates generation to the sglang-jax inference
# engine through tunix.generate.sglang_jax_sampler. Builds the weight mapping
# between the Tunix model and sglang-jax, loads the initial model state into
# the engine, and implements the BaseRollout interface.
# KNOWN ISSUE: the module docstring says "with Tunix sampler" but generation
# uses the sglang-jax sampler.
"""sglang jax rollout worker with Tunix sampler."""

from typing import Any, Dict, Optional, Tuple

from flax import nnx
import jax
import jaxtyping
from tunix.generate import mappings
from tunix.generate import sglang_jax_sampler
from tunix.rl import common
from tunix.rl.rollout import base_rollout


class SglangJaxRollout(base_rollout.BaseRollout):
  # [CLASS DOC]
  # BaseRollout implementation holding one SglangJaxSampler. generate also
  # stores the last sampler output on the instance attribute output.
  """sglang jax rollout worker."""

  def __init__(
      self,
      model: Any,
      tokenizer: Any,
      mesh: jax.sharding.Mesh,
      rollout_config: base_rollout.RolloutConfig,
  ):
    # [FUNCTION DOC]
    # Inputs: model (Tunix model used for weight mapping and initial weights),
    # tokenizer, JAX device mesh, RolloutConfig. Builds a MappingConfig for the
    # "sglang_jax" backend from rollout_mapping_config and the model, then creates
    # the sampler with an SglangJaxConfig populated from every
    # rollout_sglang_jax_* field of the rollout config (model version, context
    # length, static memory fraction, random-weight init, radix cache disable,
    # deterministic sampling, sort-based top-p/k/min-p, static LoRA settings,
    # single-process mode, LoRA target modules / rank / scaling, precompile batch
    # and token paddings, chunked prefill size, page size, load format, max
    # running requests, log level), plus rollout_sglang_jax_kwargs as extra
    # keyword arguments. Finally extracts the model's nnx state and loads it into
    # the engine (side effect: engine startup and weight transfer).
    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=model,
        backend="sglang_jax",
    )
    self._sampler = sglang_jax_sampler.SglangJaxSampler(
        tokenizer=tokenizer,
        config=sglang_jax_sampler.SglangJaxConfig(
            mesh=mesh,
            mapping_config=mapping_config,
            model_version=rollout_config.rollout_sglang_jax_model_version,
            context_length=rollout_config.rollout_sglang_jax_context_length,
            mem_fraction_static=rollout_config.rollout_sglang_jax_mem_fraction_static,
            init_with_random_weights=rollout_config.rollout_sglang_jax_init_with_random_weights,
            disable_radix_cache=rollout_config.rollout_sglang_jax_disable_radix_cache,
            enable_deterministic_sampling=rollout_config.rollout_sglang_jax_enable_deterministic_sampling,
            use_sort_for_toppk_minp=rollout_config.rollout_sglang_jax_use_sort_for_toppk_minp,
            enable_static_lora=rollout_config.rollout_sglang_jax_enable_static_lora,
            enable_single_process=rollout_config.rollout_sglang_jax_enable_single_process,
            lora_target_modules=rollout_config.rollout_sglang_jax_lora_target_modules,
            max_lora_rank=rollout_config.rollout_sglang_jax_max_lora_rank,
            lora_scaling=rollout_config.rollout_sglang_jax_lora_scaling,
            precompile_bs_paddings=rollout_config.rollout_sglang_jax_precompile_bs_paddings,
            precompile_token_paddings=rollout_config.rollout_sglang_jax_precompile_token_paddings,
            chunked_prefill_size=rollout_config.rollout_sglang_jax_chunked_prefill_size,
            page_size=rollout_config.rollout_sglang_jax_page_size,
            load_format=rollout_config.rollout_sglang_jax_load_format,
            max_running_requests=rollout_config.rollout_sglang_jax_max_running_requests,
            log_level=rollout_config.rollout_sglang_jax_log_level,
        ),
        **rollout_config.rollout_sglang_jax_kwargs,
    )
    state = nnx.state(model)
    self._sampler.load_checkpoint(state)

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    # [FUNCTION DOC]
    # Inputs: prompt strings, RolloutConfig, extra kwargs forwarded to the
    # sampler. Calls the sampler with max generation steps, max prompt length,
    # temperature, top_p, top_k, seed, echo off and padded outputs; stores the
    # result in self.output. Returns RolloutOutput with text, no logits, tokens,
    # left-padded prompt tokens and logprobs.
    # KNOWN ISSUE: rollout_config.eos_tokens and return_logprobs are not forwarded.
    """Generates samples from the model."""
    self.output = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        seed=rollout_config.seed,
        echo=False,
        pad_output=True,
        **kwargs,
    )

    return base_rollout.RolloutOutput(
        text=self.output.text,
        logits=None,
        tokens=self.output.tokens,
        left_padded_prompt_tokens=self.output.padded_prompt_tokens,
        logprobs=self.output.logprobs,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    # [FUNCTION DOC]
    # Intended to return per-token log-probs of completions under the rollout
    # model by calling common.compute_per_token_logps with the pad/eos ids and
    # taking the first element of the result.
    # KNOWN ISSUE: compute_per_token_logps expects a graphdef and a state, but
    # only the module is passed (state is missing), so the call fails; and since
    # return_logits defaults to False the result is an array, so indexing with
    # zero would select the first batch row rather than a tuple element.
    """Returns per-token log probabilities from the rollout policy."""
    return common.compute_per_token_logps(
        self.model(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=self.pad_id(),
        eos_id=self.eos_id(),
    )[0]

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    # [FUNCTION DOC]
    # Forwards params and optional filter_types to the sampler's update_params,
    # which syncs new weights into the sglang-jax engine. Returns None.
    self._sampler.update_params(params, filter_types)

  def pad_id(self) -> int:
    # [FUNCTION DOC]
    # Returns the sampler tokenizer's pad id.
    return self._sampler.tokenizer.pad_id()

  def eos_id(self) -> int:
    # [FUNCTION DOC]
    # Returns the sampler tokenizer's EOS id.
    return self._sampler.tokenizer.eos_id()

  def model(self) -> nnx.Module:
    # [FUNCTION DOC]
    # Returns the sampler's transformer object (engine-side model).
    return self._sampler.transformer

  def model_state(self):
    # [FUNCTION DOC]
    # Returns the sampler's transformer state (current engine weights).
    return self._sampler.transformer_state
