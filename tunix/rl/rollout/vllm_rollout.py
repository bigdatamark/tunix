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
# Rollout backend that delegates generation to vLLM (JAX/TPU backend) through
# tunix.generate.vllm_sampler. Builds the Tunix-to-vLLM weight mapping,
# configures the vLLM engine from RolloutConfig, loads the initial weights,
# and implements the BaseRollout interface. Per-token log-probs are taken
# from the last generation's output rather than recomputed.
# KNOWN ISSUE: the module docstring says "with Tunix sampler" but generation
# uses vLLM.
"""vLLM rollout worker with Tunix sampler."""

from typing import Any, Dict, Optional, Tuple

from flax import nnx
import jax
import jaxtyping
from tunix.generate import mappings
from tunix.generate import vllm_sampler
from tunix.rl.rollout import base_rollout


class VllmRollout(base_rollout.BaseRollout):
  # [CLASS DOC]
  # BaseRollout implementation holding one VllmSampler. generate stores the last
  # sampler output on attribute output, which get_per_token_logps returns.
  """vLLM rollout worker."""

  def __init__(
      self,
      model: Any,
      tokenizer: Any,
      cache_config_or_size: base_rollout.CacheConfig | int,
      mesh: jax.sharding.Mesh,
      rollout_config: base_rollout.RolloutConfig,
  ):
    # [FUNCTION DOC]
    # Inputs: model (Tunix model for mapping and initial weights), tokenizer,
    # cache_config_or_size (used directly as vLLM max_model_len, so effectively an
    # int), JAX mesh, RolloutConfig. Builds a MappingConfig for backend
    # "vllm_jax"; creates VllmSampler with VllmConfig from rollout config fields
    # (server mode, mapping, return_logprobs, random-weight init, TPU backend
    # type, additional config, DP attention, HBM utilization, LoRA config, mesh,
    # tensor/data/expert parallel sizes, delete-dst-buffers, reshard chunk size,
    # sampling kwargs) and engine kwargs: model version, max_model_len, async
    # scheduling, max batched tokens, max sequences, HF config path, max_logprobs
    # fixed to 1 (only sampled-token logprobs), logprobs mode, then any
    # rollout_vllm_kwargs (which may override earlier keys). Then loads the
    # model's nnx state into the engine. Side effects: engine startup, weight
    # transfer.
    # KNOWN ISSUE: if a CacheConfig object is passed instead of an int it is
    # forwarded as max_model_len unchanged.
    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=model,
        backend="vllm_jax",
    )
    self._sampler = vllm_sampler.VllmSampler(
        tokenizer=tokenizer,
        config=vllm_sampler.VllmConfig(
            server_mode=rollout_config.rollout_vllm_server_mode,
            mapping_config=mapping_config,
            return_logprobs=rollout_config.return_logprobs,
            init_with_random_weights=rollout_config.rollout_vllm_init_with_random_weights,
            tpu_backend_type=rollout_config.rollout_vllm_tpu_backend_type,
            additional_config=rollout_config.rollout_vllm_additional_config,
            enable_dp_attention=rollout_config.rollout_vllm_enable_dp_attention,
            hbm_utilization=rollout_config.rollout_vllm_hbm_utilization,
            lora_config=rollout_config.rollout_vllm_lora_config,
            mesh=mesh,
            tensor_parallel_size=rollout_config.tensor_parallel_size,
            data_parallel_size=rollout_config.data_parallel_size,
            expert_parallel_size=rollout_config.expert_parallel_size,
            delete_dst_buffers=rollout_config.rollout_vllm_delete_dst_buffers,
            reshard_chunk_size=rollout_config.rollout_vllm_reshard_chunk_size,
            engine_kwargs={
                "model": rollout_config.rollout_vllm_model_version,
                "max_model_len": cache_config_or_size,
                "async_scheduling": (
                    rollout_config.rollout_vllm_async_scheduling
                ),
                "max_num_batched_tokens": (
                    rollout_config.rollout_vllm_max_num_batched_tokens
                ),
                "max_num_seqs": rollout_config.rollout_vllm_max_num_seqs,
                "hf_config_path": rollout_config.rollout_vllm_hf_config_path,
                "max_logprobs": (
                    1
                ),  # We only need the logprobs of the sampled tokens
                "logprobs_mode": rollout_config.rollout_vllm_logprobs_mode,
                **rollout_config.rollout_vllm_kwargs,
            },
            sampling_kwargs=rollout_config.rollout_vllm_sampling_kwargs,
        ),
    )
    state = nnx.state(model)
    self._sampler.load_checkpoint(state)

  @property
  def mesh(self) -> jax.sharding.Mesh:
    # [FUNCTION DOC]
    # Property returning the sampler's device mesh.
    return self._sampler.mesh

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    # [FUNCTION DOC]
    # Inputs: prompts, RolloutConfig, extra kwargs forwarded to the sampler.
    # Calls the sampler with generation length, max prompt length, temperature,
    # top_p, top_k, seed, echo off and padded outputs; stores the result on
    # self.output. Returns RolloutOutput with text, no logits, tokens, left-padded
    # prompt tokens and logprobs.
    # KNOWN ISSUE: rollout_config.eos_tokens is not forwarded.
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
    # Ignores its token inputs and returns the logprobs from the most recent
    # generate call (stored on self.output).
    # KNOWN ISSUE: raises AttributeError if generate has not been called, and
    # returns stale/mismatched values if the given tokens are not the last
    # generated batch; despite the comment, it does return self.output.logprobs.
    """Returns per-token log probabilities from the rollout policy."""
    # b/428730696, we cannot return self.output.logprobs yet
    # May need to validate if there will be any difference from recalculation
    return self.output.logprobs

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    # [FUNCTION DOC]
    # Forwards params and optional filter_types to the sampler's update_params
    # to sync new weights into vLLM. Returns None.
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
    # Returns the sampler's transformer object.
    return self._sampler.transformer
