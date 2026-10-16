# Copyright 2025 Google LLC
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
# Sampler backend that delegates generation to an in-process SGLang-JAX Engine
# while exposing the tunix BaseSampler interface, plus weight synchronization
# from the trainer into the engine's model state.
# Contents: update_hf_key_mappings_with_lora (rewrites target paths for LoRA
# wrapped layers), SglangJaxConfig (engine configuration dataclass),
# SglangJaxSampler (the sampler), and get_or_create_event_loop (asyncio
# helper). Depends on sgl_jax Engine and SUPPORTED_LORA_TARGET_MODULES, tunix
# utils.transfer_state_with_mappings for weight transfer and
# reshard.reshard_pytree for resharding.
# KNOWN ISSUE: the module imports the top-level concurrent package and then
# uses concurrent.futures, which only works if concurrent.futures has been
# imported elsewhere first.
"""Sampler for sglang-jax-style autoregressive decoding using JAX and NNX models."""

import asyncio
import concurrent
import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from absl import logging
from flax import nnx
import jax
import jaxtyping
import numpy as np
from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.utils.common_utils import SUPPORTED_LORA_TARGET_MODULES
from tunix.generate import base_sampler
from tunix.generate import mappings
from tunix.generate import utils
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.rl import reshard


def update_hf_key_mappings_with_lora(
    mappings: Optional[Dict[str, Any]] = None,
    enable_static_lora: bool = False,
    lora_target_modules: Optional[List] = None,
):
  # [FUNCTION DOC]
  # Rewrites a key-mapping dict so targets of LoRA modules point at the wrapped
  # base layer (SGLang-JAX wraps LoRA layers, so a target like
  # model.layers.N.mlp.gate_proj.weight becomes
  # model.layers.N.mlp.gate_proj.base_layer.weight).
  # Inputs: mappings, dict from source path to a (target path, sharding) tuple,
  # or None; enable_static_lora flag; lora_target_modules list of substrings.
  # Returns mappings unchanged if it is None, LoRA is disabled, or the module
  # list is empty. Otherwise for each module name, finds the first source path
  # containing it, inserts base_layer before the last dotted component of its
  # target path, stores the new tuple, and moves to the next module.
  # Side effects: mutates and returns the same dict.
  # KNOWN ISSUE: because of the break, only the first matching entry per module
  # is rewritten; all other layers/params for that module keep their original
  # target path (e.g. only one layer's gate_proj is updated).
  """Update LoRA key_mapping into hf_key_mapping."""
  if mappings is None or not enable_static_lora or not lora_target_modules:
    return mappings

  # Note: SGLangJax implements the LoRA through wraping the base_layer, so the value in mappings needs to be updated.
  # From 'model.layers.*.mlp.gate_proj.weight' to 'model.layers.*.mlp.gate_proj.base_layer.weight'
  for module in lora_target_modules:
    for src_path, tgt_params in mappings.items():
      if module in src_path:
        tgt_path, sharding = tgt_params
        keys = tgt_path.split(".")
        new_tgt_path = ".".join(keys[:-1]) + ".base_layer." + keys[-1]
        mappings[src_path] = (new_tgt_path, sharding)
        break
  return mappings


@dataclasses.dataclass
class SglangJaxConfig:
  # [CLASS DOC]
  # Configuration dataclass for SglangJaxSampler. Required: mesh (JAX device
  # mesh whose devices the engine uses), mapping_config (MappingConfig for
  # weight transfer), model_version (model path or id passed as model_path),
  # context_length (max context). Optional with defaults: mem_fraction_static
  # 0.2; init_with_random_weights True; disable_radix_cache True;
  # enable_deterministic_sampling False; use_sort_for_toppk_minp True;
  # enable_static_lora False; enable_single_process True (required on
  # Pathways); lora_target_modules, max_lora_rank, lora_scaling (None; required
  # when static LoRA is enabled; modules may be the single entry all);
  # precompile_token_paddings and precompile_bs_paddings (lists or None);
  # chunked_prefill_size -1; page_size 128; load_format auto;
  # max_running_requests None; log_level info.
  mesh: jax.sharding.Mesh
  mapping_config: mappings.MappingConfig

  model_version: str
  context_length: int

  mem_fraction_static: float = 0.2
  init_with_random_weights: bool = True
  disable_radix_cache: bool = True
  enable_deterministic_sampling: bool = False
  # Note: use_sort_for_toppk_minp may be removed in the future. It depends on SGLang-Jax.
  use_sort_for_toppk_minp: bool = True
  enable_static_lora: bool = False
  enable_single_process: bool = (
      True  # Note: this is required when you run it in pathways.
  )

  lora_target_modules: Optional[List[str]] = None
  max_lora_rank: Optional[int] = None
  lora_scaling: Optional[float] = None

  precompile_token_paddings: Optional[List[int]] = None
  precompile_bs_paddings: Optional[List[int]] = None
  chunked_prefill_size: Optional[int] = -1
  page_size: int = 128
  load_format: str = "auto"
  max_running_requests: int = None

  # Logging knob to enable debugging at different verbosity levels.
  log_level: str = "info"


class SglangJaxSampler(base_sampler.BaseSampler):  # pylint: disable=invalid-name
  # [CLASS DOC]
  # BaseSampler implementation backed by an SGLang-JAX Engine. Holds a
  # TokenizerAdapter, the engine argument dict (args), the Engine, and the
  # effective key mappings, transpose keys and hook functions used to push
  # trainer weights into the engine model. Generation is synchronous from the
  # caller's point of view (runs the engine's async generate to completion).
  """A sampler for sglang-jax-style autoregressive decoding using JAX and NNX models.

  This class wraps an NNX model and tokenizer for performing inference
  with optimized KV cache allocation based on available HBM memory.

  Inherits from:
      base_sampler.BaseSampler
  """

  def __init__(
      self,
      tokenizer: Any,
      config: SglangJaxConfig,
      **kwargs,
  ):
    # [FUNCTION DOC]
    # Wraps tokenizer in a TokenizerAdapter unless it already is one. Builds the
    # engine argument dict from config (validating it), overlays any extra
    # kwargs, and constructs the Engine (side effect: starts the engine, loads or
    # initializes the model and allocates KV cache on the mesh devices).
    # Then computes to_hf_key_mappings by passing the config's to_hf_mappings
    # through update_hf_key_mappings_with_lora using the args' LoRA flag and
    # modules; stores to_hf_transpose_keys and to_hf_hook_fns from the config;
    # merges lora_to_hf_mappings and lora_to_hf_transpose_keys into those dicts
    # when present (dict union, LoRA entries win). Logs mappings at debug level.
    # Raises AssertionError/ValueError from validation.
    # KNOWN ISSUE: the merges mutate the dicts owned by config.mapping_config in
    # place, and raise TypeError if the base mappings or transpose keys are None
    # while the LoRA ones are set. to_hf_hook_fns is stored but never used.
    """Initializes the SglangJaxSampler.

    Args:
        tokenizer (Any): A tokenizer compatible with the model.
        config: The sglang-jax related configurations
    """
    self.tokenizer = tokenizer
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
      self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.args = self._sglang_jax_config(config)
    if kwargs:
      self.args.update(kwargs)
    self.engine = Engine(**self.args)

    self.to_hf_key_mappings = update_hf_key_mappings_with_lora(
        config.mapping_config.to_hf_mappings,
        self.args["enable_static_lora"],
        self.args["lora_target_modules"],
    )
    self.to_hf_transpose_keys = config.mapping_config.to_hf_transpose_keys
    self.to_hf_hook_fns = config.mapping_config.to_hf_hook_fns

    if config.mapping_config.lora_to_hf_mappings:
      self.to_hf_key_mappings |= config.mapping_config.lora_to_hf_mappings

    if config.mapping_config.lora_to_hf_transpose_keys:
      self.to_hf_transpose_keys |= (
          config.mapping_config.lora_to_hf_transpose_keys
      )

    logging.debug(f"{self.to_hf_key_mappings=}")

  # TODO(b/434969743): Optimize weight sharing between trainer and sglang-jax sampler.
  # TODO(b/434975493): Consider Release KV cache on the fly
  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    # [FUNCTION DOC]
    # Pushes new trainer weights into the engine. filter_types is ignored.
    # Calls utils.transfer_state_with_mappings with the updated weights as source,
    # the current transformer_state as destination, the key mappings and transpose
    # keys, reshard.reshard_pytree as the reshard function, rollout engine name
    # sglang_jax, and the model's total KV head count and head dim (None if no
    # model runner). Flattens the resulting state to leaves and assigns them to
    # the model runner's model_state_leaves (side effect: engine now uses new
    # weights).
    # KNOWN ISSUE: transformer_state and the final assignment dereference the
    # model runner unconditionally, so the None fallbacks are moot; without a
    # local scheduler (e.g. multi-process mode) this raises AttributeError.
    del filter_types
    new_state = utils.transfer_state_with_mappings(
        src_state=updated_weights,
        dst_state=self.transformer_state,
        key_mappings=self.to_hf_key_mappings,
        transpose_keys=self.to_hf_transpose_keys,
        reshard_fn=reshard.reshard_pytree,
        rollout_engine="sglang_jax",
        num_kv_heads=(
            None
            if not self._model_runner
            else self._model_runner.model_config.get_total_num_kv_heads()
        ),
        head_dim=(
            None
            if not self._model_runner
            else self._model_runner.model_config.head_dim
        ),
    )
    new_model_state_leaves, _ = jax.tree_util.tree_flatten(new_state)
    self._model_runner.model_state_leaves = new_model_state_leaves

  def load_checkpoint(self, path_or_weights: str | jaxtyping.PyTree):
    # [FUNCTION DOC]
    # Loads weights: if the argument is a PyTree (any isinstance check against
    # jaxtyping.PyTree), forwards to update_params; otherwise raises
    # NotImplementedError (path-based checkpoint loading unsupported).
    # KNOWN ISSUE: jaxtyping.PyTree accepts essentially any object including a
    # string, so the NotImplementedError branch is effectively unreachable and a
    # path string would be passed to update_params.
    # TODO(b/434741253): Consider support orbax checkpoint loading
    if isinstance(path_or_weights, jaxtyping.PyTree):
      self.update_params(updated_weights=path_or_weights, filter_types=None)
    else:
      raise NotImplementedError("Only support in memory weight sync as of now.")

  def _find_tp_size(self, mesh: jax.sharding.Mesh) -> int:
    # [FUNCTION DOC]
    # HELPER: returns the product of all mesh axis sizes (total device count),
    # used as the tensor-parallel size since data parallelism is unsupported.
    """Finds the tensor parallel size from the mesh."""
    # since sglang-jax doesn't support DP yet, simply return the total rank size.
    return math.prod(mesh.shape.values())

  def _sglang_jax_config(self, config: SglangJaxConfig):
    # [FUNCTION DOC]
    # HELPER: validates config then builds the Engine keyword dict: model_path,
    # context_length, mem_fraction_static, disable_radix_cache,
    # enable_deterministic_sampling, use_sort_for_toppk_minp, enable_static_lora,
    # enable_single_process, lora_target_modules (the full supported list when
    # config says all), max_lora_rank, max_loras_per_batch fixed at 1,
    # lora_scaling, precompile paddings, chunked_prefill_size, page_size, tp_size
    # (device count), device_indexes (flattened mesh device ids as a list),
    # load_format, max_running_requests, enable_engine_loop_run_forever_daemon
    # True, and log_level. Returns the dict.
    # KNOWN ISSUE: load_format is first set to dummy when init_with_random_weights
    # is True but then unconditionally overwritten by config.load_format, so
    # init_with_random_weights has no effect.
    self._validate_config(config)

    args = {}
    args["model_path"] = config.model_version
    args["context_length"] = config.context_length
    args["mem_fraction_static"] = config.mem_fraction_static
    if config.init_with_random_weights:
      args["load_format"] = "dummy"
    args["disable_radix_cache"] = config.disable_radix_cache
    args["enable_deterministic_sampling"] = config.enable_deterministic_sampling
    args["use_sort_for_toppk_minp"] = config.use_sort_for_toppk_minp
    args["enable_static_lora"] = config.enable_static_lora
    args["enable_single_process"] = config.enable_single_process
    if config.lora_target_modules == ["all"]:
      args["lora_target_modules"] = SUPPORTED_LORA_TARGET_MODULES
    else:
      args["lora_target_modules"] = config.lora_target_modules
    args["max_lora_rank"] = config.max_lora_rank
    args["max_loras_per_batch"] = 1
    args["lora_scaling"] = config.lora_scaling
    args["precompile_token_paddings"] = config.precompile_token_paddings
    args["precompile_bs_paddings"] = config.precompile_bs_paddings
    args["chunked_prefill_size"] = config.chunked_prefill_size
    args["page_size"] = config.page_size
    args["tp_size"] = self._find_tp_size(config.mesh)
    args["device_indexes"] = config.mesh.device_ids.flatten().tolist()
    args["load_format"] = config.load_format
    args["max_running_requests"] = config.max_running_requests
    args["enable_engine_loop_run_forever_daemon"] = True

    args["log_level"] = config.log_level
    return args

  def _validate_config(self, config: SglangJaxConfig):
    # [FUNCTION DOC]
    # HELPER: asserts precompile_token_paddings and precompile_bs_paddings are
    # lists when given. If static LoRA is enabled, asserts lora_target_modules,
    # max_lora_rank and lora_scaling are all set, and unless modules is the single
    # entry all, raises ValueError for any module not in
    # SUPPORTED_LORA_TARGET_MODULES. Raises AssertionError on assertion failures.
    if config.precompile_token_paddings is not None:
      assert isinstance(config.precompile_token_paddings, List)

    if config.precompile_bs_paddings is not None:
      assert isinstance(config.precompile_bs_paddings, List)

    if config.enable_static_lora:
      assert (
          config.lora_target_modules is not None
          and config.max_lora_rank is not None
          and config.lora_scaling is not None
      )
      # check whether the lora_target_modules are valid
      if config.lora_target_modules != ["all"]:
        for module in config.lora_target_modules:
          if module not in SUPPORTED_LORA_TARGET_MODULES:
            raise ValueError(
                f"{module} in lora_target_modules does not exist in"
                f" {SUPPORTED_LORA_TARGET_MODULES}"
            )

  @property
  def _model_runner(self):
    # [FUNCTION DOC]
    # HELPER property: returns the SGLang model runner reached via the engine's
    # scheduler_info scheduler entry (scheduler, tp_worker, worker, model_runner)
    # when the scheduler lives in-process, else None.
    if "scheduler" in self.engine.scheduler_info:
      return self.engine.scheduler_info[
          "scheduler"
      ].tp_worker.worker.model_runner
    else:
      return None

  @property
  def mesh(self) -> jax.sharding.Mesh:
    # [FUNCTION DOC]
    # Property returning the model runner's mesh if it has one that is a
    # jax.sharding.Mesh; otherwise raises AttributeError.
    if hasattr(self._model_runner, "mesh") and isinstance(
        self._model_runner.mesh, jax.sharding.Mesh
    ):
      return self._model_runner.mesh
    else:
      raise AttributeError(
          "SGLang model runner doesn't have mesh or mesh is not a"
          " jax.sharding.Mesh."
      )

  @property
  def transformer(self):
    # [FUNCTION DOC]
    # Property; always returns None because SGLang-JAX does not expose the model
    # module.
    # sglang-jax doesn't expose the underlying model
    return None

  @property
  def transformer_state(self):
    # [FUNCTION DOC]
    # Property returning the nnx State of the model runner's model (second element
    # of nnx.split). Raises AttributeError if there is no model runner.
    return nnx.split(self._model_runner.model)[1]

  def tokenize(self, input_string: str) -> jax.Array | list[int]:
    # [FUNCTION DOC]
    # Encodes the string with the adapter, prepends the BOS id if it is truthy,
    # and removes duplicated leading BOS ids. Returns a list of ints.
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    return self.tokenizer.dedup_bos_ids(bos_tok + input_ids)

  def __call__(
      self,
      input_strings: str | List[str],
      max_generation_steps: int,
      max_prompt_length: int | None = None,
      temperature: float = 0.0,
      top_p: float | None = None,
      top_k: int | None = None,
      beam_size: int | None = None,
      seed: Optional[Union[List[int], int]] = None,
      multi_sampling: int = 1,
      return_logits: bool = True,
      echo: bool = False,
      pad_output: bool = False,
      **kwargs,
  ) -> base_sampler.SamplerOutput:
    # [FUNCTION DOC]
    # Generates completions for one or more prompts.
    # Validation: raises ValueError if context_length is set and
    # max_generation_steps exceeds it.
    # Builds sampling params from the engine defaults: max_new_tokens,
    # n = multi_sampling, temperature, stop_token_ids set to the EOS id,
    # skip_special_tokens True, plus top_p/top_k when given. Extra kwargs are
    # applied via the params' update method; failures are logged (once) and
    # ignored. The params are stored on self.sampling_params (side effect) and
    # converted to one dict per prompt. If seed is given, sets sampling_seed per
    # prompt (intended: element-wise for a list, same value otherwise).
    # Tokenizes each prompt and runs generation through the loop guard. Pads each
    # prompt to max_prompt_length on the left with the pad id, where
    # max_prompt_length is replaced by the next power of two of the longest prompt
    # when missing or too small. Returns a SamplerOutput with texts and int32
    # output id arrays from the engine outputs, logits None, logprobs None, and
    # the padded prompt array.
    # Ignored inputs: beam_size, return_logits, echo, pad_output.
    # KNOWN ISSUE: the list check compares the type to typing.List with identity,
    # which is never true, so a list seed is assigned whole to every prompt
    # instead of element-wise and the length assertion never runs.
    # KNOWN ISSUE: with multi_sampling greater than 1 the engine returns more
    # outputs than prompts, but padded_prompt_tokens has one row per prompt.
    # KNOWN ISSUE: the context check ignores prompt length.
    # max_generation_steps: maximum number of tokens to generate
    if (
        self.args["context_length"] is not None
        and max_generation_steps > self.args["context_length"]
    ):
      raise ValueError(
          "`max_generation_steps` must be less than or equal to "
          "`context_length`. Received:  `max_generation_steps`="
          f"{max_generation_steps} and `max_model_len`="
          f"{self.args['context_length']}."
      )

    if isinstance(input_strings, str):
      input_strings = [input_strings]

    self.sampling_params = self.engine.get_default_sampling_params()
    self.sampling_params.max_new_tokens = max_generation_steps
    self.sampling_params.n = multi_sampling
    self.sampling_params.temperature = temperature
    self.sampling_params.stop_token_ids = [self.tokenizer.eos_id()]
    self.sampling_params.skip_special_tokens = True

    if top_p is not None:
      self.sampling_params.top_p = top_p
    if top_k is not None:
      self.sampling_params.top_k = top_k

    if kwargs:
      try:
        self.sampling_params.update(**kwargs)
        logging.log_first_n(
            logging.INFO,
            "Received additional kwargs that are not explicitly defined in the"
            f" method signature: {kwargs}. These will be forwarded to the"
            " underlying sampler, but please ensure that they are valid.",
            1,
        )
      except Exception as e:
        logging.log_first_n(
            logging.INFO,
            f"Failed to update sampling_params with kwargs: {kwargs}."
            f" Error: {e}",
            1,
        )

    sampling_params = [
        self.sampling_params.convert_to_dict() for _ in input_strings
    ]
    if seed is not None:
      if type(seed) is List:
        assert len(seed) == len(
            input_strings
        ), "seed and input_strings must have same length"
        for i, seed_i in enumerate(seed):
          sampling_params[i]["sampling_seed"] = seed_i
      else:
        for i, _ in enumerate(input_strings):
          sampling_params[i]["sampling_seed"] = seed

    prompt_ids = [self.tokenize(x) for x in input_strings]
    outputs = self._generate_with_loop_guard(
        input_ids=[ids for ids in prompt_ids],
        sampling_params=sampling_params,
    )

    max_tokens_length = max(len(x) for x in prompt_ids)

    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = utils.next_power_of_2(max_tokens_length)
    all_input_ids = [
        utils.pad_to_length(
            np.array(x, dtype=np.int32),
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for x in prompt_ids
    ]
    all_input_ids = np.array(all_input_ids, dtype=np.int32)

    all_output_ids = [
        np.array(x["output_ids"], dtype=np.int32) for x in outputs
    ]
    output_texts = [o["text"] for o in outputs]
    # To support multisampling, just return the whole list of SamplerOutput
    return base_sampler.SamplerOutput(
        text=output_texts,
        logits=None,
        tokens=all_output_ids,
        padded_prompt_tokens=all_input_ids,
        logprobs=None,
    )

  def _generate_with_loop_guard(
      self,
      *,
      input_ids: List[List[int]],
      sampling_params: List[dict],
  ):
    # [FUNCTION DOC]
    # HELPER: runs the engine's async_generate (non-streaming) coroutine to
    # completion synchronously. Gets an event loop via get_or_create_event_loop.
    # If that loop is not running, runs the coroutine on it, then cancels all
    # remaining tasks, drains them and closes the loop. If a loop is already
    # running in this thread, runs a nested helper in a new worker thread
    # (ThreadPoolExecutor) that does the same on that thread's own new loop, and
    # blocks for its result. Returns the engine's list of output dicts (with
    # output_ids and text). Exceptions from generation propagate.
    # KNOWN ISSUE: the loop it closes was installed as the thread's current event
    # loop, leaving a closed loop set for later asyncio use in that thread.
    coro = self.engine.async_generate(
        input_ids=input_ids,
        sampling_params=sampling_params,
        stream=False,
    )
    loop = get_or_create_event_loop()

    if not loop.is_running():
      try:
        return loop.run_until_complete(coro)
      finally:
        # Cancel any pending tasks before closing
        pending = asyncio.all_tasks(loop)
        for task in pending:
          task.cancel()
        # Run the loop one more time to process cancellations
        loop.run_until_complete(
            asyncio.gather(*pending, return_exceptions=True)
        )
        loop.close()

    def wrap_generate():
      # [FUNCTION DOC]
      # HELPER: worker-thread body; obtains a fresh event loop for the worker
      # thread, runs the captured coroutine to completion, then cancels and drains
      # leftover tasks and closes the loop. Returns the generation result.
      loop = get_or_create_event_loop()
      try:
        return loop.run_until_complete(coro)
      finally:
        # Cancel any pending tasks before closing
        pending = asyncio.all_tasks(loop)
        for task in pending:
          task.cancel()
        # Run the loop one more time to process cancellations
        loop.run_until_complete(
            asyncio.gather(*pending, return_exceptions=True)
        )
        loop.close()

    with concurrent.futures.ThreadPoolExecutor() as executor:
      future = executor.submit(wrap_generate)
      return future.result()


def get_or_create_event_loop():
# [FUNCTION DOC]
# HELPER: returns the currently running asyncio loop of this thread; if none
# is running, creates a new event loop, sets it as the current loop (side
# effect) and returns it.
# KNOWN ISSUE: the return inside the finally clause would swallow any
# unexpected exception raised in the try or except blocks.
  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
  finally:
    return loop
