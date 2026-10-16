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
# Sampler backend that runs generation on vLLM (TPU JAX backend, a.k.a.
# tpu-inference) inside the current process and exposes the tunix BaseSampler
# interface, plus in-place synchronization of trainer weights into the vLLM
# model state.
# Two execution modes: offline mode uses vllm.LLM with batched generate;
# server mode uses VLLMInProcessDriver (continuous batching on a background
# thread) and submits one request per prompt.
# Contents: VllmConfig (configuration dataclass) and VllmSampler.
# Import-time side effect: sets environment variable
# VLLM_ENABLE_V1_MULTIPROCESSING to 0 so engine and worker share this process
# (also importing vllm_async_driver applies the same setting).
"""Sampler for vLLM-style autoregressive decoding using JAX and NNX models."""

import atexit
import dataclasses
import gc
from itertools import count
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from absl import logging
from flax import nnx
import jax
import jaxtyping
import numpy as np
from tunix.generate import base_sampler
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.generate import utils
from tunix.generate.mappings import MappingConfig
from tunix.generate.vllm_async_driver import VLLMInProcessDriver
from tunix.rl import reshard
from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.sampling_params import SamplingParams

# Colocate vllm engine and worker in the main process
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


@dataclasses.dataclass
class VllmConfig:
  """Vllm rollout configuations."""
  # [CLASS DOC]
  # Mutable dataclass of vLLM rollout settings.
  # Sampler fields: server_mode (use the in-process driver), mapping_config
  # (MappingConfig, default empty), return_logprobs.
  # Environment fields: init_with_random_weights (load dummy weights since real
  # weights are synced later), tpu_backend_type (default jax).
  # Processed engine fields: additional_config dict, enable_dp_attention,
  # hbm_utilization (maps to gpu_memory_utilization, default 0.5), lora_config,
  # mesh (device mesh, required in practice), data/tensor parallel sizes (-1
  # means infer from mesh), expert_parallel_size (1), delete_dst_buffers (True),
  # reshard_chunk_size.
  # engine_kwargs is an init-only pass-through dict of raw vLLM EngineArgs
  # stored as _processed_engine_kwargs. _RESERVED_KEYS (not an init field) lists
  # tensor_parallel_size, data_parallel_size and expert_parallel_size, which may
  # not appear in engine_kwargs. sampling_kwargs is a dict of default extra
  # SamplingParams attributes.

  # Sampler related
  server_mode: bool = False
  mapping_config: MappingConfig = dataclasses.field(
      default_factory=MappingConfig
  )
  return_logprobs: bool = False

  # vLLM Env vars
  init_with_random_weights: bool = True
  tpu_backend_type: str = "jax"

  # vLLM engine arg related, requires additional processing before passing into engine
  additional_config: Optional[Dict[str, Any]] = None
  enable_dp_attention: bool = False
  hbm_utilization: float = 0.5
  lora_config: Optional[Dict[str, Any]] = None
  mesh: jax.sharding.Mesh = None
  data_parallel_size: int = -1
  tensor_parallel_size: int = -1
  expert_parallel_size: int = 1
  # Default to True to ensure old weights are deleted to free up HBM memory
  delete_dst_buffers: bool = True
  reshard_chunk_size: Optional[int] = None

  # vLLM engine args that can be directly passed in without additional processing, e.g. max_model_len, async_scheduling, etc.
  engine_kwargs: dataclasses.InitVar[Optional[Dict[str, Any]]] = None
  _processed_engine_kwargs: Dict[str, Any] = dataclasses.field(
      init=False, default_factory=dict
  )

  # VllmConfig fields that require special processing before being passed to
  # vLLM and must not be passed via engine_kwargs, which is a raw pass-through
  # to vLLM EngineArgs.
  _RESERVED_KEYS: frozenset[str] = dataclasses.field(
      default=frozenset(
          {"tensor_parallel_size", "data_parallel_size", "expert_parallel_size"}
      ),
      init=False,
      repr=False,
      compare=False,
  )
  # vLLM sampling args that can be directly passed in without additional processing, e.g. temperature, stop etc.
  sampling_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self, engine_kwargs: Optional[Dict[str, Any]]):
    # [FUNCTION DOC]
    # Dataclass post-init hook receiving the init-only engine_kwargs (None means
    # empty). Raises ValueError listing any reserved keys present. Stores the dict
    # as _processed_engine_kwargs, then for each item logs it and sets an
    # attribute of that name on the config.
    # KNOWN ISSUE: every engine kwarg is also set as an attribute on the config,
    # which can silently overwrite a same-named config field (e.g.
    # hbm_utilization) or add arbitrary new attributes.
    engine_kwargs = engine_kwargs or {}
    illegal = self._RESERVED_KEYS & engine_kwargs.keys()
    if illegal:
      raise ValueError(
          "VllmConfig fields must be set directly on VllmConfig, not passed"
          f" via engine_kwargs: {sorted(illegal)}"
      )
    self._processed_engine_kwargs = engine_kwargs
    if engine_kwargs:
      for key, value in engine_kwargs.items():
        logging.info(
            "Engine kwargs setting key '%s' with value '%s'.", key, value
        )
        setattr(self, key, value)


class VllmSampler(base_sampler.BaseSampler):  # pylint: disable=invalid-name
  # [CLASS DOC]
  # BaseSampler implementation backed by vLLM. Holds a TokenizerAdapter, the
  # config, the engine argument dict (args), either an LLM (offline mode) or a
  # VLLMInProcessDriver (server mode), a request id counter, and the effective
  # weight key mappings, transpose keys and hook functions.
  """A sampler for vLLM-style autoregressive decoding using JAX and NNX models.

  This class wraps an NNX model and tokenizer for performing inference
  with optimized KV cache allocation based on available HBM memory.

  Inherits from:
      base_sampler.BaseSampler
  """

  def __init__(
      self,
      tokenizer: Any,
      config: VllmConfig,
  ):
    # [FUNCTION DOC]
    # Side effects on process environment: sets TPU_BACKEND_TYPE to the config's
    # backend type when non-empty; sets NEW_MODEL_DESIGN to 1 when
    # data_parallel_size is greater than 1; sets JAX_RANDOM_WEIGHTS to 1 when
    # init_with_random_weights.
    # Wraps the tokenizer in a TokenizerAdapter if needed, stores config, builds
    # args via the config translator, and creates the engine: in server mode a
    # driver (and registers stop with atexit), otherwise an LLM built from args
    # (loads/initializes the model and KV cache on device).
    # Copies the config's to_hf_mappings into a new dict (empty if None), stores
    # transpose keys and hook fns, and if lora_config and LoRA mappings are both
    # set, merges the LoRA mappings into the key mappings.
    """Initializes the VllmSampler.

    Args:
        tokenizer (Any): A tokenizer compatible with the model.
        config: The vllm related configurations
    """

    # Select vllm TPU backend type, there are jax, torchax and torchxla
    if config.tpu_backend_type:
      os.environ["TPU_BACKEND_TYPE"] = config.tpu_backend_type

    # vLLM DP only works with the new model design
    if config.data_parallel_size > 1:
      os.environ["NEW_MODEL_DESIGN"] = "1"

    # tpu-inference backend recently removed this environment variable, however
    # still set it here for backward compatibility.
    if config.init_with_random_weights:
      os.environ["JAX_RANDOM_WEIGHTS"] = "1"

    self.tokenizer = tokenizer
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
      self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.config = config
    self.args = self._vllm_config(config)
    self._driver: VLLMInProcessDriver | None = None
    self.llm: LLM | None = None
    self._request_counter = count()

    if config.server_mode:
      self._driver = self._create_driver()
      atexit.register(self.stop)
    else:
      self.llm = LLM(**self.args)

    self.to_hf_key_mappings = dict(config.mapping_config.to_hf_mappings or {})
    self.to_hf_transpose_keys = config.mapping_config.to_hf_transpose_keys
    self.to_hf_hook_fns = config.mapping_config.to_hf_hook_fns

    # TODO(b/434959964) It's not taking effect until vLLM Jax backend support
    # lora.
    if config.lora_config and config.mapping_config.lora_to_hf_mappings:
      self.to_hf_key_mappings |= config.mapping_config.lora_to_hf_mappings

  @property
  def mesh(self) -> jax.sharding.Mesh:
    # [FUNCTION DOC]
    # Property returning the model runner's mesh if it is a jax.sharding.Mesh,
    # otherwise raises AttributeError. Raises RuntimeError if no engine exists.
    if hasattr(self._model_runner, "mesh") and isinstance(
        self._model_runner.mesh, jax.sharding.Mesh
    ):
      return self._model_runner.mesh
    else:
      raise AttributeError(
          "vLLM model runner doesn't have mesh or mesh is not a"
          " jax.sharding.Mesh."
      )

  # TODO(b/434969743): Optimize weight sharing between trainer and vllm sampler.
  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    # [FUNCTION DOC]
    # Synchronizes trainer weights into the vLLM model. filter_types is ignored.
    # Steps:
    # 1. Reset the prefix cache and call collective_rpc delete_kv_cache on the LLM
    #    or the driver's engine to free HBM.
    # 2. jax.effects_barrier to finish outstanding device work.
    # 3. If key mappings exist: optionally run the config's preprocess_src_state
    #    on the weights, then utils.transfer_state_with_mappings into the current
    #    transformer_state with key mappings, hook fns, transpose keys,
    #    reshard.reshard_pytree, delete_dst_buffers and reshard_chunk_size from
    #    config, KV head count and head size from the model config, and tp_size
    #    from args (default 1).
    #    Otherwise (direct sync, e.g. MaxText to MaxText) require a maxtext_config
    #    key in additional_config (ValueError if missing) and call
    #    utils.transfer_state_directly with destination buffers deleted.
    # 4. If the model runner has a state_leaves attribute, refresh it from the
    #    runner state (tuple of leaves for an nnx.State, else the state itself).
    # 5. collective_rpc reinitialize_kv_cache to reallocate the KV cache.
    # Raises RuntimeError if no engine is initialized.
    del filter_types

    if self.llm is not None:
      self.llm.reset_prefix_cache()
      self.llm.collective_rpc("delete_kv_cache") # will free hbm
    elif self._driver is not None:
      self._driver.llm_engine.reset_prefix_cache()
      self._driver.llm_engine.collective_rpc("delete_kv_cache")

    # Synchronization point before weight sync
    jax.effects_barrier()

    if self.to_hf_key_mappings:
      preprocess_fn = self.config.mapping_config.preprocess_src_state
      if preprocess_fn:
        updated_weights = preprocess_fn(updated_weights)

      utils.transfer_state_with_mappings(
          src_state=updated_weights,
          dst_state=self.transformer_state,
          key_mappings=self.to_hf_key_mappings,
          key_mapping_hook_fns=self.to_hf_hook_fns,
          transpose_keys=self.to_hf_transpose_keys,
          reshard_fn=reshard.reshard_pytree,
          delete_dst_buffers=self.config.delete_dst_buffers,
          reshard_chunk_size=self.config.reshard_chunk_size,
          num_kv_heads=(
              None
              if not self._model_runner
              else self._model_runner.model_config.get_total_num_kv_heads()
          ),
          head_dim=(
              None
              if not self._model_runner
              else self._model_runner.model_config.get_head_size()
          ),
          tp_size=self.args.get("tensor_parallel_size", 1),
      )
    else:
      # Direct Weight Sync (e.g. MaxText -> MaxText)
      logging.debug(
          "No key mappings configuration found. Proceeding with direct"
          " structural weight synchronization (assuming matching source/target"
          " structures)."
      )

      additional_config = self.config.additional_config or {}
      if "maxtext_config" not in additional_config:
        raise ValueError(
            "Direct weight synchronization is currently supported only for "
            "MaxText models. The required 'maxtext_config' key is missing "
            "from 'additional_config'."
        )

      utils.transfer_state_directly(
          src_state=updated_weights,
          dst_state=self.transformer_state,
          reshard_fn=reshard.reshard_pytree,
          delete_dst_buffers=True,  # Ensure old weights are deleted to free up HBM memory
          reshard_chunk_size=self.config.reshard_chunk_size,
      )

    if hasattr(self._model_runner, "state_leaves"):
      if isinstance(self._model_runner.state, nnx.State):
        self._model_runner.state_leaves = tuple(
            jax.tree_util.tree_leaves(self._model_runner.state)
        )
      else:
        self._model_runner.state_leaves = self._model_runner.state

    if self.llm is not None:
      self.llm.collective_rpc("reinitialize_kv_cache")
    elif self._driver is not None:
      self._driver.llm_engine.collective_rpc("reinitialize_kv_cache")

  def load_checkpoint(self, path_or_weights: str | jaxtyping.PyTree):
    # [FUNCTION DOC]
    # If the argument is a PyTree, forwards to update_params; otherwise raises
    # NotImplementedError.
    # KNOWN ISSUE: jaxtyping.PyTree accepts essentially any object (including a
    # path string), so the NotImplementedError branch is effectively unreachable.
    # TODO(b/434741253): Consider support orbax checkpoint loading
    if isinstance(path_or_weights, jaxtyping.PyTree):
      self.update_params(updated_weights=path_or_weights, filter_types=None)
    else:
      raise NotImplementedError("Only support in memory weight sync as of now.")

  def _vllm_config(self, config: VllmConfig):
    # [FUNCTION DOC]
    # HELPER: translates VllmConfig into keyword arguments for vLLM LLM or
    # EngineArgs. Starts from a copy of the processed engine kwargs; sets
    # load_format dummy when init_with_random_weights; gpu_memory_utilization from
    # hbm_utilization; additional_config from config (or a new empty dict) with
    # lora_config added when set; resolves tensor, data and expert parallel sizes
    # via utils.resolve_parallelism_sizes using the mesh; sets
    # tensor_parallel_size and data_parallel_size; and sets
    # additional_config.sharding.sharding_strategy with expert_parallelism, the
    # flattened mesh device id list and enable_dp_attention. Returns the dict.
    # Raises AttributeError if mesh is None; propagates errors from parallelism
    # resolution.
    # KNOWN ISSUE: when config.additional_config is provided, the same dict is
    # mutated in place (lora_config and sharding keys are added to the user's
    # config).
    """Setup vllm config from Tunix Vllm config."""
    args = config._processed_engine_kwargs.copy()

    # Init vLLM model with random weights to speed up bootstrap time, because
    # model weights are synced from trainer later on
    if config.init_with_random_weights:
      args["load_format"] = "dummy"

    args["gpu_memory_utilization"] = config.hbm_utilization

    args["additional_config"] = config.additional_config or {}

    if config.lora_config is not None:
      args["additional_config"]["lora_config"] = config.lora_config

    tp, dp, ep = utils.resolve_parallelism_sizes(
        mesh=config.mesh,
        tensor_parallel_size=config.tensor_parallel_size,
        data_parallel_size=config.data_parallel_size,
        expert_parallel_size=config.expert_parallel_size,
    )
    args["tensor_parallel_size"] = tp
    args["data_parallel_size"] = dp

    device_indexes = config.mesh.device_ids.flatten().tolist()
    args["additional_config"]["sharding"] = {
        "sharding_strategy": {
            "expert_parallelism": ep,
            "device_indexes": device_indexes,
            "enable_dp_attention": config.enable_dp_attention,
        }
    }

    return args

  def _build_engine_args(self) -> EngineArgs:
    # [FUNCTION DOC]
    # HELPER: copies args, defaults disable_log_stats to True, and returns an
    # EngineArgs built from them (TypeError on unknown keys).
    engine_kwargs = dict(self.args)
    engine_kwargs.setdefault("disable_log_stats", True)
    return EngineArgs(**engine_kwargs)

  def _create_driver(self) -> VLLMInProcessDriver:
    # [FUNCTION DOC]
    # HELPER: builds EngineArgs and returns VLLMInProcessDriver.from_engine_args
    # (auto-started, which spawns its threads).
    engine_args = self._build_engine_args()
    return VLLMInProcessDriver.from_engine_args(
        engine_args,
    )

  def stop(self):
    # [FUNCTION DOC]
    # Shuts down the server-mode driver if present (fails pending requests, stops
    # threads, shuts down the engine core) and clears it. No-op in offline mode
    # or when already stopped; registered with atexit in server mode.
    logging.debug("Shutting down VLLMInProcessDriver.")
    if self._driver is not None:
      self._driver.shutdown()
      self._driver = None

  @property
  def _model_runner(self):
    # [FUNCTION DOC]
    # HELPER property: returns the vLLM model runner from the LLM's or driver's
    # engine, following model_executor, driver_worker, model_runner. Raises
    # RuntimeError when no engine is initialized.
    if self.llm is not None:
      return self.llm.llm_engine.model_executor.driver_worker.model_runner
    if self._driver is not None:
      return self._driver.llm_engine.model_executor.driver_worker.model_runner
    raise RuntimeError("vLLM engine is not initialized.")

  @property
  def transformer(self):
    # [FUNCTION DOC]
    # Property; always returns None because vLLM does not expose the model module.
    # vLLM doesn't expose the underlying model
    return None

  @property
  def transformer_state(self):
    # [FUNCTION DOC]
    # Property returning the model runner's state (parameter pytree). Raises
    # AttributeError if the runner has no state attribute.
    if hasattr(self._model_runner, "state"):
      return self._model_runner.state
    else:
      raise AttributeError("vLLM model runner doesn't have state.")

  def tokenize(self, input_string: str) -> jax.Array | list[int]:
    # [FUNCTION DOC]
    # Encodes the string, prepends BOS if the BOS id is truthy, and removes
    # duplicated leading BOS ids. Returns a list of ints.
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    return self.tokenizer.dedup_bos_ids(bos_tok + input_ids)

  def detokenize(
      self, input_strings: List[str], request_outputs: List[RequestOutput]
  ) -> Tuple[List[str], List[float], List[int]]:
    # [FUNCTION DOC]
    # Converts vLLM RequestOutputs into per-sample lists. The number of
    # generations is taken from the first output's completion count. Returns
    # three lists indexed first by sample index then by prompt: decoded strings
    # (tokenizer decode of the full token ids), log-probability lists from
    # utils.get_logprobs_from_vllm_output, and int32 numpy token id arrays. Token
    # ids keep the trailing EOS/stop token on purpose (needed for exact multi-turn
    # reconstruction). Logs prompt and text at debug level.
    # Raises IndexError for an empty output list.
    """Detokenize the vllm outputs."""
    generations = len(request_outputs[0].outputs)
    decoded_outputs = [[] for _ in range(generations)]
    out_logprobs = [[] for _ in range(generations)]
    out_tokens = [[] for _ in range(generations)]
    for input_string, multi_sampling_output in zip(
        input_strings, request_outputs
    ):
      for idx, single_output in enumerate(multi_sampling_output.outputs):
        # KEEP the eos token in the returned token_ids — needed so multi-turn
        # consumers (agentic engine) can reconstruct the exact sequence the
        # next turn's prompt was rendered from. Combined with
        # `include_stop_str_in_output=True`, vLLM emits one eos at the end of
        # each generation. Stripping it (the previous behavior) made
        # trainer-side concatenation miss `<|im_end|>` at every turn boundary
        # and produced 30+ nat sampler-trainer logp diffs.

        out_tokens[idx].append(
            np.array(single_output.token_ids, dtype=np.int32)
        )
        decoded_outputs[idx].append(
            self.tokenizer.decode(single_output.token_ids)
        )
        logprobs = utils.get_logprobs_from_vllm_output(
            single_output.token_ids, single_output.logprobs
        )
        out_logprobs[idx].append(logprobs)
        logging.debug(
            "Prompt: %r\n\nGenerated text: %r\n\n ",
            input_string,
            decoded_outputs[idx][-1],
        )
    return decoded_outputs, out_logprobs, out_tokens

  def _generate_server_mode(
      self,
      prompts: List[TokensPrompt],
      sampling_params: Union[SamplingParams, BeamSearchParams],
  ) -> List[RequestOutput]:
    # [FUNCTION DOC]
    # HELPER: server-mode generation. Raises RuntimeError if no driver. Submits
    # each prompt with a fresh numeric request id from the counter, cloning the
    # params for prompts after the first when clone is available, then waits for
    # each Future in order and returns the outputs. Raises TypeError if a result
    # is not a RequestOutput; Future exceptions (driver errors, shutdown)
    # propagate.
    """Generate the response in server mode."""
    if self._driver is None:
      raise RuntimeError("vLLM in-process driver is not initialized.")

    futures = []
    for idx, prompt in enumerate(prompts):
      request_id = str(next(self._request_counter))
      params = sampling_params
      if idx > 0 and hasattr(sampling_params, "clone"):
        params = sampling_params.clone()
      future = self._driver.submit_request(
          request_id=request_id,
          prompt=prompt,
          params=params,
      )
      futures.append(future)

    outputs: List[RequestOutput] = []
    for future in futures:
      result = future.result()
      if not isinstance(result, RequestOutput):
        raise TypeError(
            f"Expected RequestOutput from driver, received {type(result)}."
        )
      outputs.append(result)
    return outputs

  def __call__(
      self,
      input_strings: str | List[str],
      max_generation_steps: int,
      max_prompt_length: int = None,
      temperature: float = 0.0,
      top_p: float = None,
      top_k: int = None,
      beam_size: int = None,
      seed: int = None,  # vLLM Jax backend doesn't support per request seed.
      multi_sampling: int = 1,
      return_logits: bool = True,
      echo: bool = False,
      pad_output: bool = False,
      **kwargs,
  ) -> base_sampler.SamplerOutput:
    # [FUNCTION DOC]
    # Main generation entry point. Normalizes a single string to a list. Raises
    # ValueError if max_generation_steps exceeds args max_model_len.
    # Beam path (beam_size given): uses BeamSearchParams with beam width, max
    # tokens, ignore_eos False and temperature.
    # Sampling path: starts from model default params (server mode: the model
    # config's diff sampling params via SamplingParams.from_optional, or plain
    # SamplingParams; offline: the LLM's defaults), then sets detokenize False,
    # max_tokens, n = multi_sampling, temperature, logprobs and prompt_logprobs to
    # 1 if return_logprobs else 0, stop_token_ids to EOS, skip_special_tokens
    # True, include_stop_str_in_output True, and top_p, top_k, seed when given.
    # Then applies config.sampling_kwargs overlaid with call kwargs as attributes
    # (logged; AttributeError/TypeError logged and ignored).
    # Stores the params on self.sampling_params (side effect). Tokenizes prompts,
    # generates via the driver or LLM.generate (with a progress bar), detokenizes,
    # raises ValueError if logprobs were requested but missing, and left-pads
    # prompts with the pad id to max_prompt_length (bumped to the next power of
    # two of the longest prompt when missing or too small).
    # Returns SamplerOutput with the first sample's texts and tokens for each
    # prompt, logits None, logprobs for the first sample when requested.
    # Ignored inputs: return_logits, echo, pad_output.
    # KNOWN ISSUE: with multi_sampling greater than 1 only sample index 0 per
    # prompt is returned; the other samples are generated and discarded, despite
    # the comment about supporting multisampling.
    # KNOWN ISSUE: max_model_len is read from args without a default, raising
    # KeyError unless supplied through engine_kwargs.
    # KNOWN ISSUE: BeamSearchParams is passed to LLM.generate / submit_request,
    # which expect SamplingParams; vLLM beam search requires a separate API, so
    # the beam path is likely broken.
    # KNOWN ISSUE: the logprobs-missing check can never fire since the
    # per-sample lists are never None; the seed comment says per-request seed is
    # unsupported yet the seed is still set.
    """The entry point API for vLLM Sampler"""
    if isinstance(input_strings, str):
      input_strings = [input_strings]

    # max_tokens: maximum number of tokens to generate
    if max_generation_steps > self.args["max_model_len"]:
      raise ValueError(
          "`max_generation_steps` must be less than or equal to "
          "`max_model_len`. Received:  `max_generation_steps`="
          f"{max_generation_steps} and `max_model_len`="
          f"{self.args['max_model_len']}."
      )
    if beam_size is not None:
      self.sampling_params = BeamSearchParams(
          beam_width=beam_size,
          max_tokens=max_generation_steps,
          ignore_eos=False,
          temperature=temperature,
      )
    else:
      if self._driver is not None:
        diff_params = (
            self._driver.llm_engine.model_config.get_diff_sampling_param()
        )
        if diff_params:
          sampling_params = SamplingParams.from_optional(**diff_params)
        else:
          sampling_params = SamplingParams()
      else:
        sampling_params = self.llm.get_default_sampling_params()
      sampling_params.detokenize = False
      sampling_params.max_tokens = max_generation_steps
      sampling_params.n = multi_sampling
      sampling_params.temperature = temperature
      if self.config.return_logprobs:
        sampling_params.logprobs = 1  # b/428730696
        sampling_params.prompt_logprobs = 1  # b/428730696
      else:
        sampling_params.logprobs = 0
        sampling_params.prompt_logprobs = 0
      sampling_params.stop_token_ids = [self.tokenizer.eos_id()]
      sampling_params.skip_special_tokens = True
      # Keep the stop token in the returned ``token_ids`` so multi-turn
      # consumers can reconstruct the exact sequence the model was sampled
      # on. This makes the trainer-side concatenation align with what
      # ``apply_chat_template`` produces for the next turn's prompt; without
      # it, the trailing ``<|im_end|>`` (or equivalent eos token) is missing
      # at every turn boundary in the recorded sequence, biasing logp
      # recomputation against the model's actual sampling context.
      sampling_params.include_stop_str_in_output = True

      if top_p is not None:
        sampling_params.top_p = top_p
      if top_k is not None:
        sampling_params.top_k = top_k
      if seed is not None:
        sampling_params.seed = seed

      sampling_kwargs = self.config.sampling_kwargs.copy()
      sampling_kwargs.update(kwargs)
      if sampling_kwargs:
        try:
          logging.log_first_n(
              logging.INFO,
              "Received additional kwargs that are not explicitly defined in"
              f" the method signature: {sampling_kwargs}. These will be"
              " forwarded to the underlying sampler, but please ensure that"
              " they are valid.",
              1,
          )
          for key, value in sampling_kwargs.items():
            logging.log_first_n(
                logging.DEBUG,
                f"Sampler kwargs setting key {key} with value {value}.",
                len(sampling_kwargs),
            )
            setattr(sampling_params, key, value)
        except (AttributeError, TypeError) as e:
          logging.info(
              "Failed to update sampling_params with kwargs:"
              f" {sampling_kwargs}. Error: {e}",
          )

      self.sampling_params = sampling_params

    prompt_ids = [self.tokenize(x) for x in input_strings]
    prompt_objects = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]
    if self._driver is not None:
      outputs = self._generate_server_mode(prompt_objects, self.sampling_params)
    else:
      outputs = self.llm.generate(
          prompts=prompt_objects,
          sampling_params=self.sampling_params,
          use_tqdm=True,
      )
    decoded_outputs, out_logprobs, out_tokens = self.detokenize(
        input_strings, outputs
    )
    if self.config.return_logprobs and (
        out_logprobs is None or out_logprobs[0] is None
    ):
      raise ValueError("Logprobs are not returned from the vLLM.")

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

    # To support multisampling, just return the whole list of SamplerOutput
    return base_sampler.SamplerOutput(
        text=decoded_outputs[0],
        logits=None,
        tokens=out_tokens[0],
        padded_prompt_tokens=all_input_ids,
        logprobs=out_logprobs[0] if self.config.return_logprobs else None,
    )
