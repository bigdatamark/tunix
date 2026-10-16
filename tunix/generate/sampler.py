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
# Native JAX/flax-nnx autoregressive sampler ("vanilla" sampler) for tunix
# transformer models. Supports greedy decoding, top-p/top-k sampling with
# temperature, and beam search (via beam_search module), optional per-step
# logits and log-probability capture, forbidden tokens, custom EOS sets,
# multimodal image inputs, echo of the prompt and fixed-length padded output.
# Design: prompts are tokenized, left-padded to a fixed prompt length, and
# placed in a fixed-size token buffer of length prompt_length plus
# max_generation_steps. A jitted prefill runs the full prompt through the model
# filling a KV cache and samples the first token; a jitted decode function then
# runs a lax while loop producing one token per step until every row hit an
# EOS or the buffer is full. Both jitted functions donate the sampling state so
# the KV cache is updated in place. Model params are passed as an argument
# (graphdef kept separately) so they are not static in the compiled program.
# Defines type aliases LayerCache (dict of name to array) and Cache (dict of
# layer name to LayerCache).
"""Vanilla sampler for LLM generation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import dataclasses
import inspect
from typing import Any, Optional
import warnings

from absl import logging
import flax
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import graph
from flax.nnx import statelib
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from tunix.generate import base_sampler
from tunix.generate import utils
import tunix.generate.beam_search as beam_search_lib
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.processors import image_processor as image_processor_lib

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


@flax.struct.dataclass
class _SamplingState:
  # [CLASS DOC]
  # HELPER: immutable flax struct dataclass (JAX pytree) carrying the full
  # decoding state through jit and the while loop.
  # Pytree fields: decoding_step (index of the last written token), token_buffer
  # [B, L] int32, positions [B, L] (positions ignoring pad), cache (KV cache
  # pytree), done [B] bool, total_sampling_steps (L), logits_buffer [B, L, V] or
  # None, logprobs_buffer [B, L] or None, forbidden_token_ids tuple or None,
  # seed (PRNG key), beam_search_sampling_state (beam state or None).
  # Static (non-pytree) fields: sampling_mode (greedy, top_p or beam_search),
  # num_input_tokens (padded prompt length), temperature, sampling_parameters
  # (dict with top_p and top_k, or beam_size).
  # Note total_sampling_steps and forbidden_token_ids are declared as pytree
  # fields though they hold Python values.
  """Internal sampling state."""

  # Decoding step.
  decoding_step: jnp.int32

  # Fixed-size buffer for accumulating the output tokens.
  token_buffer: jnp.ndarray  # [B, L]

  # Position indices, based on ignoring pad tokens.
  positions: jnp.ndarray  # [B, L]

  # Model state for conditioning the model on autoregressively.
  cache: dict[str, dict[str, jaxtyping.Array]]

  # Is decoding done on the given sequence?
  done: jnp.ndarray  # [B]

  # Total sampling steps (including the prompt).
  total_sampling_steps: int

  # Fixed-size buffer for accumulating the output logits.
  logits_buffer: jnp.ndarray | None  # [B, L, V]

  # Fixed-size buffer for accumulating the output logprobs.
  logprobs_buffer: jnp.ndarray | None  # [B, L]

  # List of tokens that are forbidden to be generated.
  forbidden_token_ids: tuple[int, ...] | None

  # Random seed for sampling.
  seed: jax.Array

  # The sampling mode to use, one of "greedy", "top_p" or "beam_search"
  sampling_mode: str = flax.struct.field(pytree_node=False)

  # Number of input tokens with padding.
  num_input_tokens: jnp.int32 = flax.struct.field(pytree_node=False)

  # Tempurature for top_p sampling.
  temperature: float = flax.struct.field(pytree_node=False)

  # Sampling parameters.
  # For top_p, it contains "top_p" and "top_k".
  # For beam search, it contains "beam_size"
  sampling_parameters: dict[str, float | int] = flax.struct.field(
      pytree_node=False
  )

  # Only present when sampling_mode is "beam_search".
  beam_search_sampling_state: (
      beam_search_lib._BeamSearchSamplingState | None
  ) = None


@dataclasses.dataclass(frozen=True)
class CacheConfig:
  # [CLASS DOC]
  # Frozen dataclass configuring the KV cache: cache_size (max sequence length
  # the cache holds; total prompt plus generation length must not exceed it),
  # num_layers, num_kv_heads, head_dim (the last three only used by the
  # deprecated fallback cache initializer).
  """Configuration for the KV cache."""

  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int


def sample_top_p(
    logits: jnp.ndarray,
    key: jax.Array,
    temperature: float,
    top_p: float,
    top_k: int | None,
    return_logprobs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  # [FUNCTION DOC]
  # Samples one token per row with temperature, nucleus (top-p) and top-k
  # filtering.
  # Inputs: logits [B, T, V] (only the last position is used), a PRNG key,
  # temperature (divides logits; must be nonzero), top_p threshold, top_k
  # (None or non-positive means no top-k), and return_logprobs flag.
  # Behavior: casts last-position logits to float32 and divides by temperature.
  # Fast path when top_p is at least 1 and no top-k: categorical sample over the
  # full vocabulary. Otherwise takes the top k logits (k = vocab size when no
  # top-k) sorted descending, computes softmax and its cumulative sum, masks to
  # negative infinity every entry whose cumulative probability before itself
  # already exceeds top_p (so the token crossing the threshold is kept), samples
  # categorically among the remaining and maps back to vocabulary ids.
  # Returns (token ids [B], log-probs [B] or None). Log-probs are computed from
  # the temperature-scaled but unfiltered distribution (log_softmax over the
  # full vocabulary), not from the truncated distribution.
  # No side effects.
  """Sample a token using top-p sampling."""
  # Upcast to float32 for numerical stability of softmax and subsequent cumsum.
  next_token_logits = logits[:, -1].astype(jnp.float32) / temperature

  # top_k=0 or None both mean "no top-k filtering" — use full vocabulary.
  _no_topk = top_k is None or top_k <= 0
  # Skip softmax and sorting if top_p is 1.0 and top_k is full vocab.
  if top_p >= 1.0 and _no_topk:
    next_token = jax.random.categorical(key, logits=next_token_logits)
    if not return_logprobs:
      return next_token, None
    logp = jax.nn.log_softmax(next_token_logits, axis=-1)
    logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
    logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
    return next_token, logp_sampled

  k = next_token_logits.shape[-1] if _no_topk else top_k
  logits_sorted, indices = jax.lax.top_k(next_token_logits, k=k)

  probs_sorted = jax.nn.softmax(logits_sorted, axis=-1)
  cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
  mask = cumsum_probs - probs_sorted > top_p
  logits_sorted = jnp.where(mask, -jnp.inf, logits_sorted)

  next_token_idx = jax.random.categorical(key, logits=logits_sorted)
  next_token = jnp.take_along_axis(indices, next_token_idx[..., None], axis=-1)
  next_token = jnp.squeeze(next_token, axis=-1)

  if return_logprobs:
    logp = jax.nn.log_softmax(next_token_logits, axis=-1)
    logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
    logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  else:
    logp_sampled = None

  return next_token, logp_sampled


def sample_best(
    logits, return_logprobs: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  # [FUNCTION DOC]
  # Greedy selection: returns (argmax token id of the last position for each
  # row, shape [B]; and if return_logprobs, its float32 log-softmax
  # log-probability [B], else None). Input logits are [B, T, V].
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True)
  next_token = next_token[:, 0]
  if not return_logprobs:
    return next_token, None
  logp = jax.nn.log_softmax(logits[:, -1].astype(jnp.float32), axis=-1)
  logp_sampled = jnp.take_along_axis(logp, next_token[..., None], axis=-1)
  logp_sampled = jnp.squeeze(logp_sampled, axis=-1)
  return next_token, logp_sampled


def _init_cache(
    n_layers: int,
    cache_size: int,
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype,
) -> Cache:
  # [FUNCTION DOC]
  # HELPER: deprecated fallback KV cache constructor. Returns a dict with keys
  # layer_0 to layer_{n_layers-1}, each holding k and v zero arrays of shape
  # [batch_size, cache_size, num_kv_heads, head_dim] in the given dtype and an
  # int32 end_index zero vector of shape [batch_size].
  # KNOWN ISSUE: the docstring says it returns the cache for one attention
  # block, but it returns caches for all layers.
  """Create KV cache for the transformer.

  Args:
    n_layers: The number of attention layers.
    cache_size: The size of the cache.
    batch_size: The batch size.
    num_kv_heads: The number of KV attention heads.
    head_dim: The dimension of the KV attention head.
    dtype: The data type of the cache.

  Returns:
    The KV cache for one attention block.
  """

  shape = (batch_size, cache_size, num_kv_heads, head_dim)
  # Jax array is immutable, so updates to each layer creates new arrays.
  return {
      f'layer_{i}': {
          'k': jnp.zeros(shape, dtype=dtype),
          'v': jnp.zeros(shape, dtype=dtype),
          'end_index': jnp.zeros((batch_size,), dtype=jnp.int32)
      }
      for i in range(n_layers)
  }


class Sampler(base_sampler.BaseSampler):
  # [CLASS DOC]
  # BaseSampler implementation running the model directly with JAX. Keeps the
  # model split into a graphdef and its variables (nested State plus a flat list
  # of Variable leaves used as the jitted function argument), a TokenizerAdapter,
  # the CacheConfig, an optional image processor, the jitted prefill and decode
  # functions, and a flag telling whether the model call accepts
  # decode_only_last_token. The EOS id array (eos_ids) is set on each call.
  """Sampler for transformer model."""

  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      cache_config: CacheConfig,
      image_processor: image_processor_lib.ImageProcessor | None = None,
  ):
    # [FUNCTION DOC]
    # Wraps the tokenizer in a TokenizerAdapter if needed; stores cache config and
    # image processor; extracts the graphdef and variables (nnx.variables) from the
    # model and flattens the variables into a list of Variable leaves. Creates
    # jitted versions of the decode function (donating the sampling-state
    # argument) and of the prefill function (donating the state, echo static).
    # Inspects the model's call signature to set the decode_only_last_token
    # support flag. No device computation happens here.
    """Initializes the sampler.

    Args:
      transformer: an instance of the transformer.
      tokenizer: a tokenizer for the given model.
      cache_config: configuration for the KV cache.
      image_processor: The image processor.
    """
    self.tokenizer = tokenizer
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
      self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self.cache_config = cache_config
    self.image_processor = image_processor
    self._transformer_graphdef: graph.NodeDef = nnx.graphdef(transformer)
    self._transformer_state: list[statelib.State] = nnx.variables(transformer)
    self._flattened_transformer_state: list[statelib.State] = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    # We separate out state and graph def so that the state can be passed as an
    # argument to _decode_fn, resulting in it not being treated as a static
    # arg. This greatly reduces the size of the HLO and reduces compile time.
    #
    # We donate the sampling_state (argnum 1) containing the KV cache arrays.
    # JAX arrays are immutable, so updating the cache at each decoding step
    # would normally force JAX to allocate a new memory buffer and copy old
    # contents. Since the KV cache memory footprint scales with batch size and
    # prompt+decoding length (reaching gigabytes), this continuous reallocation
    # and copying triggers massive memory overhead and OOMs. Donating the input
    # state allows the XLA compiler to reuse the memory buffer in-place,
    # completely avoiding allocation/copy overhead.
    self._compiled_decode_fn = jax.jit(self._decode_fn, donate_argnums=(1,))
    self._compiled_prefill_fn = jax.jit(
        self._prefill_fn,
        donate_argnums=(1,),
        static_argnames=('echo',),
    )
    self._supports_decode_only_last_token = (
        'decode_only_last_token'
        in inspect.signature(transformer.__call__).parameters
    )

  def model_def_and_state(self) -> tuple[graph.NodeDef, statelib.State]:
    # [FUNCTION DOC]
    # Returns a tuple of the model graphdef and the flat list of Variable leaves
    # (despite the annotation, not a State).
    """Returns the transformer graphdef and state."""
    return self._transformer_graphdef, self._flattened_transformer_state

  @property
  def transformer(self) -> nnx.Module:
    # [FUNCTION DOC]
    # Property: reconstructs and returns a model module by merging the graphdef
    # with the flat variable leaves (new module object on every access).
    return nnx.merge(
        self._transformer_graphdef, self._flattened_transformer_state
    )

  @property
  def transformer_state(self) -> statelib.State:
    # [FUNCTION DOC]
    # Property getter returning the stored nested variable State.
    return self._transformer_state

  @transformer_state.setter
  def transformer_state(self, state: statelib.State) -> None:
    # [FUNCTION DOC]
    # Property setter replacing model weights (used for weight sync from trainer).
    # Collects the Variable types in the new state. If it contains nnx.Param: full
    # replacement; the new state must have the same tree structure as the current
    # one (ValueError otherwise) and replaces it. Else the state must contain only
    # nnx.LoRAParam (ValueError otherwise): the current LoRA subset must match the
    # new state's structure, and the new state is merged into the current non-LoRA
    # subset. Finally recomputes the flat leaf list.

    def get_all_param_types(tree):
      # [FUNCTION DOC]
      # HELPER: returns the set of Python types of all leaves of tree, treating
      # nnx.Variable instances as leaves.
      param_types = set()
      jax.tree_util.tree_map(
          lambda x: param_types.add(type(x)),
          tree,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
      return param_types

    def check_tree_structure(tree1, tree2):
      # [FUNCTION DOC]
      # HELPER: raises ValueError (showing both treedefs) if the two trees have
      # different pytree structures. Then compares leaves pairwise with the
      # shape/dtype/sharding helper and raises ValueError (showing both trees) if
      # any pair differs. Mixed sharding classes (e.g. single-device vs named) are
      # always considered different.
      if jax.tree_util.tree_structure(tree1) != jax.tree_util.tree_structure(
          tree2
      ):
        raise ValueError(
            'New state must have the same structure as the old state.'
            f' {jax.tree_util.tree_structure(tree1)} vs'
            f' {jax.tree_util.tree_structure(tree2)}'
        )

      def check_shape_dtype_sharding(x, y):
        # [FUNCTION DOC]
        # HELPER: returns True when two leaves have identical shape and dtype and
        # equivalent sharding.

        def equivalent_sharding(x, y):
          # [FUNCTION DOC]
          # HELPER: sharding equivalence. Two single-device shardings are equivalent if
          # their device sets match. Otherwise both must be NamedSharding on the same
          # mesh, and every partition-spec entry present in the first spec but not in
          # the second must be None/empty or name a mesh axis of size 1 (memory kind is
          # ignored on purpose).
          # KNOWN ISSUE: the comparison is a one-directional set difference that ignores
          # positions, so specs differing in axis order or with extra entries only in
          # the second spec are considered equivalent; a tuple spec entry would be used
          # directly as a mesh axis key and fail.
          # Lift the condition on memory_kind due to offloading.
          # Besides it seems jax.jit might change some shardings of the params
          # to equivalent representation so here we check if the specs are
          # equivalent instead of checking the identity.
          if isinstance(
              x.sharding, jax.sharding.SingleDeviceSharding
          ) and isinstance(y.sharding, jax.sharding.SingleDeviceSharding):
            return x.sharding.device_set == y.sharding.device_set
          if not (
              isinstance(x.sharding, jax.sharding.NamedSharding)
              and isinstance(y.sharding, jax.sharding.NamedSharding)
          ):
            return False
          if x.sharding.mesh != y.sharding.mesh:
            return False
          mesh = x.sharding.mesh
          diff_spec = list(set(x.sharding.spec) - set(y.sharding.spec))
          for spec in diff_spec:
            if spec and mesh.shape[spec] != 1:
              return False
          return True

        return (
            jnp.shape(x) == jnp.shape(y)
            and x.dtype == y.dtype
            and equivalent_sharding(x, y)
        )

      if not all(
          jax.tree_util.tree_leaves(
              jax.tree_util.tree_map(check_shape_dtype_sharding, tree1, tree2)
          )
      ):
        raise ValueError(
            'New state must have the same shape, dtype and sharding as the old'
            f' state. {tree1} vs {tree2}'
        )

    param_types = get_all_param_types(state)

    if nnx.Param in param_types:
      # Full state replacement.
      check_tree_structure(self._transformer_state, state)
      self._transformer_state = state
    else:
      # LoRA state replacement.
      if not (len(param_types) == 1 and nnx.LoRAParam in param_types):
        raise ValueError(
            'Only LoRAParam is supported. Received invalid `param_types`: '
            f'{param_types}'
        )
      original_lora_params = statelib.filter_state(
          self._transformer_state, nnx.LoRAParam
      )
      check_tree_structure(original_lora_params, state)
      base_state = statelib.filter_state(
          self._transformer_state, filterlib.Not(nnx.LoRAParam)
      )
      self._transformer_state = statelib.merge_state(base_state, state)

    self._flattened_transformer_state = jax.tree.leaves(
        self._transformer_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  @property
  def dtype(self) -> jnp.dtype:
    # [FUNCTION DOC]
    # Property: returns the model config's dtype if the model has a config with a
    # dtype attribute, otherwise the dtype of the first flat variable leaf. Merges
    # a fresh model on each access (cost only, no side effects).
    if hasattr(self.transformer, 'config') and (
        hasattr(self.transformer.config, 'dtype')
    ):
      return self.transformer.config.dtype
    return self._flattened_transformer_state[0].dtype

  def init_sample_state(
      self,
      all_input_ids: jax.Array,
      total_sampling_steps: int,
      include_logits: bool,
      forbidden_token_ids: tuple[int, ...] | None,
      temperature: float,
      top_p: Optional[float],
      top_k: Optional[int],
      seed: jax.Array,
      beam_size: Optional[int],
      include_logprobs: bool = False,
  ) -> _SamplingState:
    # [FUNCTION DOC]
    # Builds the initial _SamplingState.
    # Inputs: all_input_ids [B, P] left-padded prompts; total_sampling_steps L;
    # include_logits; forbidden_token_ids; temperature; top_p; top_k; seed key;
    # beam_size; include_logprobs.
    # Behavior: token buffer [B, L] filled with pad id, prompt copied into the first
    # P columns; mask true for non-pad prompt positions and all generation slots;
    # positions computed with utils.build_positions_from_mask. done all False.
    # Cache from the model's init_cache(batch, cache_size, dtype) if available,
    # otherwise the deprecated helper (emits DeprecationWarning). Optional float32
    # zero logits buffer [B, L, num_embed of the model] and logprobs buffer [B, L].
    # Sampling mode: beam_search if beam_size given (stores beam_size), top_p if
    # top_p given (stores top_p and top_k), greedy otherwise; utils
    # check_sampling_mode_conflict raises when both beam and top_p are given.
    # decoding_step starts at P - 1.
    # KNOWN ISSUE: top_k without top_p is silently ignored (greedy is used), and
    # the stored num_input_tokens is a JAX array in a static field.
    """Initializes the sampling state given input prompts."""
    batch_size = all_input_ids.shape[0]
    num_input_tokens = all_input_ids.shape[1]

    token_buffer = jnp.full(
        (
            batch_size,
            total_sampling_steps,
        ),
        self.tokenizer.pad_id(),
        dtype=jnp.int32,
    )
    input_mask = jnp.ones_like(token_buffer, dtype=jnp.bool_)
    token_buffer = token_buffer.at[:, :num_input_tokens].set(all_input_ids)
    input_mask = input_mask.at[:, :num_input_tokens].set(
        all_input_ids != self.tokenizer.pad_id()
    )
    positions = utils.build_positions_from_mask(input_mask)

    done = jnp.zeros((batch_size,), dtype=jnp.bool_)

    if hasattr(self.transformer, 'init_cache'):
      cache = self.transformer.init_cache(
          batch_size, self.cache_config.cache_size, self.dtype
      )
    else:
      warnings.warn(
          'Using deprecated _init_cache in Tunix sampler. Models are now'
          ' required to have their own init_cache attribute.',
          DeprecationWarning,
      )
      cache = _init_cache(
          n_layers=self.cache_config.num_layers,
          cache_size=self.cache_config.cache_size,
          batch_size=batch_size,
          num_kv_heads=self.cache_config.num_kv_heads,
          head_dim=self.cache_config.head_dim,
          dtype=self.dtype,
      )

    if include_logits:
      logits_buffer = jnp.zeros(
          (batch_size, total_sampling_steps, self.transformer.num_embed),
          dtype=jnp.float32,
      )
    else:
      logits_buffer = None

    if include_logprobs:
      logprobs_buffer = jnp.zeros(
          (batch_size, total_sampling_steps),
          dtype=jnp.float32,
      )
    else:
      logprobs_buffer = None
    sampling_parameters = {}
    sampling_mode = [None]

    if beam_size is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'beam_search')
      sampling_parameters['beam_size'] = beam_size

    if top_p is not None:
      utils.check_sampling_mode_conflict(sampling_mode, 'top_p')
      sampling_parameters['top_p'] = top_p
      sampling_parameters['top_k'] = top_k

    if sampling_mode[0] is None:
      sampling_mode[0] = 'greedy'

    logging.debug('Using sampling mode: %s', sampling_mode[0])

    return _SamplingState(
        decoding_step=num_input_tokens - 1,
        num_input_tokens=jnp.array(num_input_tokens, dtype=jnp.int32),
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        logprobs_buffer=logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        sampling_parameters=sampling_parameters,
        seed=seed,
        sampling_mode=sampling_mode[0],
        beam_search_sampling_state=None,
    )

  def tokenize(self, input_string: str) -> np.ndarray | list[int]:
    # [FUNCTION DOC]
    # Encodes a string, prepends BOS if its id is truthy, removes duplicate
    # leading BOS ids and returns an int32 numpy array.
    """Tokenizes the input string."""
    input_ids = self.tokenizer.encode(input_string)
    bos_tok = [self.tokenizer.bos_id()] if self.tokenizer.bos_id() else []
    input_ids = np.array(
        self.tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=np.int32
    )
    return input_ids

  def _sample(
      self,
      logits: jnp.ndarray,
      eos: jax.Array,
      cache: dict[str, dict[str, jaxtyping.Array]],
      sampler_state: _SamplingState,
  ) -> _SamplingState:
    # [FUNCTION DOC]
    # HELPER: picks the next token from model logits and advances the state.
    # Uses only the last position's logits ([B, 1, V]); sets forbidden token
    # logits to negative infinity. Beam search mode delegates to
    # beam_search_step (passing the first EOS id as pad id) and takes its
    # reordered cache, token buffer, done, logits and logprobs buffers. Greedy
    # uses sample_best; top_p uses sample_top_p with a key derived by folding the
    # decoding step into the seed; any other mode raises ValueError. For non-beam
    # modes the token (and log-prob when buffered) is written at column
    # decoding_step + 1. Then done is OR-ed with whether that new token is in eos.
    # Returns a new state with decoding_step incremented and the given cache.
    """Samples a token from the logits."""

    logits = logits[:, -1][:, None, :]  # B, 1, V
    decoding_step = sampler_state.decoding_step
    token_buffer = sampler_state.token_buffer
    done = sampler_state.done
    logits_buffer = sampler_state.logits_buffer
    logprobs_buffer = sampler_state.logprobs_buffer
    beam_search_state = sampler_state.beam_search_sampling_state
    if sampler_state.forbidden_token_ids:
      logits = logits.at[:, :, sampler_state.forbidden_token_ids].set(-jnp.inf)

    if sampler_state.sampling_mode == 'beam_search':
      beam_search_state, updated_args = beam_search_lib.beam_search_step(
          logits=logits,
          done=done,
          token_buffer=token_buffer,
          cache=cache,
          logits_buffer=logits_buffer,
          state=beam_search_state,
          pad_token_id=eos[0],
          decoding_step=decoding_step,
          logprobs_buffer=logprobs_buffer,
      )
      cache = updated_args['cache']
      token_buffer = updated_args['token_buffer']
      done = updated_args['done']
      logits_buffer = updated_args['logits_buffer']
      logprobs_buffer = updated_args['logprobs_buffer']
    else:
      if sampler_state.sampling_mode == 'greedy':
        next_token_candidate, logp = sample_best(
            logits, return_logprobs=(logprobs_buffer is not None)
        )
      elif sampler_state.sampling_mode == 'top_p':
        key = jax.random.fold_in(sampler_state.seed, decoding_step)
        next_token_candidate, logp = sample_top_p(
            logits,
            key,
            sampler_state.temperature,
            sampler_state.sampling_parameters['top_p'],
            sampler_state.sampling_parameters['top_k'],
            return_logprobs=(logprobs_buffer is not None),
        )
      else:
        raise ValueError(
            'Unsupported sampling mode: %s' % sampler_state.sampling_mode
        )
      token_buffer = token_buffer.at[:, decoding_step + 1].set(
          next_token_candidate
      )
      if logprobs_buffer is not None:
        logprobs_buffer = logprobs_buffer.at[:, decoding_step + 1].set(logp)

    done = done | jnp.isin(token_buffer[:, decoding_step + 1], eos)
    return _SamplingState(
        decoding_step=sampler_state.decoding_step + 1,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        positions=sampler_state.positions,
        logits_buffer=logits_buffer,
        logprobs_buffer=logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        temperature=sampler_state.temperature,
        sampling_parameters=sampler_state.sampling_parameters,
        seed=sampler_state.seed,
        sampling_mode=sampler_state.sampling_mode,
        beam_search_sampling_state=beam_search_state,
    )

  def _prefill_fn(
      self,
      params: statelib.State,
      sampler_state: _SamplingState,
      images: jnp.ndarray | None = None,
      echo: bool = True,
  ) -> _SamplingState:
    # [FUNCTION DOC]
    # HELPER (jitted with state donated and echo static): runs the prompt through
    # the model and samples the first generated token.
    # Slices the first num_input_tokens columns of the token buffer and positions.
    # Builds the attention mask: the model's get_attention_mask on the non-pad
    # mask, right-padded with zeros to cache_size, if the model provides it,
    # otherwise utils.make_causal_attn_mask with cache_size. Merges the model
    # from params. Passes images when given, and decode_only_last_token True when
    # supported and echo is False. Calls the model with tokens, positions, cache,
    # mask to get logits and the filled cache.
    # If logits are buffered, writes them into the logits buffer starting at
    # column num_input_tokens (last-token logits only) or column 1 (full prompt
    # logits, shifted so the logit at i predicts token i+1).
    # For beam search, expands all state arrays by beam size via
    # init_batched_beam_state. Then calls the sampling helper with the model logits
    # and self.eos_ids and returns the resulting state. Column 0 of the logits
    # buffer never receives logits.
    """Performs prefill."""
    batch_size = sampler_state.token_buffer.shape[0]

    tokens = jax.lax.dynamic_slice(
        sampler_state.token_buffer,
        start_indices=jnp.zeros(
            (sampler_state.token_buffer.ndim,), dtype=jnp.int32
        ),
        slice_sizes=(batch_size, sampler_state.num_input_tokens),
    )
    step_positions = jax.lax.dynamic_slice(
        sampler_state.positions,
        start_indices=jnp.zeros(
            (sampler_state.token_buffer.ndim,), dtype=jnp.int32
        ),
        slice_sizes=(batch_size, sampler_state.num_input_tokens),
    )

    input_mask = tokens != self.tokenizer.pad_id()

    if hasattr(self.transformer, 'get_attention_mask'):
      attention_mask = self.transformer.get_attention_mask(
          tokens, inputs_mask=input_mask
      )
      seq_len = attention_mask.shape[-1]
      padding = self.cache_config.cache_size - seq_len
      attention_mask = jnp.pad(
          attention_mask,
          (*((0, 0) for _ in range(attention_mask.ndim - 1)), (0, padding)),
      )
    else:
      attention_mask = utils.make_causal_attn_mask(
          input_mask, self.cache_config.cache_size
      )

    transformer = nnx.merge(self._transformer_graphdef, params)
    kwargs = {} if images is None else {'images': images}
    decode_only_last_token = (
        self._supports_decode_only_last_token
        and not echo
    )
    if decode_only_last_token:
      kwargs['decode_only_last_token'] = True
    logits, cache = transformer(
        tokens,
        step_positions,
        sampler_state.cache,
        attention_mask,
        **kwargs,
    )
    token_buffer = sampler_state.token_buffer
    done = sampler_state.done
    positions = sampler_state.positions
    beam_search_sampling_state = None
    if sampler_state.logits_buffer is not None:
      start_idx = (
          sampler_state.num_input_tokens if decode_only_last_token else 1
      )
      logits_buffer = jax.lax.dynamic_update_slice(
          sampler_state.logits_buffer,
          logits.astype(sampler_state.logits_buffer.dtype),
          (0, start_idx, 0),
      )
    else:
      logits_buffer = sampler_state.logits_buffer

    if sampler_state.sampling_mode == 'beam_search':
      # init beam state in prefill instead of init as one minor optimization
      # to avoid running unnecessary prefill for
      # duplicated input prompt per beam.
      sampling_state, updated_args = beam_search_lib.init_batched_beam_state(
          logits=logits,
          input_token_buffer=sampler_state.token_buffer,
          initial_cache=cache,
          done=sampler_state.done,
          positions=sampler_state.positions,
          logits_buffer=sampler_state.logits_buffer,
          beam_size=int(sampler_state.sampling_parameters['beam_size']),
      )
      beam_search_sampling_state = sampling_state
      logits = updated_args['logits']
      cache = updated_args['cache']
      token_buffer = updated_args['token_buffer']
      done = updated_args['done']
      positions = updated_args['positions']
      logits_buffer = updated_args['logits_buffer']

    updated_sampling_state = _SamplingState(
        decoding_step=sampler_state.decoding_step,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        logprobs_buffer=sampler_state.logprobs_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
        temperature=sampler_state.temperature,
        sampling_parameters=sampler_state.sampling_parameters,
        seed=sampler_state.seed,
        sampling_mode=sampler_state.sampling_mode,
        beam_search_sampling_state=beam_search_sampling_state,
    )
    updated_sampler_state = self._sample(
        logits=logits,
        cache=cache,
        eos=self.eos_ids,
        sampler_state=updated_sampling_state,
    )
    return updated_sampler_state

  def _decode_fn(
      self,
      params: statelib.State,
      sampling_state: _SamplingState,
  ) -> _SamplingState:
    # [FUNCTION DOC]
    # HELPER (jitted with state donated): runs a lax while loop applying one
    # sampling step per iteration while decoding_step is below
    # total_sampling_steps and at least one row is not done. Returns the final
    # state.
    # KNOWN ISSUE: the last iteration runs with decoding_step equal to
    # total_sampling_steps minus 1 and writes the token at index
    # total_sampling_steps, which is out of bounds and silently dropped by JAX.
    """Internal generating function (to be jitted)."""

    def sample_with_params(sampler_state: _SamplingState):
      # [FUNCTION DOC]
      # HELPER: while-loop body; applies one sampling step with the captured params.
      return self._sample_step(params, sampler_state)

    def cond_fn(sampler_state: _SamplingState):
      # [FUNCTION DOC]
      # HELPER: while-loop condition; true while decoding_step is below
      # total_sampling_steps and any row is not done.
      return (
          sampler_state.decoding_step < sampler_state.total_sampling_steps
      ) & jnp.any(jnp.logical_not(sampler_state.done))

    return jax.lax.while_loop(cond_fn, sample_with_params, sampling_state)

  def _sample_step(
      self, params: statelib.State, sampler_state: _SamplingState
  ) -> _SamplingState:
    # [FUNCTION DOC]
    # HELPER: one decode step. Takes the token at decoding_step for each row and
    # its position, builds the attention mask with utils.compute_attention_masks
    # from the step, cache size and a mask of pad positions over the whole token
    # buffer, merges the model and runs it on the single token with the cache.
    # Calls the sampling helper, then if logits are buffered writes this step's
    # logits at column decoding_step + 1. Returns the updated state.
    # KNOWN ISSUE: in beam search mode the logits buffer is reordered by beam
    # selection inside the sampling helper and afterwards this step's logits are
    # written in the old (pre-reorder) beam order, misaligning them. Generated
    # tokens equal to the pad id are also treated as padding in the mask.
    """Performs a single sampling step."""
    batch_size = sampler_state.token_buffer.shape[0]
    decoding_step = sampler_state.decoding_step

    last_token = sampler_state.token_buffer[:, decoding_step]
    last_token = last_token.reshape((batch_size, 1))
    step_positions = jnp.expand_dims(
        sampler_state.positions[:, decoding_step], -1
    )

    input_mask = sampler_state.token_buffer == self.tokenizer.pad_id()
    attention_mask = utils.compute_attention_masks(
        decoding_step, self.cache_config.cache_size, input_mask
    )

    transformer = nnx.merge(self._transformer_graphdef, params)
    logits, cache = transformer(
        last_token,
        positions=step_positions,
        cache=sampler_state.cache,
        attention_mask=attention_mask,
    )
    updated_sampler_state = self._sample(
        logits=logits,
        cache=cache,
        eos=self.eos_ids,
        sampler_state=sampler_state,
    )

    if updated_sampler_state.logits_buffer is not None:
      next_logits = jnp.squeeze(logits, 1)
      logits_buffer = updated_sampler_state.logits_buffer.at[
          :, decoding_step + 1
      ].set(next_logits)
    else:
      logits_buffer = None

    updated_sampler_state = dataclasses.replace(
        updated_sampler_state,
        logits_buffer=logits_buffer,
    )
    return updated_sampler_state

  def __call__(
      self,
      input_strings: str | Sequence[str],
      max_generation_steps: int,
      max_prompt_length: int | None = None,
      echo: bool = False,
      return_logits: bool = False,
      return_logprobs: bool = False,
      eos_tokens: Sequence[int] | None = None,
      forbidden_tokens: Iterable[int] | None = None,
      temperature: float = 0.0,
      top_p: Optional[float] = None,
      top_k: Optional[int] = None,
      beam_size: Optional[int] = None,
      seed: int | None = None,
      pad_output: bool = False,
      images: (
          str
          | np.ndarray
          | list[str | np.ndarray | list[str | np.ndarray] | None]
          | jnp.ndarray
          | None
      ) = None,
  ) -> base_sampler.SamplerOutput:
    # [FUNCTION DOC]
    # Main entry point; samples completions for one or more prompts.
    # Side effect: sets self.eos_ids to a JAX array of eos_tokens or the
    # tokenizer EOS id.
    # Steps: normalize input to a list; forbidden tokens to a tuple (or None);
    # tokenize prompts; if images and an image processor are given, process images
    # into a JAX array (otherwise raw images are passed through); compute
    # max_prompt_length (next power of two of the longest prompt when missing or
    # too small); left-pad prompts with the pad id; total_sampling_steps =
    # max_prompt_length + max_generation_steps, raising ValueError if it exceeds
    # cache_size. Seed: PRNGKey(0) if None, PRNGKey(seed) for an int, else used
    # as given. Initialize state, run jitted prefill then jitted decode. For beam
    # search, finalize to the top beam per prompt.
    # Output extraction:
    # - pad_output True: utils.padded_fill_tokens_and_logits produces fixed-length
    #   (total steps if echo else max_generation_steps) tokens/logits and lengths;
    #   texts decode the first length tokens; logprobs, if requested, are sliced
    #   from start (first non-pad index if echo, else prompt length) to the first
    #   EOS after the prompt and right-padded with zeros to that fixed length.
    # - pad_output False: per row slices tokens (and logits/logprobs when
    #   requested) from the same start to the first EOS index after the prompt
    #   (EOS excluded), moved to host; texts are decoded slices.
    # Returns SamplerOutput(text, logits list or empty list, tokens, padded prompt
    # array, logprobs or None).
    # Raises ValueError on cache overflow or conflicting sampling modes.
    # KNOWN ISSUE: eos_ids is read inside the jitted functions at trace time and
    # baked in as a constant; jit caching does not key on it, so changing
    # eos_tokens between calls with identical shapes reuses the old EOS set.
    # KNOWN ISSUE: the cache-size error message says must be less than but equality
    # is allowed. top_p sampling with temperature 0 divides by zero.
    """Samples a completion of the input string.

    If top_p is provided, the sampling mode will be top_p.
    If beam_size is provided, the sampling mode will be beam_search.
    If None of them are provided, the sampling mode will be greedy.

    Args:
      input_strings: input prompts to feed to the model for sampling.
      max_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      max_prompt_length: maximum length of the prompt. Specify to avoid
        recompilation on different prompt lengths.
      echo: whgether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.
      eos_tokens: end of sequence tokens to stop generation. If None, the
        tokenizer's eos_id will be used.
      forbidden_tokens: Optional Iterable of token IDs that are disallowed.
      temperature: temperature for sampling.
      top_p: top-p sampling threshold.
      top_k: top-k sampling threshold.
      beam_size: beam size for beam search.
      seed: random seed for sampling.
      pad_output: whether to pad the output to maximum length. If this set as
        True, the output len will be max_generation_steps if echo is False,
        otherwise it will be max_generation_steps + max_prompt_length. The
        padding now only supports right padding. Can modify to support left
        padding if needed.
      images: input images to process. Can be a string/array, list of
        strings/arrays, or list of list of strings/arrays depending on whether
        there is one, multiple, or varying number of images per batch.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    self.eos_ids = jnp.array(eos_tokens or [self.tokenizer.eos_id()])
    input_strings = (
        [input_strings] if isinstance(input_strings, str) else input_strings
    )

    forbidden_token_ids = tuple(forbidden_tokens) if forbidden_tokens else None

    tokens = [self.tokenize(x) for x in input_strings]

    processed_images = images
    if images is not None and self.image_processor is not None:
      processed_images = self.image_processor(images)
      processed_images = jnp.array(processed_images)

    max_tokens_length = max(len(x) for x in tokens)
    if max_prompt_length is None or max_prompt_length < max_tokens_length:
      max_prompt_length = utils.next_power_of_2(max_tokens_length)

    all_input_ids = np.array([
        utils.pad_to_length(
            x,
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,
        )
        for x in tokens
    ])

    total_sampling_steps = max_prompt_length + max_generation_steps
    if total_sampling_steps > self.cache_config.cache_size:
      raise ValueError(
          f'Total sampling steps {total_sampling_steps} must be less than the'
          f' cache size {self.cache_config.cache_size}.'
      )

    if seed is None:
      seed = jax.random.PRNGKey(0)
    elif isinstance(seed, int):
      seed = jax.random.PRNGKey(seed)
    sampling_state = self.init_sample_state(
        jnp.array(all_input_ids),
        include_logits=return_logits,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        beam_size=beam_size,
        include_logprobs=return_logprobs,
    )
    sampling_state = self._compiled_prefill_fn(
        self._flattened_transformer_state,
        sampling_state,
        processed_images,
        echo=echo,
    )

    sampling_state = self._compiled_decode_fn(
        self._flattened_transformer_state, sampling_state
    )
    token_buffers = sampling_state.token_buffer
    logits_buffers = sampling_state.logits_buffer

    final_logprobs_buffer = sampling_state.logprobs_buffer

    if sampling_state.sampling_mode == 'beam_search':
      updated_args = beam_search_lib.finalize_beam_search_state(
          sampling_state.beam_search_sampling_state,
          sampling_state.token_buffer,
          sampling_state.logits_buffer,
          sampling_state.logprobs_buffer,
      )
      token_buffers = updated_args['token_buffer']
      logits_buffers = updated_args['logits_buffer']
      final_logprobs_buffer = updated_args['logprobs_buffer']
      # delete the sampling state in case the further referece
      # if need more internal states, they should be updated by
      # finalize_beam_search_state
      del sampling_state
    if pad_output:
      max_len = total_sampling_steps if echo else max_generation_steps
      lengths, out_tokens, out_logits = utils.padded_fill_tokens_and_logits(
          token_buffers,
          logits_buffers,
          return_logits,
          echo,
          self.tokenizer.pad_id(),
          self.eos_ids,
          max_prompt_length,
          max_len,
      )
      out_tokens, lengths = jax.device_get(out_tokens), jax.device_get(lengths)
      decoded_outputs = [
          self.tokenizer.decode(tokens[:length].tolist())
          for tokens, length in zip(out_tokens, lengths)
      ]
      out_logprobs = []
      if return_logprobs:
        token_buffers = jax.device_get(token_buffers)
        final_logprobs_buffer = jax.device_get(final_logprobs_buffer)
        for i in range(len(token_buffers)):
          start_idx = (
              utils.np_find_first_non_pad_idx(
                  token_buffers[i], self.tokenizer.pad_id()
              )
              if echo
              else max_prompt_length
          )
          end_idx = (
              utils.np_find_first_eos_idx(
                  token_buffers[i][max_prompt_length:], self.eos_ids
              )
              + max_prompt_length
          )
          length = end_idx - start_idx
          # Slice logprobs and pad to max_len
          sliced_logprobs = final_logprobs_buffer[i][start_idx:end_idx]
          padded_logprobs = np.pad(
              sliced_logprobs,
              (0, max_len - length),
              mode='constant',
              constant_values=0.0,
          )
          out_logprobs.append(padded_logprobs.tolist())

    else:
      out_tokens = []
      out_logits = []
      out_logprobs = []
      token_buffers = jax.device_get(token_buffers)
      if return_logprobs:
        final_logprobs_buffer = jax.device_get(final_logprobs_buffer)
      if return_logits:
        logits_buffers = jax.device_get(logits_buffers)
      for i in range(len(token_buffers)):
        token_buffer = token_buffers[i]
        start_idx = (
            utils.np_find_first_non_pad_idx(
                token_buffer, self.tokenizer.pad_id()
            )
            if echo
            else max_prompt_length
        )
        end_idx = (
            utils.np_find_first_eos_idx(
                token_buffer[max_prompt_length:], self.eos_ids
            )
            + max_prompt_length
        )
        out_tokens.append(token_buffer[start_idx:end_idx])
        if return_logits:
          out_logits.append(logits_buffers[i][start_idx:end_idx])
        if return_logprobs:
          # Extract logprobs for the generated tokens
          out_logprobs.append(
              final_logprobs_buffer[i][start_idx:end_idx].tolist()
          )

      decoded_outputs = [
          self.tokenizer.decode(tokens.tolist()) for tokens in out_tokens
      ]

    result = base_sampler.SamplerOutput(
        text=decoded_outputs,
        logits=out_logits if return_logits else [],
        tokens=out_tokens,
        padded_prompt_tokens=all_input_ids,
        logprobs=out_logprobs if return_logprobs else None,
    )
    return result
