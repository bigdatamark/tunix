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

# [MODULE DOC]
# Flax NNX implementation of the Gemma 4 decoder-only transformer (text only).
# Key features beyond Gemma 3:
# - Configurable cyclic attention pattern of LOCAL_SLIDING / GLOBAL layers;
#   global layers may use a different number of kv heads
#   (num_global_kv_heads), a larger head size (global_key_size, default 512),
#   partial RoPE (global_rope_proportion of the rotary frequencies; the rest
#   unrotated) and shared key/value projections (k_eq_v_global).
# - QK RMSNorm plus a parameter-free RMS normalization of values.
# - KV-cache sharing: the last frac_shared_layers of layers reuse the K/V of
#   an earlier layer (create_kv_cache_sharing_patterns) and keep no cache of
#   their own; they may use a different FFN width
#   (override_kv_shared_ffw_hidden).
# - Ring-buffer KV caches of size sliding_window_size for local layers
#   (use_sliding_window_kv_cache, default on).
# - Optional TPU splash (flash) attention for prefill / training, run under
#   shard_map; mutually exclusive with the sliding window cache.
# - Per-layer input embeddings (PLE, per_layer_input_dim > 0): an extra
#   embedding table and projection producing one small vector per layer that
#   gates an additive residual update in each layer.
# - Optional mixture-of-experts branch (moe.MoERagged) added to the dense MLP.
# - Per-layer learned scalar skip_scale on the layer output; final logit
#   soft-capping (default 30.0); RMSNorm scale applied directly (init 1).
# - Separate compute dtype (dtype) and parameter dtype (param_dtype).
# Types: LayerCache dict with 'k', 'v' [B, cache_len, K, H] and 'end_index'
# int32 [B]; Cache maps 'layer_<i>' (non-shared layers only) to LayerCache.
# K_MASK is a large negative masking value. GEMMA4_ATTENTION_PATTERN is the
# default 5 local + 1 global pattern. Side effect at import:
# env_utils.setup_sharding_environment().
"""Gemma4 model."""

import dataclasses
import enum
from functools import partial
import itertools
from typing import Tuple
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.shard_map import shard_map
from jax.interpreters import pxla
import jax.sharding as shd
from jax.sharding import PartitionSpec as P
import jaxtyping
import numpy as np
from tunix.generate.mappings import BackendMappingMixin
from tunix.models.gemma4 import moe
from tunix.utils import compat
from tunix.utils import env_utils
from tunix.utils.sharding_utils import shard


env_utils.setup_sharding_environment()


LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class RematConfig(enum.Enum):
  # [CLASS DOC]
  # Enum of rematerialization modes: NONE, BLOCK (remat attention and MLP
  # blocks individually), DECODER (remat whole decoder layers).
  NONE = enum.auto()
  BLOCK = enum.auto()
  DECODER = enum.auto()


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  # [CLASS DOC]
  # Frozen dataclass of partition-spec tuples for Gemma 4 weights and
  # activations: Gemma 3 fields plus MoE expert weights (exp_weight_edf,
  # exp_weight_efd) and per-layer-embedding weights (per_layer_model_projection,
  # per_layer_input_gate, per_layer_projection, per_layer_input_embedding).
  """Sharding configuration for gemma transformer."""

  emb_vd: Tuple[str | None, ...]
  q_weight_ndh: Tuple[str | None, ...]
  kv_weight_cndh: Tuple[str | None, ...]
  qkv_weight_cndh: Tuple[str | None, ...]
  o_weight_nhd: Tuple[str | None, ...]
  ffw_weight_df: Tuple[str | None, ...]
  ffw_weight_fd: Tuple[str | None, ...]
  rms_norm_weight: Tuple[str | None, ...]
  act_btd: Tuple[str | None, ...]
  act_btf: Tuple[str | None, ...]
  act_btnh: Tuple[str | None, ...]
  vision_proj: Tuple[str | None, ...]
  vision_soft_emb_norm_weight: Tuple[str | None, ...]
  # MoE sharding
  exp_weight_edf: Tuple[str | None, ...]
  exp_weight_efd: Tuple[str | None, ...]
  # PLE sharding
  per_layer_model_projection: Tuple[str | None, ...]
  per_layer_input_gate: Tuple[str | None, ...]
  per_layer_projection: Tuple[str | None, ...]
  per_layer_input_embedding: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    # [FUNCTION DOC]
    # Static factory for the default sharding; with is_sampling True, 'fsdp' is
    # removed from weight specs and act_btd's feature dimension is unsharded.
    # Returns a new ShardingConfig.
    fsdp = 'fsdp' if not is_sampling else None

    return ShardingConfig(
        emb_vd=('tp', fsdp),
        q_weight_ndh=('tp', fsdp, None),
        kv_weight_cndh=(None, 'tp', fsdp, None),
        qkv_weight_cndh=(None, 'tp', fsdp, None),
        o_weight_nhd=('tp', None, fsdp),
        ffw_weight_df=(fsdp, 'tp'),
        ffw_weight_fd=('tp', fsdp),
        rms_norm_weight=('tp',),
        act_btd=('fsdp', None, None if is_sampling else 'tp'),
        act_btf=('fsdp', None, 'tp'),
        act_btnh=('fsdp', None, 'tp', None),
        vision_proj=(fsdp, 'tp'),
        vision_soft_emb_norm_weight=('tp',),
        exp_weight_edf=(fsdp, None, None, 'tp'),
        exp_weight_efd=(fsdp, 'tp', None),
        per_layer_model_projection=(fsdp, None, 'tp'),
        per_layer_input_gate=(fsdp, 'tp'),
        per_layer_projection=('tp', fsdp),
        per_layer_input_embedding=('tp', None, fsdp),
    )


@dataclasses.dataclass(slots=True, kw_only=True)
class ModelConfig:
  # [CLASS DOC]
  # Keyword-only dataclass of Gemma 4 hyperparameters. Core sizes (num_layers,
  # num_embed, embed_dim, hidden_dim, num_heads, head_dim, num_kv_heads);
  # final_logit_softcap (30.0, None disables); sliding_window_size;
  # per_layer_input_dim (0 disables PLE); num_global_kv_heads and
  # global_key_size for GLOBAL layers; attention_pattern (None uses
  # GEMMA4_ATTENTION_PATTERN); frac_shared_layers (fraction of trailing layers
  # that share KV); global/local rope proportions; k_eq_v_global;
  # override_kv_shared_ffw_hidden; RoPE base frequencies (10k local, 1M global)
  # and scale factors; shd_config; remat_config; param_dtype and dtype (both
  # float32 by default); use_flash_attention and flash_attention_block_size;
  # use_sliding_window_kv_cache; MoE settings (enable_moe, num_experts,
  # num_experts_per_tok, expert_dim, moe_dense_hidden_dim).
  """Transformer config."""

  num_layers: int
  num_embed: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  final_logit_softcap: float = 30.0
  sliding_window_size: int | None = None
  per_layer_input_dim: int = 0
  num_global_kv_heads: int | None = None
  global_key_size: int = 512
  attention_pattern: tuple['AttentionType', ...] | None = None
  frac_shared_layers: float = 0.0
  global_rope_proportion: float = 0.25
  local_rope_proportion: float = 1.0
  k_eq_v_global: bool = False
  override_kv_shared_ffw_hidden: int | None = None

  local_base_frequency: int = 10_000
  global_base_frequency: int = 1_000_000
  local_scale_factor: float = 1.0
  global_scale_factor: float = 1.0

  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE
  param_dtype: jnp.dtype = jnp.float32
  dtype: jnp.dtype = jnp.float32
  use_flash_attention: bool = False
  flash_attention_block_size: int = 1024
  use_sliding_window_kv_cache: bool = True

  # MoE config
  enable_moe: bool = False
  num_experts: int | None = None
  num_experts_per_tok: int | None = None
  expert_dim: int | None = None
  moe_dense_hidden_dim: int | None = None

  def __post_init__(self):
    # [FUNCTION DOC]
    # Dataclass post-init validation: raises ValueError when both
    # use_sliding_window_kv_cache and use_flash_attention are True.
    # TODO(tunix-dev): support flash attention with sliding window KV cache
    if self.use_sliding_window_kv_cache and self.use_flash_attention:
      raise ValueError(
          'Flash attention and sliding window KV cache are mutually exclusive.'
      )

  @classmethod
  def gemma4_e2b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: Gemma 4 E2B config: 35 layers, vocab 262144, embed 1536,
    # hidden 6144, 8 heads, head_dim 256, 1 kv head, window 512, PLE dim 256,
    # 20 of 35 layers KV-shared with FFN width 12288 for those layers, pattern
    # 4 local + 1 global. Other fields default.
    return cls(
        num_layers=35,
        num_embed=262144,
        embed_dim=1536,
        hidden_dim=1536 * 4,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        sliding_window_size=512,
        shd_config=sharding_config,
        per_layer_input_dim=256,
        frac_shared_layers=20.0 / 35,
        override_kv_shared_ffw_hidden=int(1536 * 4 * 2),
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
    )

  @classmethod
  def gemma4_e4b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: Gemma 4 E4B config: 42 layers, vocab 262144, embed 2560,
    # hidden 10240, 8 heads, head_dim 256, 2 kv heads, window 512, PLE dim 256,
    # 18 of 42 layers KV-shared, pattern 5 local + 1 global.
    return cls(
        num_layers=42,
        num_embed=262144,
        embed_dim=2560,
        hidden_dim=2560 * 4,
        num_heads=8,
        head_dim=256,
        num_kv_heads=2,
        sliding_window_size=512,
        shd_config=sharding_config,
        per_layer_input_dim=256,
        frac_shared_layers=18.0 / 42,
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
    )

  @classmethod
  def gemma4_31b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: Gemma 4 31B dense config: 60 layers, vocab 262144, embed
    # 5376, hidden 21504, 32 heads, head_dim 256, 16 kv heads (4 for global
    # layers), window 1024, global layers share k and v projections, pattern
    # 5 local + 1 global.
    return cls(
        num_layers=60,
        num_embed=262144,
        embed_dim=5376,
        hidden_dim=5376 * 4,
        num_heads=32,
        head_dim=256,
        num_kv_heads=16,
        num_global_kv_heads=4,
        sliding_window_size=1024,
        shd_config=sharding_config,
        k_eq_v_global=True,
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
    )

  @classmethod
  def gemma4_26b_a4b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: Gemma 4 26B-A4B MoE config: 30 layers, vocab 262144, embed
    # 2816, dense MLP hidden 2112, 16 heads, head_dim 256, 8 kv heads (2 global),
    # window 1024, MoE with 128 experts of width 704 and 8 experts per token,
    # k_eq_v for global layers, global rope proportion 0.25, pattern 5 local + 1
    # global.
    return cls(
        num_layers=30,
        num_embed=262144,
        embed_dim=2816,
        hidden_dim=2112,  # Dense shared MLP branch
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        num_global_kv_heads=2,
        sliding_window_size=1024,
        shd_config=sharding_config,
        enable_moe=True,
        num_experts=128,
        expert_dim=704,
        num_experts_per_tok=8,
        moe_dense_hidden_dim=2112,
        k_eq_v_global=True,
        global_rope_proportion=0.25,
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
    )


class Embedder(nnx.Module):
  # [CLASS DOC]
  # Token embedder with tied unembedding, and (when per_layer_input_dim > 0)
  # the per-layer-input machinery: per_layer_model_projection (Einsum
  # 'BTD,DNP->BTNP', weight [D, num_layers, P], weights scaled by D^-0.5 at
  # use time), per_layer_projection_norm (RMSNorm over P) and
  # per_layer_input_embedding table [vocab, num_layers, P].
  """Embedder module."""

  def __init__(
      self,
      config: ModelConfig,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Stores config, vocab_size, embed_dim, param_dtype and creates the
    # parameters listed in the class doc (normal init in param_dtype, sharding
    # from config.shd_config).
    self.config = config
    self.vocab_size = config.num_embed
    self.embed_dim = config.embed_dim
    self.param_dtype = config.param_dtype

    self.input_embedding = nnx.Param(
        nnx.initializers.normal(dtype=self.param_dtype)(
            rngs.params(), (self.vocab_size, self.embed_dim)
        ),
        sharding=config.shd_config.emb_vd,
    )

    if config.per_layer_input_dim > 0:
      self.per_layer_model_projection = Einsum(
          einsum_str='BTD,DNP->BTNP',
          shape=(self.embed_dim, config.num_layers, config.per_layer_input_dim),
          sharding=config.shd_config.per_layer_model_projection,
          w_scale=(float(self.embed_dim) ** -0.5),
          rngs=rngs,
          dtype=self.config.dtype,
          param_dtype=self.param_dtype,
      )

      self.per_layer_projection_norm = RMSNorm(
          config.per_layer_input_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=self.config.dtype,
          param_dtype=self.param_dtype,
      )
      self.per_layer_input_embedding = nnx.Param(
          nnx.initializers.normal(dtype=self.param_dtype)(
              rngs.params(),
              (self.vocab_size, config.num_layers, config.per_layer_input_dim),
          ),
          sharding=config.shd_config.per_layer_input_embedding,
      )

  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Looks up tokens x [B, T], multiplies by sqrt(embed_dim) in the table dtype,
    # casts to config.dtype and applies act_btd sharding. Returns [B, T, D].
    x = self.input_embedding[(x,)]
    x *= jnp.sqrt(x.shape[-1]).astype(x.dtype)
    x = jnp.astype(x, self.config.dtype)
    x = shard(x, self.config.shd_config.act_btd)
    return x

  def encode_per_layer_input(
      self, x: jaxtyping.ArrayLike, t: jaxtyping.ArrayLike
  ) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Computes per-layer inputs. Inputs: x = token embeddings [B, T, D] (output
    # of encode), t = token ids [B, T]. Out-of-range ids are replaced by 0.
    # Projects x to [B, T, num_layers, P] and RMS-normalizes over P; looks up
    # per_layer_input_embedding[t] and scales it by sqrt(P); returns
    # (projection + embedding) / sqrt(2), shape [B, T, num_layers, P].
    t = jnp.where(
        jnp.logical_and(t >= 0, t < self.vocab_size), t, jnp.zeros_like(t)
    )
    x = self.per_layer_model_projection(x)
    x = self.per_layer_projection_norm(x)
    y = self.per_layer_input_embedding.value[t]
    y *= jnp.sqrt(self.config.per_layer_input_dim).astype(y.dtype)
    return (x + y) * jax.lax.rsqrt(2.0).astype(x.dtype)

  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Casts x and the embedding table to config.dtype and returns logits
    # x @ table^T ([..., vocab]).
    x = jnp.astype(x, self.config.dtype)
    w = jnp.astype(self.input_embedding.value, self.config.dtype)
    return jnp.dot(x, w.T)


class Einsum(nnx.Module):
  # [CLASS DOC]
  # Parameterized einsum with explicit compute dtype and optional constant
  # weight multiplier w_scale.
  """Einsum module."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
      w_scale: float | None = None,
  ):
    # [FUNCTION DOC]
    # Stores einsum_str, dtype, w_scale and shape; creates weight w with normal
    # init in param_dtype and the given sharding.
    self.einsum_str = einsum_str
    self.dtype = dtype
    self.w_scale = w_scale

    self.shape = shape
    self.w = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape),
        sharding=sharding,
    )

  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Multiplies w by w_scale if set, casts x and w to dtype and returns
    # einsum(einsum_str, x, w).
    w = self.w.value
    if self.w_scale is not None:
      w = w * self.w_scale
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(w, self.dtype)
    return jnp.einsum(self.einsum_str, x, w)


def find_last_one_index(attn_mask: jnp.ndarray) -> jnp.ndarray:
  # [FUNCTION DOC]
  # For attn_mask [B, 1, S] returns [B] with the index of the last true entry
  # per row (0 for all-false rows); computed with argmax on the reversed rows
  # then squeezing the query axis (errors if that axis is not 1).
  """Finds the index of the last (rightmost) '1' from attn_mask."""
  cache_len = attn_mask.shape[-1]

  # 1. check if the entire row is all zeros.
  all_zeros_mask = jnp.all(attn_mask == 0, axis=-1)

  # 2. reverse the rows in the attn_mask
  reversed_matrix = attn_mask[:, :, ::-1]

  # 3. find the fist 1 from the right.
  first_one_from_right = jnp.argmax(reversed_matrix, axis=-1)

  # 4. covert back to the original index
  last_one_index_original = cache_len - 1 - first_one_from_right

  # 5. return the final index, 0 for rows are all zeros.
  final_indices = jnp.where(
      all_zeros_mask,
      0,
      last_one_index_original,
  )

  return final_indices.squeeze(axis=-1)


def create_sliding_window_mask(
    attn_mask: jnp.ndarray,  # [B, seq_len, cache_len] seq_len=1 for decoding
    sliding_window_size: int,
) -> jnp.ndarray:
  # [FUNCTION DOC]
  # HELPER: decode-time sliding window mask over cache slot indices (no ring
  # buffer): [B, 1, S] true for slots in [last - window + 1, last] where last
  # is find_last_one_index(attn_mask). Used when the sliding window cache is
  # disabled.
  """Helper function to create sliding window mask for local attention."""
  upper_index = find_last_one_index(attn_mask)

  # 1. compute the window start position
  window_start_pos = upper_index - sliding_window_size + 1

  # 2. create window mask
  abs_pos = jnp.arange(attn_mask.shape[-1])
  window_mask = abs_pos[None, :] >= window_start_pos[:, None]

  # 3. create causal mask
  causal_mask = abs_pos[None, :] <= upper_index[:, None]

  # 4. create final mask
  final_mask = window_mask & causal_mask
  return final_mask[:, None, :]  # [B, 1, cache_len]


class RMSNorm(nnx.Module):
  # [CLASS DOC]
  # RMSNorm computed in float32 with epsilon 1e-6 and a learned scale
  # initialized to ones and applied directly (no "1 +" offset); output cast to
  # the configured dtype.
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      rngs: nnx.Rngs,
      sharding: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    # [FUNCTION DOC]
    # Creates scale [dim] of ones in param_dtype, sharded with
    # sharding.rms_norm_weight (sharding is a whole ShardingConfig), and stores
    # dtype.
    self.scale = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), dim).astype(param_dtype),
        sharding=sharding.rms_norm_weight,
    )
    self.dtype = dtype

  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Casts x to float32, multiplies by rsqrt(mean(x^2) + 1e-6) and by scale
    # (broadcast over leading dims), and casts to dtype.
    x = jnp.astype(x, jnp.float32)
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06).astype(x.dtype)
    scale = jnp.expand_dims(self.scale.value, axis=range(len(x.shape) - 1))
    normed_inputs = normed_inputs * scale
    return normed_inputs.astype(self.dtype)


def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
    rope_proportion: float = 1.0,
) -> jax.Array:
  # [FUNCTION DOC]
  # Applies half-split RoPE to inputs [B, L, N, H] at positions [B, L] with
  # partial rotation: of the H/2 frequency pairs, the first
  # int(rope_proportion * H / 2) use timescales base_frequency^(2i/H); the rest
  # get infinite timescale (angle 0, i.e. left unrotated). Angles are divided
  # by scale_factor. Raises ValueError if scale_factor < 1. Returns same shape
  # and dtype as inputs.
  """Applies RoPE.

  Let B denote batch size, L denote sequence length, N denote number of heads,
  and H denote head dimension. Note that H must be divisible by 2.

  Args:
    inputs: Array of shape [B, L, N, H].
    positions:  Array of shape [B, L].
    base_frequency: Base frequency used to compute rotations.
    scale_factor: The scale factor used for positional interpolation, allowing
      an expansion of sequence length beyond the pre-trained context length.
    rope_proportion: The proportion of the head dimension to apply RoPE to.

  Returns:
    Array of shape [B, L, N, H].
  """
  head_dim = inputs.shape[-1]
  rope_angles = int(rope_proportion * head_dim // 2)
  nope_angles = head_dim // 2 - rope_angles
  freq_exponents = (2.0 / head_dim) * jnp.arange(
      0, rope_angles, dtype=jnp.float32
  )
  timescale = jnp.pad(
      base_frequency**freq_exponents,
      (0, nope_angles),
      mode='constant',
      constant_values=(0, jnp.inf),
  )

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  if scale_factor < 1.0:
    raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')
  sinusoid_inp /= scale_factor

  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)


K_MASK = -2.3819763e38


class AttentionType(enum.Enum):
  # [CLASS DOC]
  # Enum of attention kinds: GLOBAL (=1) full causal attention, LOCAL_SLIDING
  # (=2) windowed attention.
  GLOBAL = 1
  LOCAL_SLIDING = 2


GEMMA4_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def create_kv_cache_sharing_patterns(
    num_layers: int,
    frac_shared_layers: float,
    share_global: bool,
    share_local: bool,
    attention_types: tuple[AttentionType, ...],
) -> list[int]:
  # [FUNCTION DOC]
  # Returns, for each layer i, the index of the layer whose K/V it uses.
  # The first int(num_layers - frac_shared_layers * num_layers) layers use
  # their own (index i). Each later layer maps to the last unshared layer
  # (index num_unshared - 1) if GLOBAL and share_global, to the second-to-last
  # unshared layer (num_unshared - 2) if LOCAL_SLIDING and share_local,
  # otherwise to itself.
  # KNOWN ISSUE: assumes the last unshared layer is GLOBAL and the one before
  # it is LOCAL_SLIDING; other patterns would share K/V across mismatched
  # attention types. The docstring wording ("for which KV cache is used") is
  # vague.
  """Creates a list of layer indices for which KV cache is used."""
  kv_cache_sharing_patterns = []
  num_unshared_layers = int(num_layers - frac_shared_layers * num_layers)
  for i in range(num_layers):
    if i < num_unshared_layers:
      kv_cache_sharing_patterns.append(i)
    else:
      if attention_types[i] == AttentionType.GLOBAL and share_global:
        kv_cache_sharing_patterns.append(num_unshared_layers - 1)
      elif attention_types[i] == AttentionType.LOCAL_SLIDING and share_local:
        kv_cache_sharing_patterns.append(num_unshared_layers - 2)
      else:
        kv_cache_sharing_patterns.append(i)
  return kv_cache_sharing_patterns


class Attention(nnx.Module):
  # [CLASS DOC]
  # Gemma 4 attention layer. Configured per attention type: RoPE base
  # frequency, scale factor and rope proportion; for GLOBAL layers kv heads
  # and head_dim may be overridden by num_global_kv_heads / global_key_size.
  # Always separate q_einsum; keys/values come from k_einsum (k_eq_v: values
  # equal keys) or kv_einsum. Supports KV caches (full or ring buffer for local
  # layers), reading another layer's K/V (kv sharing), naive einsum attention
  # (MHA/MQA/GQA) or TPU splash attention.
  """Attention module."""

  def __init__(
      self,
      config: ModelConfig,
      attn_type: AttentionType,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Computes rope_proportion, rope_base_frequency and rope_scale_factor from
    # config and attn_type; sets num_kv_heads and head_dim (global overrides when
    # GLOBAL; global_key_size is always non-None by default, so global layers use
    # 512 unless overridden). Creates attn_vec_einsum [N, H, D], q_einsum
    # [N, D, H], then either k_einsum [K, D, H] (when k_eq_v_global and GLOBAL;
    # sharded like q weights) or kv_einsum [2, K, D, H], and the QK RMSNorms.
    self.config = config
    self.rope_proportion = (
        config.global_rope_proportion
        if attn_type == AttentionType.GLOBAL
        else config.local_rope_proportion
    )
    self.attn_type = attn_type
    self.rope_base_frequency = (
        config.local_base_frequency
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_base_frequency
    )
    self.rope_scale_factor = (
        config.local_scale_factor
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_scale_factor
    )

    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    if attn_type == AttentionType.GLOBAL:
      if config.num_global_kv_heads is not None:
        self.num_kv_heads = config.num_global_kv_heads
      if config.global_key_size is not None:
        self.head_dim = config.global_key_size

    self.attn_vec_einsum = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, self.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=config.shd_config.o_weight_nhd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.q_einsum = Einsum(
        einsum_str='BTD,NDH->BTNH',
        shape=(config.num_heads, config.embed_dim, self.head_dim),
        rngs=rngs,
        sharding=config.shd_config.q_weight_ndh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    k_eq_v = (
        config.k_eq_v_global if attn_type == AttentionType.GLOBAL else False
    )
    if k_eq_v:
      self.k_einsum = Einsum(
          einsum_str='BSD,KDH->BSKH',
          shape=(
              self.num_kv_heads,
              config.embed_dim,
              self.head_dim,
          ),
          rngs=rngs,
          sharding=config.shd_config.q_weight_ndh,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    else:
      self.kv_einsum = Einsum(
          einsum_str='BSD,CKDH->CBSKH',
          shape=(
              2,
              self.num_kv_heads,
              config.embed_dim,
              self.head_dim,
          ),
          rngs=rngs,
          sharding=(None, None, 'fsdp', None)
          if self.num_kv_heads == 1
          else config.shd_config.kv_weight_cndh,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    self._query_norm = RMSNorm(
        self.head_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self._key_norm = RMSNorm(
        self.head_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      kv_shared_cache: LayerCache | None = None,
  ) -> tuple[
      LayerCache | None,
      jaxtyping.Array,
      tuple[jaxtyping.Array, jaxtyping.Array],
  ]:
    # [FUNCTION DOC]
    # Attention computation.
    # Inputs: x [B, T, D]; segment_pos [B, T]; cache (LayerCache or None);
    # attn_mask bool [B, T, S]; kv_shared_cache (dict with 'k', 'v' and, during
    # decode, 'end_index') to reuse another layer's K/V instead of computing them.
    # Steps: q = RoPE(QKnorm(q_einsum(x))). If kv_shared_cache is given (cache
    # must be None, asserted), take k/v from it; else compute k (and v = k for
    # k_eq_v, else from kv_einsum), shard, RMS-normalize v without scale,
    # normalize k with _key_norm and apply RoPE.
    # Cache update when cache is given: prefill (T > 1) writes the last
    # min(T, cache_len) k/v into slots (position modulo cache_len) for the ring
    # cache, or into slots [0, T) otherwise, keeps attention over the fresh
    # k/v, and sets end_index += T; decode (T == 1) writes at end_index[0] modulo
    # cache_len and attends over the whole cache. Without a cache, new_cache holds
    # only the fresh 'k' and 'v'.
    # Attention: if use_flash_attention and T > 1, runs a splash attention kernel
    # (LocalMask with window (W-1, 0) or CausalMask over T x T, one per head)
    # under shard_map using act_btnh mesh axes. Otherwise computes logits (grouped
    # for GQA), truncates the mask to the first T keys during prefill, and for
    # LOCAL_SLIDING layers: in decode with the ring cache, remaps mask columns
    # from logical positions to physical ring slots (using end_index, minus 1
    # when reading a shared cache already advanced by its origin layer) and
    # masks never-written slots; in decode without ring cache uses
    # create_sliding_window_mask; in prefill applies a |i - j| < W band mask.
    # Then masks with K_MASK, softmax in float32, aggregates values, projects
    # with attn_vec_einsum and applies act_btd sharding.
    # Output: (new_cache, attn_output [B, T, D], (keys, values) used).
    # KNOWN ISSUE: prefill always writes the non-ring cache from slot 0 and
    # attends only to the fresh tokens (no chunked prefill); the splash path
    # ignores attn_mask (padding is not masked); decode uses end_index[0] for
    # the whole batch.
    x = x.astype(self.config.dtype)
    seq_len = x.shape[1]
    query_proj = self.q_einsum(x)
    query_proj = shard(query_proj, self.config.shd_config.act_btnh)
    query_proj = self._query_norm(query_proj)
    query_proj = apply_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
        rope_proportion=self.rope_proportion,
    )

    if kv_shared_cache is not None:
      assert cache is None
      key_proj = kv_shared_cache['k']
      value_proj = kv_shared_cache['v']
    else:
      if hasattr(self, 'k_einsum'):  # case where k_eq_v is True
        key_proj = self.k_einsum(x)
        value_proj = key_proj
      else:
        key_proj, value_proj = self.kv_einsum(x)

      key_proj = shard(key_proj, self.config.shd_config.act_btnh)
      value_proj = shard(value_proj, self.config.shd_config.act_btnh)

      # Apply norms to computed KV
      value_var = jnp.mean(jnp.square(value_proj), axis=-1, keepdims=True)
      value_proj = value_proj * jax.lax.rsqrt(value_var + 1e-06)
      key_proj = self._key_norm(key_proj)
      key_proj = apply_rope(
          key_proj,
          segment_pos,
          base_frequency=self.rope_base_frequency,
          scale_factor=self.rope_scale_factor,
          rope_proportion=self.rope_proportion,
      )

    if cache is not None:
      assert kv_shared_cache is None
      # Update cache with new kv projections
      cache_len = cache['v'].shape[1]
      if seq_len > 1:  # prefill
        if self.config.use_sliding_window_kv_cache:
          # Sliding window cache update (prefill).
          # Does not support chunked prefill.
          valid_len = min(seq_len, cache_len)
          latest_indices = jnp.arange(seq_len - valid_len, seq_len) % cache_len
          cache_v = (
              cache['v']
              .at[:, latest_indices, ...]
              .set(value_proj[:, -valid_len:, ...])
          )
          cache_k = (
              cache['k']
              .at[:, latest_indices, ...]
              .set(key_proj[:, -valid_len:, ...])
          )
        else:
          cache_v = cache['v'].at[:, :seq_len, ...].set(value_proj)
          cache_k = cache['k'].at[:, :seq_len, ...].set(key_proj)

        new_cache = {
            'v': cache_v,
            'k': cache_k,
            'end_index': cache['end_index'] + seq_len,
        }
      else:  # decode
        end_index = cache['end_index'][0]
        slice_indices = (0, end_index % cache_len, 0, 0)
        value_proj = jax.lax.dynamic_update_slice(
            cache['v'], value_proj, slice_indices
        )
        key_proj = jax.lax.dynamic_update_slice(
            cache['k'], key_proj, slice_indices
        )
        new_cache = {
            'v': value_proj,
            'k': key_proj,
            'end_index': cache['end_index'] + seq_len,
        }
    else:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
      }

    _, _, qh, _ = query_proj.shape

    if self.config.use_flash_attention and seq_len > 1:
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)

      mesh = pxla.thread_resources.env.physical_mesh
      if self.attn_type == AttentionType.LOCAL_SLIDING:
        mask = mask_lib.LocalMask(
            (seq_len, seq_len),
            window_size=(self.config.sliding_window_size - 1, 0),
            offset=0,
        )
      else:
        mask = mask_lib.CausalMask((seq_len, seq_len))

      multi_head_mask = mask_lib.MultiHeadMask([mask for _ in range(qh)])

      block_sizes = splash.BlockSizes(
          block_q=self.config.flash_attention_block_size,
          block_kv=self.config.flash_attention_block_size,
          block_q_dkv=self.config.flash_attention_block_size,
          block_kv_dkv=self.config.flash_attention_block_size,
          block_kv_dkv_compute=self.config.flash_attention_block_size,
          block_q_dq=self.config.flash_attention_block_size,
          block_kv_dq=self.config.flash_attention_block_size,
      )

      shd_b, shd_t, shd_n, shd_h = self.config.shd_config.act_btnh
      head_shards = (
          mesh.shape[shd_n] if shd_n is not None and shd_n in mesh.shape else 1
      )
      q_seq_shards = (
          mesh.shape[shd_t] if shd_t is not None and shd_t in mesh.shape else 1
      )

      splash_attn_kernel = splash.make_splash_mha(
          multi_head_mask,
          block_sizes=block_sizes,
          head_shards=head_shards,
          q_seq_shards=q_seq_shards,
      )

      shd_spec = P(shd_b, shd_n, shd_t, shd_h)
      unsharded_seq = P(shd_b, shd_n, None, shd_h)
      kernel_spec = splash_attn_kernel.manual_sharding_spec(
          shd.NamedSharding(mesh, P(shd_n, shd_t))
      )

      @partial(
          shard_map,
          mesh=mesh,
          in_specs=(kernel_spec, shd_spec, unsharded_seq, unsharded_seq),
          out_specs=shd_spec,
          check_rep=False,
      )
      def sharded_splash_attn(kernel, q_block, k_block, v_block):
        # [FUNCTION DOC]
        # HELPER: per-shard body for shard_map; vmaps the splash kernel over the
        # batch of q/k/v blocks ([B, N, T, H]) and returns the attention output.
        return jax.vmap(kernel)(q_block, k_block, v_block)

      qkv: jaxtyping.Array = sharded_splash_attn(
          splash_attn_kernel, query_proj, key_proj, value_proj
      )
      encoded = qkv.transpose(0, 2, 1, 3)
    else:
      if self.use_gqa:
        b, t, kg, h = query_proj.shape
        n_groups = kg // self.num_kv_heads
        query_reshaped = query_proj.reshape(
            (b, t, self.num_kv_heads, n_groups, h)
        )
        logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_reshaped, key_proj)
        b, t, k, g, s = logits.shape
        logits = logits.reshape((b, t, k * g, s))
      else:
        logits = jnp.einsum('BTNH,BSNH->BTNS', query_proj, key_proj)

      if seq_len > 1:
        # Only compute attention scores for the actual sequence length.
        attn_mask = attn_mask[..., :seq_len]

      if self.attn_type == AttentionType.LOCAL_SLIDING:
        if (
            segment_pos.shape[1] == 1
            and self.config.use_sliding_window_kv_cache
        ):
          # for decoding with sliding window cache
          active_cache = cache if cache is not None else kv_shared_cache
          if active_cache is None:
            raise ValueError(
                'Cache or shared cache is required for local sliding attention'
                ' in decoding.'
            )
          cache_len = key_proj.shape[1]
          end_idx = active_cache['end_index']
          if cache is None and kv_shared_cache is not None:
            # In case of shared KV cache, the origin layer already updated the
            # end index. We need to subtract 1 to get the correct end index of
            # the previous token.
            end_idx = end_idx - 1
          end_idx = end_idx[:, None, None]
          p = jnp.arange(cache_len)[None, None, :]

          # map physical index to logical index
          logical_indices = end_idx - ((end_idx - p) % cache_len)

          # identify uninitialized slots (before the cache fills up)
          valid_physical = logical_indices >= 0
          logical_indices = jnp.maximum(0, logical_indices)

          attn_mask = jnp.take_along_axis(attn_mask, logical_indices, axis=-1)
          attn_mask = attn_mask * valid_physical
        elif segment_pos.shape[1] == 1:
          # for decoding without sliding window cache
          sliding_mask = create_sliding_window_mask(
              attn_mask,
              sliding_window_size=self.config.sliding_window_size,
          )
          attn_mask = sliding_mask * attn_mask
        else:  # for prefill
          all_ones = jnp.ones_like(attn_mask)
          sliding_mask = jnp.triu(
              all_ones, -1 * self.config.sliding_window_size + 1
          ) * jnp.tril(all_ones, self.config.sliding_window_size - 1)
          attn_mask = sliding_mask * attn_mask

      attn = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
      attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
          key_proj.dtype
      )

      if self.use_gqa:
        b, t, kg, s = attn.shape
        n_groups = kg // self.num_kv_heads
        probs_reshaped = attn.reshape((b, t, self.num_kv_heads, n_groups, s))
        encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs_reshaped, value_proj)
        b, t, k, g, h = encoded.shape
        encoded = encoded.reshape((b, t, k * g, h))
      else:
        encoded = jnp.einsum('BTNS,BSNH->BTNH', attn, value_proj)

    attn_output = self.attn_vec_einsum(encoded)
    attn_output = shard(attn_output, self.config.shd_config.act_btd)
    return new_cache, attn_output, (key_proj, value_proj)

  @property
  def use_gqa(self):
    # [FUNCTION DOC]
    # Property: True when this layer's kv head count differs from num_heads and
    # is greater than 1 (grouped-query attention).
    return self.num_kv_heads != self.config.num_heads and self.num_kv_heads > 1

  def __call__(self, x, segment_pos, cache, attn_mask, kv_shared_cache=None):
    # [FUNCTION DOC]
    # Calls block, wrapped in nnx.remat (graph_updates=False) when the config's
    # remat_config is BLOCK (enum or value). Returns block's 3-tuple.
    remat_config = getattr(self.config, 'remat_config', RematConfig.NONE)
    if (
        remat_config == RematConfig.BLOCK
        or remat_config == RematConfig.BLOCK.value
    ):
      # nnx.remat needs to be applied to the unbound function and take self
      # as the first argument. graph_updates=False prevents TraceContextError
      # when mutating params across jax transformation trace levels.
      return nnx.remat(self.block.__func__, graph_updates=False)(
          self, x, segment_pos, cache, attn_mask, kv_shared_cache
      )
    else:
      return self.block(
          x, segment_pos, cache, attn_mask, kv_shared_cache=kv_shared_cache
      )

  def init_cache(self, batch_size, max_seq_len, dtype):
    # [FUNCTION DOC]
    # Creates this layer's zero cache: cache_len = max_seq_len, or
    # min(max_seq_len, sliding_window_size) for LOCAL_SLIDING layers when the
    # ring cache is enabled. 'k' and 'v' are [batch, cache_len, K, H] numpy
    # zeros of dtype eagerly placed with act_btnh sharding; 'end_index' int32
    # zeros [batch] sharded on the batch axis. Returns the dict.
    cache_len = max_seq_len
    if (
        self.config.use_sliding_window_kv_cache
        and self.attn_type == AttentionType.LOCAL_SLIDING
        and self.config.sliding_window_size is not None
    ):
      cache_len = min(max_seq_len, self.config.sliding_window_size)

    cache_shape = (batch_size, cache_len, self.num_kv_heads, self.head_dim)
    k = shard(
        np.zeros(cache_shape, dtype),
        self.config.shd_config.act_btnh,
        eager=True
    )
    v = shard(
        np.zeros(cache_shape, dtype),
        self.config.shd_config.act_btnh,
        eager=True,
    )
    end_index = shard(
        np.zeros((batch_size,), np.int32),
        self.config.shd_config.act_btnh[:1],
        eager=True,
    )
    return {'k': k, 'v': v, 'end_index': end_index}


class FeedForward(nnx.Module):
  # [CLASS DOC]
  # Gated GELU MLP (gate_proj, up_proj, down_proj; bias-free, zero init),
  # with optional hidden width override.
  """Feed forward module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      hidden_dim: int | None = None,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Creates the three nnx.Linear layers with width hidden_dim (or
    # config.hidden_dim if None), compute dtype config.dtype and param_dtype.
    self.config = config
    h_dim = hidden_dim if hidden_dim is not None else config.hidden_dim
    self.gate_proj = nnx.Linear(
        config.embed_dim,
        h_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            config.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    self.up_proj = nnx.Linear(
        config.embed_dim,
        h_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            config.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.down_proj = nnx.Linear(
        h_dim,
        config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(), config.shd_config.ffw_weight_fd
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def block(self, x):
    # [FUNCTION DOC]
    # Returns down_proj(gelu(gate_proj(x)) * up_proj(x)).
    return self.down_proj(nnx.gelu(self.gate_proj(x)) * self.up_proj(x))

  def __call__(self, x):
    # [FUNCTION DOC]
    # Calls block, wrapped in nnx.remat when remat_config is BLOCK.
    remat_config = getattr(self.config, 'remat_config', RematConfig.NONE)
    if (
        remat_config == RematConfig.BLOCK
        or remat_config == RematConfig.BLOCK.value
    ):
      return nnx.remat(self.block.__func__, graph_updates=False)(self, x)
    else:
      return self.block(x)


class DecoderLayer(nnx.Module):
  # [CLASS DOC]
  # Gemma 4 decoder layer: pre-norm attention with post-norm and residual;
  # dense MLP (plus, if MoE is enabled, a parallel MoE branch with its own
  # pre/post norms, summed with the post-normed dense output) followed by
  # post_ffw_norm and residual; optional per-layer-input gated update; final
  # multiplication by the learned scalar skip_scale.
  """Decoder layer."""

  def __init__(
      self,
      config: ModelConfig,
      attn_type: AttentionType,
      *,
      hidden_dim: int | None = None,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Creates pre_attention_norm, attn, post_attention_norm, pre_ffw_norm, mlp
    # (width hidden_dim or config default), MoE modules when enable_moe
    # (moe_pre_ffw_norm, moe, moe_post_ffw_norm, dense_post_ffw_norm),
    # post_ffw_norm, PLE modules when per_layer_input_dim > 0
    # (per_layer_input_gate [D, P], per_layer_projection [P, D],
    # post_per_layer_input_norm), and skip_scale (ones [1]).

    self.config = config
    self.pre_attention_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    self.attn = Attention(
        config=config,
        attn_type=attn_type,
        rngs=rngs,
    )
    self.post_attention_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.pre_ffw_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.mlp = FeedForward(config=config, hidden_dim=hidden_dim, rngs=rngs)

    if config.enable_moe:
      self.moe_pre_ffw_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
      self.moe = moe.MoERagged(
          config=config,
          rngs=rngs,
      )
      self.moe_post_ffw_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
      self.dense_post_ffw_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    self.post_ffw_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    if config.per_layer_input_dim > 0:

      self.per_layer_input_gate = Einsum(
          einsum_str='BTD,DP->BTP',
          shape=(config.embed_dim, config.per_layer_input_dim),
          sharding=config.shd_config.per_layer_input_gate,
          rngs=rngs,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

      self.per_layer_projection = Einsum(
          einsum_str='BTP,PD->BTD',
          shape=(config.per_layer_input_dim, config.embed_dim),
          sharding=config.shd_config.per_layer_projection,
          rngs=rngs,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

      self.post_per_layer_input_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

    self.skip_scale = nnx.Param(jnp.ones((1,), dtype=config.param_dtype))

  def block(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
      per_layer_input=None,
      kv_shared_cache=None,
  ):
    # [FUNCTION DOC]
    # Inputs: x [B, T, D], segment_pos, cache, attn_mask, per_layer_input
    # [B, T, P] or None, kv_shared_cache.
    # h = post_attention_norm(attn(pre_attention_norm(x))) + x.
    # f = mlp(pre_ffw_norm(h)); with MoE f = dense_post_ffw_norm(f) +
    # moe_post_ffw_norm(moe(moe_pre_ffw_norm(h))). out = post_ffw_norm(f) + h.
    # With PLE: out += post_per_layer_input_norm(per_layer_projection(
    # gelu(per_layer_input_gate(out)) * per_layer_input)). out *= skip_scale.
    # Returns (cache, out, (k, v)).
    norm = self.pre_attention_norm(x)
    cache, attn, kv = self.attn(
        norm, segment_pos, cache, attn_mask, kv_shared_cache=kv_shared_cache
    )
    attn = self.post_attention_norm(attn)
    attn += x

    norm_ffw = self.pre_ffw_norm(attn)
    ffw = self.mlp(norm_ffw)
    if self.config.enable_moe:
      ffw = self.dense_post_ffw_norm(ffw)
      moe_norm_ffw = self.moe_pre_ffw_norm(attn)
      moe_out = self.moe(moe_norm_ffw)
      moe_out = self.moe_post_ffw_norm(moe_out)
      ffw += moe_out
    ffw = self.post_ffw_norm(ffw)

    ffw += attn

    if self.config.per_layer_input_dim > 0 and per_layer_input is not None:
      gating_input = ffw
      mapped = self.per_layer_input_gate(gating_input)
      mapped = jax.nn.gelu(mapped) * per_layer_input
      mapped = self.per_layer_projection(mapped)
      mapped = self.post_per_layer_input_norm(mapped)
      ffw += mapped

    ffw = ffw * self.skip_scale.value
    return cache, ffw, kv

  def __call__(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
      per_layer_input=None,
      kv_shared_cache=None,
  ):
    # [FUNCTION DOC]
    # Calls block, wrapped in nnx.remat when remat_config is DECODER.
    remat_config = getattr(self.config, 'remat_config', RematConfig.NONE)
    if (
        remat_config == RematConfig.DECODER
        or remat_config == RematConfig.DECODER.value
    ):
      return nnx.remat(self.block.__func__, graph_updates=False)(
          self,
          x,
          segment_pos,
          cache,
          attn_mask,
          per_layer_input,
          kv_shared_cache,
      )
    else:
      return self.block(
          x, segment_pos, cache, attn_mask, per_layer_input, kv_shared_cache
      )

  def init_cache(self, batch_size, max_seq_len, dtype):
    # [FUNCTION DOC]
    # Delegates to attn.init_cache.
    return self.attn.init_cache(batch_size, max_seq_len, dtype)


class Gemma4(BackendMappingMixin, nnx.Module):
  # [CLASS DOC]
  # Full Gemma 4 model: Embedder, DecoderLayers following the attention
  # pattern, KV sharing map, final RMSNorm, tied unembedding and logit
  # soft-capping. Mixes in BackendMappingMixin.
  """Gemma4 model."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    # [FUNCTION DOC]
    # Builds the embedder; derives per-layer attention types by cycling the
    # configured (or default) pattern; computes kv_cache_sharing_patterns
    # (sharing both global and local) and shared_layer_origins (set of layers
    # whose K/V are reused); creates each DecoderLayer (shared layers use
    # override_kv_shared_ffw_hidden when set) and the final norm.
    self.config = config
    self.embedder = Embedder(config, rngs=rngs)

    pattern = (
        config.attention_pattern
        if config.attention_pattern
        else GEMMA4_ATTENTION_PATTERN
    )
    attention_types = [
        attn_type
        for _, attn_type in zip(
            range(config.num_layers), itertools.cycle(pattern)
        )
    ]
    self.kv_cache_sharing_patterns = create_kv_cache_sharing_patterns(
        num_layers=config.num_layers,
        frac_shared_layers=config.frac_shared_layers,
        share_global=True,
        share_local=True,
        attention_types=tuple(attention_types),
    )
    # Layers that shared layers depend on.
    self.shared_layer_origins = {
        j for i, j in enumerate(self.kv_cache_sharing_patterns) if i != j
    }

    self.layers = compat.ModuleList()
    for i in range(config.num_layers):
      attn_type = attention_types[i]
      h_dim = config.hidden_dim
      if (
          self.kv_cache_sharing_patterns[i] != i
          and config.override_kv_shared_ffw_hidden is not None
      ):
        h_dim = config.override_kv_shared_ffw_hidden
      self.layers.append(
          DecoderLayer(
              config=config, attn_type=attn_type, hidden_dim=h_dim, rngs=rngs
          )
      )

    self.final_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def __call__(
      self,
      tokens,
      positions=None,
      cache=None,
      attention_mask=None,
      decode_only_last_token=False,
      segment_ids=None,
  ):
    # [FUNCTION DOC]
    # Forward pass. Inputs: tokens [B, T]; positions (default 0..T-1 per row);
    # cache (Cache or None); attention_mask [B, T, S]; decode_only_last_token
    # (only compute logits for the last position); segment_ids (ignored).
    # Embeds tokens and computes per-layer inputs if enabled. For each layer:
    # shared layers get no own cache and read K/V from their origin layer
    # (fresh projections recorded during this prefill, or the origin's updated
    # cache during decode); other layers use cache['layer_<i>']. Records origin
    # layers' K/V during prefill and stores non-shared layers' new caches.
    # Applies final norm, optional last-token slicing, unembedding in float32
    # and soft-capping tanh(logits/cap)*cap.
    # Output: (logits [B, T or 1, vocab] float32, new cache or None when no
    # cache was passed).
    # KNOWN ISSUE: prefill is detected as T > 1, so a single-token prompt is
    # treated as decode (shared layers then read from new_cache).
    # Accepted for RL pipeline compatibility; unused until seq packing lands.
    del segment_ids
    if positions is None:
      B, T = tokens.shape  # pylint: disable=invalid-name
      positions = jnp.tile(jnp.arange(T)[None, :], (B, 1))

    new_cache = {}
    x = self.embedder.encode(tokens)

    per_layer_inputs = None
    if self.config.per_layer_input_dim > 0:
      per_layer_inputs = self.embedder.encode_per_layer_input(x, tokens)

    # Stores the raw KV projections for the current forward pass. Used for
    # KV cache sharing during prefill.
    transient_kvs = {}
    is_prefill = tokens.shape[1] > 1

    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'

      shared_idx = self.kv_cache_sharing_patterns[i]
      is_shared = shared_idx != i
      if is_shared:
        assert shared_idx in self.shared_layer_origins
        layer_cache = None
        shared_layer_name = f'layer_{shared_idx}'
        if is_prefill:
          # During prefill, use full KV projections from the shared layer.
          shared_k, shared_v = transient_kvs[shared_layer_name]
          kv_shared_cache = {'k': shared_k, 'v': shared_v}
        else:
          # During decoding, use the shared layer's cache (which may be
          # an optimized sliding window ring cache).
          kv_shared_cache = new_cache.get(shared_layer_name)
      else:
        layer_cache = cache[layer_name] if cache else None
        kv_shared_cache = None

      layer_cache, x, layers_kvs = layer(
          x,
          positions,
          layer_cache,
          attention_mask,
          per_layer_input=per_layer_inputs[:, :, i, :]
          if per_layer_inputs is not None
          else None,
          kv_shared_cache=kv_shared_cache,
      )
      if is_prefill and i in self.shared_layer_origins:
        transient_kvs[layer_name] = layers_kvs
      if not is_shared:
        new_cache[layer_name] = layer_cache

    x = self.final_norm(x)
    if decode_only_last_token:
      # Only compute logits for the last token. This can significantly reduce
      # memory requirements during prefill (when sampling), since we only need
      # the logits for the last token to sample from.
      x = x[:, -1:, :]
    logits = self.embedder.decode(x).astype(jnp.float32)

    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap

    return logits, (None if cache is None else new_cache)

  def init_cache(self, batch_size, max_seq_len, dtype):
    # [FUNCTION DOC]
    # Returns a dict of per-layer caches for non-shared layers only
    # ('layer_<i>' -> layer.init_cache(batch_size, max_seq_len, dtype)).
    cache = {}
    for i, layer in enumerate(self.layers):
      if self.kv_cache_sharing_patterns[i] != i:
        continue  # Skip shared layers.
      cache[f'layer_{i}'] = layer.init_cache(batch_size, max_seq_len, dtype)
    return cache

  def get_model_input(self):
    # [FUNCTION DOC]
    # Returns dummy inputs: tokens and positions int32 ones [2, 2], cache None,
    # attention_mask bool ones [2, 1, 2] (broadcast over queries).
    """Returns a dummy model input for the transformer.

    This dummy input has a batch size compatible with FSDP sharding on a
    2-device axis.
    """
    dummy_batch_size = 2
    dummy_seq_len = 2
    return {
        'tokens': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'positions': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'cache': None,
        'attention_mask': jnp.ones(
            (dummy_batch_size, 1, dummy_seq_len), dtype=jnp.bool
        ),
    }

  @property
  def num_embed(self) -> int:
    # [FUNCTION DOC]
    # Property: vocabulary size from config.
    return self.config.num_embed
