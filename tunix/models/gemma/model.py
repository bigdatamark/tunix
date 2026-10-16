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
# Flax NNX implementation of the Gemma (v1, v1.1) and Gemma 2 decoder-only
# transformer, plus helpers to load linen-format checkpoints into it.
# Contents:
# - Enums AttentionType (GLOBAL / LOCAL_SLIDING) and RematConfig (NONE / BLOCK /
#   DECODER) controlling attention kind and activation rematerialization.
# - ShardingConfig (logical mesh-axis tuples per weight/activation) and
#   ModelConfig (architecture hyperparameters with factory classmethods for
#   gemma 2b/7b, 1.1 variants and gemma2 2b/9b).
# - Layers: Embedder, Einsum, Attention (MHA / MQA / GQA with RoPE, optional
#   logit soft-cap and sliding window), FeedForward (gated GELU), DecoderLayer,
#   RMSNorm.
# - Gemma: full model (embed -> N decoder layers -> final norm -> tied
#   unembedding -> optional final logit soft-cap), KV-cache creation.
# - GemmaWithScoreHead: Gemma backbone plus a scalar linear score head.
# - Linen-to-NNX conversion helpers (module_from_linen_variables and private
#   key/value mappers).
# Shape letters used throughout: B batch, T/L query length, S cache/key length,
# D embed dim, N query heads, K kv heads, G group size, H head dim, F hidden dim.
# Type aliases: LayerCache is a dict with keys 'k', 'v' (arrays shaped
# [B, cache_size, K, H]) and 'end_index' (int32 [B]); Cache maps
# 'layer_<i>' to a LayerCache.
# Side effects at import: calls env_utils.setup_sharding_environment(). Defines
# K_MASK, a very large negative float used to mask attention logits.
"""Gemma transformer."""

from collections.abc import Iterable
import dataclasses
import enum
from typing import Any, Callable, Tuple

import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping
from tunix.generate.mappings import BackendMappingMixin
from tunix.models.gemma import params as params_lib
from tunix.utils import compat
from tunix.utils import env_utils


LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


env_utils.setup_sharding_environment()


class AttentionType(enum.Enum):
  # [CLASS DOC]
  # Enum of attention kinds per layer: GLOBAL (=1) attends over the whole cache;
  # LOCAL_SLIDING (=2) additionally restricts keys to a window around the query
  # position (see _create_sliding_mask).
  GLOBAL = 1
  LOCAL_SLIDING = 2


class RematConfig(enum.Enum):
  # [CLASS DOC]
  # Enum selecting gradient rematerialization (activation checkpointing):
  # NONE stores all activations; BLOCK wraps the attention block and the
  # feed-forward block individually in nnx.remat; DECODER wraps each whole
  # decoder layer in nnx.remat.
  NONE = enum.auto()  # No remat, all activations will be stored in HBM.
  BLOCK = enum.auto()  # Remat the entire attn block.
  DECODER = enum.auto()  # Remat the entire decoder layer.


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  # [CLASS DOC]
  # Frozen, slotted dataclass holding PartitionSpec-style tuples (mesh axis
  # names such as 'fsdp', 'tp' or None per tensor dimension) for each weight and
  # activation: emb_vd (embedding [vocab, D]), q_weight_ndh, kv_weight_cndh,
  # qkv_weight_cndh, o_weight_nhd, ffw_weight_df, ffw_weight_fd,
  # rms_norm_weight, act_btd, act_btf, act_btnh, score_weight_d1.
  # Used as the `sharding` metadata on nnx.Param and for activation sharding
  # constraints via shard().
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
  score_weight_d1: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    # [FUNCTION DOC]
    # Static factory returning the default ShardingConfig.
    # Input: is_sampling (bool, default False). When True, the 'fsdp' axis is
    # replaced by None for all weights (weights are not FSDP-sharded during
    # sampling) and act_btd's last dimension is unsharded instead of 'tp'.
    # Activations always keep 'fsdp' on the batch dimension.
    # Output: a new ShardingConfig. No exceptions or side effects.
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
        score_weight_d1=(fsdp, None),
    )


@dataclasses.dataclass(slots=True)
class ModelConfig:
  # [CLASS DOC]
  # Mutable, slotted dataclass of Gemma architecture hyperparameters:
  # num_layers, num_embed (vocab size), embed_dim, hidden_dim (FFN width),
  # num_heads, head_dim, num_kv_heads, final_logit_softcap (None disables),
  # use_post_attn_norm, use_post_ffw_norm, attention_types (one AttentionType
  # per layer), attn_logits_soft_cap (optional), sliding_window_size
  # (required when any layer is LOCAL_SLIDING), shd_config (defaults to the
  # non-sampling default sharding) and remat_config (default NONE).
  # Classmethod factories return preset configurations for released checkpoints.
  """Configuration for the gemma transformer."""

  num_layers: int
  num_embed: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  final_logit_softcap: float | None
  use_post_attn_norm: bool
  use_post_ffw_norm: bool
  attention_types: Iterable[AttentionType]
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE

  @classmethod
  def gemma_2b(cls):
    # [FUNCTION DOC]
    # Classmethod returning the Gemma 2B config: 18 layers, vocab 256128,
    # embed 2048, hidden 16384, 8 heads, head_dim 256, 1 kv head (MQA), no final
    # soft-cap, all GLOBAL attention, no post-attn / post-ffw norms.
    num_layers = 18
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=2048,
        hidden_dim=16384,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        final_logit_softcap=None,
        attention_types=(AttentionType.GLOBAL,) * num_layers,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )

  @classmethod
  def gemma_2b_it(cls):
    # [FUNCTION DOC]
    # Classmethod; instruction-tuned 2B uses the same architecture, so it returns
    # the gemma_2b config.
    return cls.gemma_2b()

  @classmethod
  def gemma1p1_2b_it(cls):  # gemma1.1-2b-it
    # [FUNCTION DOC]
    # Classmethod for Gemma 1.1 2B instruction-tuned; returns the gemma_2b config.
    return cls.gemma_2b()

  @classmethod
  def gemma_7b(cls):
    # [FUNCTION DOC]
    # Classmethod returning the Gemma 7B config: 28 layers, vocab 256128,
    # embed 3072, hidden 24576, 16 heads, head_dim 256, 16 kv heads (full MHA),
    # no final soft-cap, all GLOBAL attention, no post norms.
    num_layers = 28
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=3072,
        hidden_dim=24576,
        num_heads=16,
        head_dim=256,
        num_kv_heads=16,
        final_logit_softcap=None,
        attention_types=(AttentionType.GLOBAL,) * num_layers,
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )

  @classmethod
  def gemma_7b_it(cls):
    # [FUNCTION DOC]
    # Classmethod; instruction-tuned 7B returns the gemma_7b config.
    return cls.gemma_7b()

  @classmethod
  def gemma1p1_7b_it(cls):  # gemma1.1-7b-it
    # [FUNCTION DOC]
    # Classmethod for Gemma 1.1 7B instruction-tuned; returns the gemma_7b config.
    return cls.gemma_7b()

  @classmethod
  def gemma2_2b(cls):
    # [FUNCTION DOC]
    # Classmethod returning the Gemma 2 2B config: 26 layers, vocab 256128,
    # embed 2304, hidden 9216, 8 heads, head_dim 256, 4 kv heads (GQA), final
    # logit soft-cap 30.0, attention types alternating LOCAL_SLIDING then GLOBAL
    # (pattern repeated num_layers/2 times), post-attn and post-ffw norms on,
    # attention logit soft-cap 50.0, sliding window 4096.
    num_layers = 26
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=2304,
        hidden_dim=9216,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        final_logit_softcap=30.0,
        attention_types=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        )
        * int(num_layers / 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  @classmethod
  def gemma2_2b_it(cls):
    # [FUNCTION DOC]
    # Classmethod; Gemma 2 2B instruction-tuned returns the gemma2_2b config.
    return cls.gemma2_2b()

  @classmethod
  def gemma2_9b(cls):
    # [FUNCTION DOC]
    # Classmethod returning the Gemma 2 9B config: 42 layers, vocab 256128,
    # embed 3584, hidden 28672, 16 heads, head_dim 256, 8 kv heads, final
    # soft-cap 30.0, alternating LOCAL_SLIDING/GLOBAL layers, post norms on,
    # attention soft-cap 50.0, sliding window 4096.
    num_layers = 42
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=3584,
        hidden_dim=28672,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        final_logit_softcap=30.0,
        attention_types=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        )
        * int(num_layers / 2),
        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  @classmethod
  def gemma2_9b_it(cls):
    # [FUNCTION DOC]
    # Classmethod; Gemma 2 9B instruction-tuned returns the gemma2_9b config.
    return cls.gemma2_9b()


def shard(x: jnp.ndarray, s: Tuple[str, ...]):
  # [FUNCTION DOC]
  # HELPER: applies an activation sharding constraint.
  # Inputs: x (array), s (tuple of mesh axis names / None, one per dim).
  # Behavior: reads the current physical mesh from the thread-local pxla
  # resources; if the mesh is empty or the first JAX device is a CPU, returns x
  # unchanged; otherwise returns jax.lax.with_sharding_constraint(x,
  # NamedSharding(mesh, PartitionSpec(*s))).
  # Output: the (possibly constrained) array. No side effects.
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )


class Embedder(nnx.Module):
  # [CLASS DOC]
  # Token embedding table shared between input embedding and output
  # unembedding (tied weights). Holds parameter input_embedding of shape
  # [vocab_size, embed_dim] and the ShardingConfig.
  """Embedder module."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    # [FUNCTION DOC]
    # Creates input_embedding as an nnx.Param initialized from a normal
    # distribution using rngs.params(), shaped [vocab_size, embed_dim], with
    # sharding metadata shd_config.emb_vd; stores shd_config.
    self.input_embedding = nnx.Param(
        nnx.initializers.normal()(rngs.params(), (vocab_size, embed_dim)),
        sharding=shd_config.emb_vd,
    )
    self.shd_config = shd_config

  @jax.named_scope('embedder_encode')
  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Embeds integer token ids x (shape [B, L]) by indexing the table, scales
    # the result by sqrt(embed_dim) cast to the embedding dtype, applies the
    # act_btd sharding constraint and returns [B, L, D].
    x = self.input_embedding[(x,)]
    x *= jnp.sqrt(x.shape[-1]).astype(x.dtype)
    x = shard(x, self.shd_config.act_btd)
    return x

  @jax.named_scope('embedder_decode')
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Unembeds hidden states x ([..., D]) by a dot product with the transposed
    # embedding table, returning logits [..., vocab].
    return jnp.dot(x, self.input_embedding.value.T)

  @property
  def embed_dim(self):
    # [FUNCTION DOC]
    # Property: embedding dimension (second dim of the table).
    return self.input_embedding.value.shape[1]

  @property
  def num_embed(self):
    # [FUNCTION DOC]
    # Property: vocabulary size (first dim of the table).
    return self.input_embedding.value.shape[0]


class Einsum(nnx.Module):
  # [CLASS DOC]
  # Parameterized einsum layer: stores an einsum equation string, the weight
  # shape and a normally initialized weight parameter w with given sharding.
  """Einsum is a convenience module for parameterized tensor multiplication."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
  ):
    # [FUNCTION DOC]
    # Stores einsum_str and shape, and creates w as an nnx.Param of the given
    # shape drawn from a normal initializer with rngs.params(), annotated with
    # the provided sharding tuple.
    self.einsum_str = einsum_str
    self.shape = shape
    self.w = nnx.Param(
        nnx.initializers.normal()(rngs.params(), shape), sharding=sharding
    )

  @jax.named_scope('einsum')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Returns the einsum of the input x with the weight w using einsum_str.
    return jnp.einsum(self.einsum_str, x, self.w.value)


@jax.named_scope('rope')
def apply_rope(
    inputs: jaxtyping.Array,  # [B, L]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    max_wavelength: int = 10_000,
) -> jaxtyping.Array:
  # [FUNCTION DOC]
  # Applies rotary position embeddings (RoPE, half-split / non-interleaved
  # variant) along the last axis.
  # Inputs: inputs of shape [B, L, N, H] (the inline comment "[B, L]" is
  # just the leading dims), positions [B, L], head_dim H, max_wavelength
  # (default 10000).
  # Behavior: frequency exponents are 2*i/H for i in [0, H/2); timescale is
  # max_wavelength to that power; angle = position / timescale, broadcast to
  # [B, L, 1, H/2]. Splits the input into first and second halves along the
  # last axis and rotates: first*cos - second*sin, second*cos + first*sin,
  # concatenated back.
  # Output: array with the same shape, cast back to the input dtype.
  """Applies RoPE."""
  fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
  timescale = max_wavelength**fraction

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)


K_MASK = -2.3819763e38  # Set to a large negative number.


def _create_sliding_mask(
    segment_pos: jnp.ndarray,  # [B, seq_len]
    cache_len: int,
    sliding_window_size: int,
):
  # [FUNCTION DOC]
  # HELPER: builds a boolean sliding-window mask for local attention.
  # Inputs: segment_pos [B, T], cache_len (number of key slots), and
  # sliding_window_size W.
  # Behavior: key slot index c is allowed for a query at position p when
  # p - W + 1 <= c <= p + W - 1 (computed via broadcasting [1,1,cache_len]
  # against [B,T,1]; the two comparisons are combined by multiplication).
  # Output: mask of shape [B, T, cache_len].
  # KNOWN ISSUE: the docstring describes the window as [pos - window,
  # pos + window] but the code uses exclusive bounds (pos - W + 1 .. pos + W - 1).
  # It also compares absolute positions to cache slot indices, which is only
  # correct while the cache has not wrapped around.
  """Helper function to create sliding mask for local attention.

  It generates the sliding mask based on current segment position, the window
  is [segment_pos - window, segment_pos + window]

  Args:
    segment_pos: Current segment position of shape [B, seq_len].
    cache_len: The length of the cache.
    sliding_window_size: The size of the sliding window.

  Returns:
    The sliding mask of shape [B, seq_len, cache_len].
  """
  cache_positions = jnp.arange(cache_len)

  cache_positions = cache_positions[None, None, :]  # [1, 1, cache_len]
  segment_pos = segment_pos[:, :, None]  # [B, seq_len, 1]

  # abs_pos - window <= key_abs_pos <= abs_pos + window
  sliding_mask = cache_positions >= segment_pos - sliding_window_size + 1
  sliding_mask *= cache_positions <= segment_pos + sliding_window_size - 1
  return sliding_mask  # [B, seq_len, cache_len]


class Attention(nnx.Module):
  # [CLASS DOC]
  # Multi-head self-attention layer supporting MHA (num_heads == num_kv_heads,
  # one fused qkv einsum), MQA (num_kv_heads == 1) and GQA (otherwise, separate q
  # and kv einsums with grouped computation). Applies RoPE to queries and keys,
  # scales queries by head_dim^-0.5, optional tanh soft-capping of logits,
  # optional sliding-window masking, softmax, value aggregation and output
  # projection. Supports a left-aligned KV cache and optional remat.
  """Attention module."""

  def __init__(
      self,
      num_heads: int,
      num_kv_heads: int,
      features: int,
      head_dim: int,
      attn_type: AttentionType,
      *,
      rngs: nnx.Rngs,
      sliding_window_size: int | None = None,
      attn_logits_soft_cap: float | None = None,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      remat_config: RematConfig = RematConfig.BLOCK,
  ):
    # [FUNCTION DOC]
    # Inputs: num_heads, num_kv_heads, features (D), head_dim (H), attn_type,
    # rngs, optional sliding_window_size, attn_logits_soft_cap, shd_config and
    # remat_config (default BLOCK).
    # Raises ValueError when attn_type is LOCAL_SLIDING and sliding_window_size
    # is None.
    # Creates attn_vec_einsum ('BTNH,NHD->BTD', weight [N, H, D]). If
    # num_heads == num_kv_heads creates qkv_einsum ('BTD,SNDH->SBTNH', weight
    # [3, N, D, H]); otherwise q_einsum ('BTD,NDH->BTNH', [N, D, H]) and
    # kv_einsum ('BSD,CKDH->CBSKH', [2, K, D, H]). The kv weight sharding is
    # (None, None, 'fsdp', None) when num_kv_heads == 1, else kv_weight_cndh.
    # Stores the remaining configuration on self.
    if attn_type == AttentionType.LOCAL_SLIDING and sliding_window_size is None:
      raise ValueError(
          '`sliding_window_size` must be set if `attn_type` is Local Sliding.'
      )

    self.sliding_window_size = sliding_window_size
    self.attn_type = attn_type
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.shd_config = shd_config
    self.remat_config = remat_config
    self.attn_vec_einsum = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(num_heads, head_dim, features),
        rngs=rngs,
        sharding=shd_config.o_weight_nhd,
    )
    if num_heads == num_kv_heads:
      self.qkv_einsum = Einsum(
          einsum_str='BTD,SNDH->SBTNH',
          shape=(3, num_heads, features, head_dim),
          rngs=rngs,
          sharding=shd_config.qkv_weight_cndh,
      )
    else:
      self.q_einsum = Einsum(
          einsum_str='BTD,NDH->BTNH',
          shape=(num_heads, features, head_dim),
          rngs=rngs,
          sharding=shd_config.q_weight_ndh,
      )
      self.kv_einsum = Einsum(
          einsum_str='BSD,CKDH->CBSKH',
          shape=(2, num_kv_heads, features, head_dim),
          rngs=rngs,
          sharding=(None, None, 'fsdp', None)
          if num_kv_heads == 1
          else shd_config.kv_weight_cndh,
      )

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Core attention computation (un-rematted).
    # Inputs: x [B, T, D], segment_pos [B, T], cache (LayerCache or None),
    # attn_mask boolean [B, T, S] where S is cache size (or T without cache).
    # Steps: project to q, k, v (fused or separate); apply act_btnh sharding;
    # apply RoPE to q and k; scale q by head_dim^-0.5. If a cache is given,
    # write k and v into the cache arrays at slot (end_index[0] modulo cache
    # length) along the sequence axis with dynamic_update_slice, and use the full
    # cache arrays as keys/values. Compute logits: for GQA reshape q to
    # [B, T, K, G, H], einsum with keys to [B, T, K, G, S] and reshape back to
    # [B, T, N, S]; otherwise plain 'BTNH,BSNH->BTNS'. Optionally soft-cap
    # logits as tanh(logits/cap)*cap. For LOCAL_SLIDING layers multiply the mask
    # by the sliding mask. Masked positions are set to K_MASK (mask broadcast
    # over the head axis), softmax over keys in float then cast to key dtype,
    # aggregate values (grouped for GQA), project with attn_vec_einsum and
    # apply act_btd sharding.
    # Output: tuple (new_cache, attn_output [B, T, D]); new_cache has the updated
    # 'k', 'v' and end_index + T, or is None when no cache was given.
    # KNOWN ISSUE: only end_index[0] is used as the write offset for the whole
    # batch, so all batch rows are assumed to share the same cache position.
    seq_len = x.shape[1]

    if self.use_qkv_einsum:
      query_proj, key_proj, value_proj = self.qkv_einsum(x)
    else:
      query_proj = self.q_einsum(x)
      key_proj, value_proj = self.kv_einsum(x)

    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)

    query_proj = apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
    )
    query_scaled = query_proj * self.head_dim**-0.5
    key_proj = apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
    )

    # Cache is left aligned.
    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )

    if self.use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = query_scaled.shape
      query_scaled = query_scaled.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_scaled, key_proj)
      b, t, k, g, s = logits.shape
      logits = logits.reshape((b, t, k * g, s))
    else:
      # [batch_size, seq_len, num_heads, cache_size]
      # If cache is None, then cache_size = seq_len.
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

    if self.attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / self.attn_logits_soft_cap)
      logits = logits * self.attn_logits_soft_cap

    if self.attn_type == AttentionType.LOCAL_SLIDING:
      sliding_mask = _create_sliding_mask(
          segment_pos,
          cache_len=attn_mask.shape[-1],
          sliding_window_size=self.sliding_window_size,
      )
      attn_mask = sliding_mask * attn_mask

    padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)

    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)

    if self.use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = probs.shape
      probs = probs.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs, value_proj)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
      # [batch_size, seq_len, num_heads, head_dim]
      encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)

    attn_output = self.attn_vec_einsum(encoded)
    attn_output = shard(attn_output, self.shd_config.act_btd)

    if cache is not None:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, attn_output

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Forward entry point. If remat_config is BLOCK (compared both to the enum
    # and its integer value) calls block via nnx.remat on the unbound function
    # with self as first argument; otherwise calls block directly. Same inputs
    # and outputs as block.
    if (
        self.remat_config == RematConfig.BLOCK
        or self.remat_config == RematConfig.BLOCK.value
    ):
      # nnx.remat needs to be applied to the unbound function and take self
      # as the first argument.
      return nnx.remat(self.block.__func__)(
          self, x, segment_pos, cache, attn_mask
      )
    else:
      return self.block(x, segment_pos, cache, attn_mask)

  @property
  def head_dim(self):
    # [FUNCTION DOC]
    # Property: head dimension, taken from attn_vec_einsum.shape[1].
    return self.attn_vec_einsum.shape[1]

  @property
  def num_heads(self):
    # [FUNCTION DOC]
    # Property: number of query heads (qkv_einsum.shape[1] when fused, else
    # q_einsum.shape[0]).
    return (
        self.qkv_einsum.shape[1]
        if self.use_qkv_einsum
        else self.q_einsum.shape[0]
    )

  @property
  def num_kv_heads(self):
    # [FUNCTION DOC]
    # Property: number of kv heads (qkv_einsum.shape[1] when fused, else
    # kv_einsum.shape[1]).
    return (
        self.qkv_einsum.shape[1]
        if self.use_qkv_einsum
        else self.kv_einsum.shape[1]
    )

  @property
  def use_qkv_einsum(self):
    # [FUNCTION DOC]
    # Property: True if a non-None qkv_einsum attribute exists (fused MHA path).
    return hasattr(self, 'qkv_einsum') and self.qkv_einsum is not None

  @property
  def use_gqa(self):
    # [FUNCTION DOC]
    # Property: True for grouped-query attention, i.e. num_kv_heads differs
    # from num_heads and is greater than 1 (MQA with one kv head is handled by
    # plain broadcasting-free einsum instead).
    # KNOWN ISSUE: for MQA (1 kv head, more query heads) use_gqa is False and the
    # non-grouped einsum 'BTNH,BSNH->BTNS' is used, which relies on the key head
    # axis broadcasting (size 1 vs N) inside einsum.
    return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1


class FeedForward(nnx.Module):
  # [CLASS DOC]
  # Gated feed-forward network: down_proj(gelu(gate_proj(x)) * up_proj(x)),
  # all bias-free nnx.Linear layers with zero-initialized kernels.
  """Feed forward module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Stores config and creates gate_proj and up_proj (embed_dim -> hidden_dim,
    # kernel sharding ffw_weight_df) and down_proj (hidden_dim -> embed_dim,
    # sharding ffw_weight_fd), all without bias and with zeros initializer.
    self.config = config
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, config.shd_config.ffw_weight_df
        ),
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, config.shd_config.ffw_weight_df
        ),
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, config.shd_config.ffw_weight_fd
        ),
    )

  @jax.named_scope('feed_forward')
  def block(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Computes gelu(gate_proj(x)) * up_proj(x), applies the act_btf sharding
    # constraint, then down_proj. Input [B, T, D], output [B, T, D].
    ff_gate = self.gate_proj(x)
    gate_value = nnx.gelu(ff_gate)

    ff1 = self.up_proj(x)
    activations = gate_value * ff1
    activations = shard(activations, self.config.shd_config.act_btf)

    outputs = self.down_proj(activations)
    return outputs

  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Calls block through nnx.remat when config.remat_config is BLOCK,
    # otherwise directly.
    # KNOWN ISSUE: unlike Attention.__call__, this only compares against the
    # enum member, not its integer value, so a config holding the raw value
    # would skip remat here.
    if self.config.remat_config == RematConfig.BLOCK:
      return nnx.remat(self.block.__func__)(self, x)
    else:
      return self.block(x)


class DecoderLayer(nnx.Module):
  # [CLASS DOC]
  # One transformer decoder layer: pre-attention RMSNorm, Attention, optional
  # post-attention RMSNorm, residual add, pre-FFW RMSNorm, FeedForward, optional
  # post-FFW RMSNorm, residual add.
  """Transformer block."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      attn_type: AttentionType,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Builds pre_attention_norm, attn (Attention configured from config and the
    # given attn_type, passing soft cap, sliding window, sharding and remat
    # config), optional post_attn_norm (if config.use_post_attn_norm),
    # pre_ffw_norm, mlp (FeedForward) and optional post_ffw_norm (if
    # config.use_post_ffw_norm). Stores config.
    self.config = config
    self.pre_attention_norm = RMSNorm(
        config.embed_dim, rngs=rngs, shd_config=config.shd_config
    )
    self.attn = Attention(
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        features=config.embed_dim,
        head_dim=config.head_dim,
        attn_type=attn_type,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        sliding_window_size=config.sliding_window_size,
        rngs=rngs,
        shd_config=config.shd_config,
        remat_config=config.remat_config,
    )
    if config.use_post_attn_norm:
      self.post_attn_norm = RMSNorm(
          config.embed_dim, rngs=rngs, shd_config=config.shd_config
      )

    self.pre_ffw_norm = RMSNorm(
        config.embed_dim, rngs=rngs, shd_config=config.shd_config
    )
    self.mlp = FeedForward(
        config=config,
        rngs=rngs,
    )
    if config.use_post_ffw_norm:
      self.post_ffw_norm = RMSNorm(
          config.embed_dim, rngs=rngs, shd_config=config.shd_config
      )

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Runs the layer: h = attn(pre_attention_norm(x)) (optionally
    # post-normalized) + x; out = mlp(pre_ffw_norm(h)) (optionally
    # post-normalized) + h. Inputs as in Attention.block.
    # Output: (updated layer cache or None, out [B, T, D]).
    inputs_normalized = self.pre_attention_norm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
    )

    if self.config.use_post_attn_norm:
      attn_output = self.post_attn_norm(attn_output)

    attn_output += x

    outputs = self.pre_ffw_norm(attn_output)
    outputs = self.mlp(outputs)

    if self.config.use_post_ffw_norm:
      outputs = self.post_ffw_norm(outputs)

    outputs += attn_output
    return cache, outputs

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Calls block through nnx.remat when config.remat_config is DECODER,
    # otherwise directly. Same inputs/outputs as block.
    if self.config.remat_config == RematConfig.DECODER:
      return nnx.remat(self.block.__func__)(
          self, x, segment_pos, cache, attn_mask
      )
    else:
      return self.block(x, segment_pos, cache, attn_mask)


class RMSNorm(nnx.Module):
  # [CLASS DOC]
  # Root-mean-square layer norm with a zero-initialized scale applied as
  # (1 + scale), epsilon 1e-6.
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    # [FUNCTION DOC]
    # Creates scale, a zero-initialized nnx.Param of shape [dim] with sharding
    # shd_config.rms_norm_weight.
    self.scale = nnx.Param(
        nnx.initializers.zeros_init()(rngs.params(), dim),
        sharding=shd_config.rms_norm_weight,
    )

  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Computes x * rsqrt(mean(x^2 over last axis) + 1e-6), then multiplies by
    # (1 + scale) with scale reshaped to rank of x. Returns same shape as x.
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

    # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
    # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
    # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
    scale = jnp.expand_dims(self.scale.value, axis=range(len(x.shape) - 1))
    normed_inputs = normed_inputs * (1 + scale)
    return normed_inputs


def _flatten_path(path: tuple[str | int, ...]) -> str:
  # [FUNCTION DOC]
  # HELPER: converts a key path tuple into a dotted string for error messages;
  # string items are joined with '.', integer items rendered as '[i]' (with
  # the preceding dot removed). Raises ValueError for any other item type.
  def f(item) -> str:
    # [FUNCTION DOC]
    # HELPER: formats one path item: string as-is, int as '[int]', otherwise
    # raises ValueError.
    if isinstance(item, str):
      return f'{item}'
    elif isinstance(item, int):
      return f'[{item}]'
    else:
      raise ValueError(f'Unexpected type {type(item)}')

  return '.'.join([f(item) for item in path]).replace('.[', '[')


def module_from_linen_variables(
    module_factory: Callable[[], nnx.Module],
    variables: flax.typing.VariableDict,
    map_key_fn: None | (
        Callable[[tuple[str, ...]], tuple[str | int, ...]]
    ) = None,
    assign_val_fn: None | (
        Callable[
            [
                dict[tuple[str, ...], Any],
                tuple[str | int, ...],
                flax.typing.VariableDict,
            ],
            dict[tuple[str, ...], Any],
        ]
    ) = None,
) -> nnx.Module:
  # [FUNCTION DOC]
  # Builds an NNX module and fills its state from a linen variables dict.
  # Inputs: module_factory (no-arg callable producing the module),
  # variables (nested linen variable dict), optional map_key_fn (maps each
  # flattened linen path tuple to an NNX state path; default strips the first
  # element when the top-level dict contains 'params', else identity), optional
  # assign_val_fn (assigns a value into the flat state dict; default sets
  # state[path].value = val).
  # Behavior: abstractly creates the module with nnx.eval_shape, splits into
  # graph def and state, flattens the state into a path-to-variable dict,
  # iterates over flattened variables, maps each key and assigns its value,
  # then rebuilds a State from the flat dict and merges with the graph def.
  # Output: the populated nnx.Module.
  # Raises ValueError if a mapped path does not exist in the module state.
  # Side effects: none beyond mutating the temporary state dict. Parameters
  # never assigned keep abstract ShapeDtypeStruct values.
  """Returns an `nnx.Module` initialized with the `variables` of a linen module.

  Args:
    module_factory: A no-args callable that returns an `nnx.Module`.
    variables: A dictionary of variables.
    map_key_fn: An optional function for mapping keys in the `variables`
      dictionary to keys in the `nnx.Module`'s state. If not provided it is
      assumed that after removing the collection name the keys in the
      `variables` dictionary are the same as the keys in the `nnx.Module`'s
      state.
    assign_val_fn: An optional function for assigning values in the `variables`
      dictionary to keys in the `nnx.Module`'s state. If not provided it is
      assumed that the values in the `variables` dictionary are the same as the
      values in the `nnx.Module`'s state.

  Returns:
    An `nnx.Module` initialized with the `variables` of a linen module.
  """
  if map_key_fn is None:

    def map_key_fn(path: tuple[str, ...]) -> tuple[str | int, ...]:
      # [FUNCTION DOC]
      # HELPER: default key mapper; drops the leading collection name when
      # variables has a 'params' key, otherwise returns the path unchanged.
      return path[1:] if 'params' in variables else path

  if assign_val_fn is None:

    def assign_val_fn(
        state: dict[tuple[str, ...], Any],
        mapped_path: tuple[str | int, ...],
        val: Any,
    ) -> dict[tuple[str, ...], Any]:
      # [FUNCTION DOC]
      # HELPER: default value assigner; sets state[mapped_path].value to val and
      # returns the state dict.
      state[mapped_path].value = val
      return state

  mdl: nnx.Module = nnx.eval_shape(module_factory)
  graph_def, state = nnx.split(mdl)
  state = dict(state.flat_state())
  for path, val in flax.traverse_util.flatten_dict(variables).items():
    mapped_path = map_key_fn(path)
    if mapped_path not in state:
      raise ValueError(
          f"'{mdl.__class__.__name__}.{_flatten_path(mapped_path)}' doesn't "
          f' exist (original path={path}).'
      )
    state = assign_val_fn(state, mapped_path, val)
  state = nnx.State.from_flat_path(state)

  return nnx.merge(graph_def, state)


def _map_linen_var_names(key: tuple[str, ...]) -> tuple[str | int, ...]:
  # [FUNCTION DOC]
  # HELPER: maps a flattened linen Gemma parameter path to the NNX path.
  # Each element: 'layer_<n>' becomes 'layers', n (int); 'gating_einsum'
  # becomes 'gate_proj', 'kernel'; 'linear' becomes 'down_proj', 'kernel';
  # 'post_attention_norm' becomes 'post_attn_norm'; anything else is kept.
  # Raises ValueError if an element contains 'layer_' with a non-empty prefix.
  # Output: new tuple path.
  """Maps linen variable names to nnx variable names."""
  new_key = []
  for k in key:
    if k.startswith('layer_'):
      prefix, suffix = k.split('layer_')
      if prefix:
        raise ValueError(
            'Invalid parameter key format. Expected key to start directly '
            f"with 'layer_', but found a prefix: '{prefix}' in key '{k}'."
        )
      new_key.append('layers')
      new_key.append(int(suffix))
    elif k == 'gating_einsum':
      new_key.append('gate_proj')
      new_key.append('kernel')
    elif k == 'linear':
      new_key.append('down_proj')
      new_key.append('kernel')
    elif k == 'post_attention_norm':
      new_key.append('post_attn_norm')
    else:
      new_key.append(k)

  return tuple(new_key)


def _assign_linen_params_to_nnx_state(
    state: dict[tuple[str, ...], Any],
    mapped_path: tuple[str | int, ...],
    val: Any,
) -> dict[tuple[str, ...], Any]:
  # [FUNCTION DOC]
  # HELPER: value assigner for linen Gemma params. If the path contains
  # 'gate_proj', val is the stacked gating weight [2, ...]: val[0] is written to
  # gate_proj kernel and val[1] to the sibling up_proj kernel (same prefix,
  # last two elements replaced by 'up_proj', 'kernel'). Otherwise assigns val
  # directly. Returns the mutated state dict.
  if 'gate_proj' in mapped_path:
    state[mapped_path].value = val[0]
    state[mapped_path[:-2] + ('up_proj', 'kernel')].value = val[1]
  else:
    state[mapped_path].value = val
  return state


class Gemma(BackendMappingMixin, nnx.Module):
  # [CLASS DOC]
  # Full Gemma / Gemma 2 decoder-only language model in NNX. Combines an
  # Embedder, a list of DecoderLayers (one per config attention type), a final
  # RMSNorm and tied unembedding with optional final logit soft-capping.
  # Inherits BackendMappingMixin (BACKEND_PACKAGE_PATH set to this module name)
  # for weight mapping to other inference backends.
  """Gemma transformer."""

  BACKEND_PACKAGE_PATH = __name__

  @classmethod
  def from_params(cls, params: params_lib.Params, version: str) -> 'Gemma':
    # [FUNCTION DOC]
    # Classmethod constructing a Gemma from linen params.
    # Inputs: params (dict with key 'transformer' holding linen variables),
    # version string such as '2b', '7b-it', '1.1-2b-it', '2-9b'.
    # Version to config: prefix '1.1-' becomes 'gemma1p1_', prefix '2-' becomes
    # 'gemma2_', otherwise 'gemma_' is prepended; then '.' and '-' are replaced by
    # '_' and the ModelConfig classmethod of that name is called.
    # Raises ValueError('Unsupported version') if no such classmethod exists.
    # Output: module_from_linen_variables using _map_linen_var_names and
    # _assign_linen_params_to_nnx_state, with rngs seeded 0.

    if version.startswith('1.1-'):
      config_id = version.replace('1.1-', 'gemma1p1_')
    elif version.startswith('2-'):
      config_id = version.replace('2-', 'gemma2_')
    else:
      config_id = 'gemma_' + version
    config_id = config_id.replace('.', '_').replace('-', '_')
    try:
      config = getattr(ModelConfig, config_id)()
    except AttributeError as exc:
      raise ValueError(f'Unsupported version: {version}') from exc

    return module_from_linen_variables(
        module_factory=lambda: cls(config, rngs=nnx.Rngs(params=0)),
        variables=params['transformer'],
        map_key_fn=_map_linen_var_names,
        assign_val_fn=_assign_linen_params_to_nnx_state,
    )

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Builds embedder, one DecoderLayer per (layer index, attention type) pair
    # (zip truncates to the shorter of num_layers and attention_types),
    # final_norm, and stores config and final_logits_softcap.
    self.config = config
    self.embedder = Embedder(
        vocab_size=config.num_embed,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=config.shd_config,
    )
    self.layers = compat.ModuleList([
        DecoderLayer(
            config=config,
            attn_type=attn_type,
            rngs=rngs,
        )
        for _, attn_type in zip(
            range(config.num_layers), config.attention_types
        )
    ])
    self.final_norm = RMSNorm(
        config.embed_dim, rngs=rngs, shd_config=config.shd_config
    )
    self.final_logits_softcap = config.final_logit_softcap

  def __call__(
      self,
      last_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jaxtyping.Array,  # [B, L, L']
      output_hidden_states: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    # [FUNCTION DOC]
    # Forward pass. Inputs: last_tokens int [B, L], positions [B, L], cache
    # (Cache or None), attention_mask bool [B, L, S], output_hidden_states flag.
    # Behavior: embed tokens; run each layer with its 'layer_<i>' cache entry
    # (or None), collecting updated entries into a new cache dict; apply final
    # norm; if output_hidden_states, sow the normed hidden states as
    # nnx.Intermediate 'all_hidden_states' (side effect on module state);
    # unembed to logits; if final soft-cap is set apply tanh(logits/cap)*cap.
    # Output: (logits [B, L, vocab], new cache or None).
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      last_tokens: input sequence of tokens.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.
      output_hidden_states: whether to output the hidden states.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    new_cache = None if cache is None else {}
    x = self.embedder.encode(last_tokens)
    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(
          x,
          positions,
          layer_cache,
          attention_mask,
      )
      if cache is not None:
        new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)
    logits = self.embedder.decode(x)

    if self.final_logits_softcap is not None:
      logits /= self.final_logits_softcap
      logits = jnp.tanh(logits) * self.final_logits_softcap
    return logits, new_cache  # pytype: disable=bad-return-type

  @property
  def embed_dim(self) -> int:
    # [FUNCTION DOC]
    # Property: embedder embedding dimension.
    return self.embedder.embed_dim

  @property
  def num_embed(self) -> int:
    # [FUNCTION DOC]
    # Property: embedder vocabulary size.
    return self.embedder.num_embed

  @property
  def num_layers(self) -> int:
    # [FUNCTION DOC]
    # Property: number of decoder layers.
    return len(self.layers)

  def init_cache(
      self, batch_size: int, cache_size: int, dtype: jnp.dtype
  ) -> Cache:
    # [FUNCTION DOC]
    # Creates a zeroed KV cache: for each layer i, 'layer_<i>' maps to 'k' and
    # 'v' arrays of shape [batch_size, cache_size, num_kv_heads, head_dim] with
    # the given dtype, and 'end_index' int32 zeros of shape [batch_size].
    """Initializes the cache for the model."""
    config = self.config
    shape = (batch_size, cache_size, config.num_kv_heads, config.head_dim)
    # Jax array is immutable, so updates to each layer creates new arrays.
    return {
        f'layer_{i}': {
            'k': jnp.zeros(shape, dtype=dtype),
            'v': jnp.zeros(shape, dtype=dtype),
            'end_index': jnp.zeros((batch_size,), dtype=jnp.int32)
        }
        for i in range(config.num_layers)
    }

  def get_model_input(self):
    # [FUNCTION DOC]
    # Returns a dummy keyword-input dict for tracing/sharding: last_tokens and
    # positions int32 ones [2, 1], cache None, attention_mask bool ones [2, 1, 1].
    """Returns a dummy model input for the transformer.

    This dummy input has a batch size compatible with FSDP sharding on a
    2-device axis.
    """
    dummy_batch_size = 2
    dummy_seq_len = 1
    return {
        'last_tokens': jnp.ones(
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


class GemmaWithScoreHead(nnx.Module):
  # [CLASS DOC]
  # Wraps a Gemma backbone with a linear scalar score head (e.g. for reward or
  # value models), reading the final normalized hidden states.
  """Gemma transformer with a score head."""

  def __init__(self, transformer: Gemma, rngs: nnx.Rngs):
    # [FUNCTION DOC]
    # Stores the transformer and creates score, a bias-free nnx.Linear from
    # embed_dim to 1 with normal init and sharding score_weight_d1.
    """Initializes the transformer with a score head.

    Args:
      transformer: The transformer backbone.
      rngs: The random number generator.
    """

    self.transformer = transformer
    self.score = nnx.Linear(
        in_features=transformer.embed_dim,
        out_features=1,
        use_bias=False,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(),
            transformer.config.shd_config.score_weight_d1,
        ),
        rngs=rngs,
    )

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Runs the transformer with output_hidden_states=True (logits discarded),
    # pops the sown nnx.Intermediate state from the transformer (side effect:
    # removes it), takes the last sown 'all_hidden_states' value [B, L, D] and
    # returns score(hidden) of shape [B, L, 1].
    self.transformer(*args, **kwargs, output_hidden_states=True)
    hidden_states = nnx.pop(self.transformer, nnx.Intermediate)[
        'all_hidden_states'
    ].value[-1]
    score = self.score(hidden_states)
    return score
