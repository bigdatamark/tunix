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
# Flax NNX implementation of the Gemma 3 decoder-only transformer (text and
# optional SigLIP vision input). Structure mirrors the Gemma/Gemma 2 module
# with these differences: per-layer attention pattern of 5 LOCAL_SLIDING
# layers followed by 1 GLOBAL layer (cycled); RMSNorm on queries and keys
# (QK-norm) before RoPE; separate RoPE base frequency and scale factor for
# local vs global layers; configurable query pre-attention scaling; always-on
# post-attention and post-FFW norms; no logit soft-capping; parameters created
# in a configurable param_dtype (default bfloat16); optional vision encoder
# whose image embeddings are projected into the text embedding space and
# scattered into the token sequence at placeholder positions.
# Contents: RematConfig, ShardingConfig, QueryPreAttentionNormalisation,
# ModelConfig (with preset factories for 270m/1b/4b/12b/27b), Embedder, Einsum,
# apply_rope, AttentionType, GEMMA3_ATTENTION_PATTERN, find_last_one_index,
# create_sliding_window_mask, Attention, FeedForward, DecoderLayer, RMSNorm,
# Gemma3.
# Types: LayerCache is a dict with 'k', 'v' ([B, cache_size, K, H]) and
# 'end_index' (int32 [B]); Cache maps 'layer_<i>' to LayerCache.
# K_MASK is a very large negative logit used for masking.
# Side effect at import: env_utils.setup_sharding_environment().
"""Gemma3 model."""

import dataclasses
import enum
import itertools
from typing import Tuple

import einops
import flax
from flax import nnx
import jax
from jax import numpy as jnp
import jaxtyping
from tunix.generate.mappings import BackendMappingMixin
from tunix.models.gemma3 import merge_embeddings as merge_embeddings_lib
from tunix.models.gemma3 import utils
from tunix.models.gemma3 import vision
from tunix.utils import compat
from tunix.utils import env_utils
from tunix.utils import sharding_utils


env_utils.setup_sharding_environment()


LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class RematConfig(enum.Enum):
  # [CLASS DOC]
  # Enum selecting rematerialization: NONE (store activations), BLOCK (remat
  # attention and FFN blocks separately), DECODER (remat whole decoder layers).
  NONE = enum.auto()  # No remat, all activations will be stored in HBM.
  BLOCK = enum.auto()  # Remat the entire attn block.
  DECODER = enum.auto()


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  # [CLASS DOC]
  # Frozen dataclass of partition-spec tuples for each weight/activation (same
  # fields as the Gemma ShardingConfig minus score head) plus vision_proj
  # (vision-to-text projection weight), vision_soft_emb_norm_weight and siglip
  # (a vision.SigLIPShardingConfig or None) for the vision tower.
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
  siglip: vision.SigLIPShardingConfig | None

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    # [FUNCTION DOC]
    # Static factory: default sharding. When is_sampling is True, 'fsdp' is
    # dropped from all weights and act_btd's last dim is not 'tp'-sharded.
    # Vision sharding comes from vision.SigLIPShardingConfig.get_default_sharding
    # with the same flag. Returns a new ShardingConfig.
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
        siglip=vision.SigLIPShardingConfig.get_default_sharding(is_sampling),
    )


class QueryPreAttentionNormalisation(enum.Enum):
  # [CLASS DOC]
  # Enum choosing the query scaling factor applied before attention:
  # BY_ONE_OVER_SQRT_HEAD_DIM (head_dim^-0.5) or
  # BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS ((embed_dim // num_heads)^-0.5).
  # KNOWN ISSUE: the class docstring says "Initialization strategy", which does
  # not describe what the enum controls.
  """Initialization strategy."""

  # Whether to scale the query by 1/sqrt(head_dim)
  BY_ONE_OVER_SQRT_HEAD_DIM = enum.auto()

  # Whether to scale the query by `1/sqrt(embed_dim // num_heads)`
  BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS = enum.auto()


@dataclasses.dataclass(slots=True, kw_only=True)
class ModelConfig:
  # [CLASS DOC]
  # Keyword-only, slotted dataclass of Gemma 3 hyperparameters: num_layers,
  # num_embed, embed_dim, hidden_dim, num_heads, head_dim, num_kv_heads,
  # sliding_window_size, local/global RoPE base frequencies (default 10000),
  # local/global RoPE scale factors (default 1.0), query_pre_attn_norm,
  # vision_config (SigLIPConfig or None for text-only), shd_config,
  # remat_config (default NONE), param_dtype (default bfloat16).
  # Classmethod presets accept a sharding_config and (for 4b and larger) a
  # text_only flag that leaves vision_config None when True (default) or sets
  # a default SigLIPConfig otherwise.
  """Transformer config."""

  num_layers: int
  num_embed: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  sliding_window_size: int | None = None
  local_base_frequency: int = 10_000
  global_base_frequency: int = 10_000
  local_scale_factor: float = 1.0
  global_scale_factor: float = 1.0
  query_pre_attn_norm: QueryPreAttentionNormalisation = (
      QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
  )

  vision_config: vision.SigLIPConfig | None = None

  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE
  param_dtype: jnp.dtype = jnp.bfloat16

  @classmethod
  def gemma3_270m(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: Gemma3 270M text config: 18 layers, vocab 262144, embed 640,
    # hidden 2048, 4 heads, head_dim 256, 1 kv head, sliding window 512, local
    # base freq 10k, global base freq 1M, given sharding.
    """Gemma3-270M text-only config."""
    return cls(
        num_layers=18,
        num_embed=262144,
        embed_dim=640,
        hidden_dim=2048,
        num_heads=4,
        head_dim=256,
        num_kv_heads=1,
        sliding_window_size=512,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        shd_config=sharding_config,
    )

  @classmethod
  def gemma3_270m_it(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: instruction-tuned 270M; same as gemma3_270m.
    """Gemma3-270M instruction-tuned text-only config."""
    return cls.gemma3_270m(sharding_config=sharding_config)

  @classmethod
  def _gemma3_1b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # HELPER: classmethod building the Gemma3 1B text config: 26 layers, vocab
    # 262144, embed 1152, hidden 6912, 4 heads, head_dim 256, 1 kv head, window
    # 512, base freqs 10k / 1M.
    return cls(
        num_layers=26,
        num_embed=262144,
        embed_dim=1152,
        hidden_dim=6 * 1152,
        num_heads=4,
        head_dim=256,
        num_kv_heads=1,
        sliding_window_size=512,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        shd_config=sharding_config,
    )

  @classmethod
  def gemma3_1b_it(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: 1B instruction-tuned; returns _gemma3_1b.
    """Gemma3-1B instruction-tuned text-only config."""
    return cls._gemma3_1b(sharding_config=sharding_config)

  @classmethod
  def gemma3_1b_pt(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: 1B pretrained; returns _gemma3_1b.
    """Gemma3-1B text-only config."""
    return cls._gemma3_1b(sharding_config=sharding_config)

  @classmethod
  def _gemma3_4b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      text_only: bool = True,
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # HELPER: classmethod building Gemma3 4B: 34 layers, vocab 262144, embed
    # 2560, hidden 10240, 8 heads, head_dim 256, 4 kv heads, window 1024, base
    # freqs 10k / 1M, global scale factor 8.0, vision config unless text_only.
    # KNOWN ISSUE: docstring says "text-only" but the config is multimodal when
    # text_only is False (same for the 4b/12b/27b public variants).
    """Gemma3-4B text-only config."""
    return cls(
        num_layers=34,
        num_embed=262144,
        embed_dim=2560,
        hidden_dim=2560 * 8 // 2,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        sliding_window_size=1024,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        vision_config=None if text_only else vision.SigLIPConfig(),
        shd_config=sharding_config,
    )

  @classmethod
  def gemma3_4b_it(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      text_only: bool = True,
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: 4B instruction-tuned; returns _gemma3_4b(sharding, text_only).
    """Gemma3-4B instruction-tuned text-only config."""
    return cls._gemma3_4b(sharding_config=sharding_config, text_only=text_only)

  @classmethod
  def gemma3_4b_pt(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      text_only: bool = True,
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: 4B pretrained; returns _gemma3_4b(sharding, text_only).
    """Gemma3-4B text-only config."""
    return cls._gemma3_4b(sharding_config=sharding_config, text_only=text_only)

  @classmethod
  def _gemma3_12b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      text_only: bool = True,
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # HELPER: classmethod building Gemma3 12B: 48 layers, vocab 262144, embed
    # 3840, hidden 15360, 16 heads, head_dim 256, 8 kv heads, query scaling by
    # 1/sqrt(head_dim), window 1024, base freqs 10k / 1M, global scale 8.0,
    # vision config unless text_only.
    """Gemma3-12B text-only config."""
    return cls(
        num_layers=48,
        num_embed=262144,
        embed_dim=30 * 128,
        hidden_dim=8 * 30 * 128 // 2,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        sliding_window_size=1024,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        vision_config=None if text_only else vision.SigLIPConfig(),
        shd_config=sharding_config,
    )

  @classmethod
  def gemma3_12b_it(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      text_only: bool = True,
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: 12B instruction-tuned; returns _gemma3_12b.
    """Gemma3-12B instruction-tuned text-only config."""
    return cls._gemma3_12b(sharding_config=sharding_config, text_only=text_only)

  @classmethod
  def gemma3_12b_pt(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      text_only: bool = True,
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: 12B pretrained; returns _gemma3_12b.
    """Gemma3-12B text-only config."""
    return cls._gemma3_12b(sharding_config=sharding_config, text_only=text_only)

  @classmethod
  def _gemma3_27b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      text_only: bool = True,
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # HELPER: classmethod building Gemma3 27B: 62 layers, vocab 262144, embed
    # 5376, hidden 21504, 32 heads, head_dim 128, 16 kv heads, query scaling by
    # 1/sqrt(embed_dim // num_heads) (=1/sqrt(168)), window 1024, base freqs
    # 10k / 1M, global scale 8.0, vision config unless text_only.
    """Gemma3-27B text-only config."""
    return cls(
        num_layers=62,
        num_embed=262144,
        embed_dim=5376,
        hidden_dim=5376 * 8 // 2,
        num_heads=32,
        head_dim=128,
        num_kv_heads=16,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS,
        sliding_window_size=1024,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        vision_config=None if text_only else vision.SigLIPConfig(),
        shd_config=sharding_config,
    )

  @classmethod
  def gemma3_27b_it(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      text_only: bool = True,
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: 27B instruction-tuned; returns _gemma3_27b.
    """Gemma3-27B instruction-tuned text-only config."""
    return cls._gemma3_27b(sharding_config=sharding_config, text_only=text_only)

  @classmethod
  def gemma3_27b_pt(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      text_only: bool = True,
  ) -> 'ModelConfig':
    # [FUNCTION DOC]
    # Classmethod: 27B pretrained; returns _gemma3_27b.
    """Gemma3-27B text-only config."""
    return cls._gemma3_27b(sharding_config=sharding_config, text_only=text_only)


class Embedder(nnx.Module):
  # [CLASS DOC]
  # Token embedder with tied unembedding, plus optional multimodal projection:
  # when vision_proj_dim is truthy it owns mm_soft_embedding_norm (RMSNorm over
  # vision width) and mm_input_projection (Einsum '...TM,MD->...TD', weight
  # [vision_proj_dim, embed_dim]).
  """Embedder module."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      vision_proj_dim: int | None = None,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    # [FUNCTION DOC]
    # Creates input_embedding [vocab_size, embed_dim] (normal init, param_dtype,
    # sharding emb_vd); if vision_proj_dim is given, creates the vision norm
    # (sharding vision_soft_emb_norm_weight) and projection (sharding
    # vision_proj). Stores shd_config.
    self.input_embedding = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(
            rngs.params(), (vocab_size, embed_dim)
        ),
        sharding=shd_config.emb_vd,
    )
    if vision_proj_dim:
      self.mm_soft_embedding_norm = RMSNorm(
          vision_proj_dim,
          rngs=rngs,
          param_dtype=param_dtype,
          sharding=shd_config.vision_soft_emb_norm_weight,
      )
      self.mm_input_projection = Einsum(
          einsum_str='...TM,MD->...TD',
          shape=(vision_proj_dim, self.embed_dim),
          rngs=rngs,
          sharding=shd_config.vision_proj,
          param_dtype=param_dtype,
      )
    self.shd_config = shd_config

  @jax.named_scope('embedder_encode')
  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Looks up token ids x [B, L], multiplies by sqrt(embed_dim) in the
    # embedding dtype, applies act_btd sharding constraint. Returns [B, L, D].
    x = self.input_embedding[(x,)]
    x *= jnp.sqrt(x.shape[-1]).astype(x.dtype)
    x = sharding_utils.shard(x, self.shd_config.act_btd)
    return x

  @jax.named_scope('embedder_decode')
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Returns logits x @ input_embedding^T ([..., vocab]).
    return jnp.dot(x, self.input_embedding.value.T)

  @jax.named_scope('embedder_encode_vision')
  def encode_vision(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Maps vision encoder outputs [..., T, vision_width] into text embedding
    # space: RMSNorm then projection to [..., T, D]. Raises AttributeError if
    # the embedder was built without vision_proj_dim.
    x = self.mm_soft_embedding_norm(x)
    x = self.mm_input_projection(x)
    return x

  @property
  def embed_dim(self):
    # [FUNCTION DOC]
    # Property: embedding dim (table dim 1).
    return self.input_embedding.value.shape[1]

  @property
  def num_embed(self):
    # [FUNCTION DOC]
    # Property: vocab size (table dim 0).
    return self.input_embedding.value.shape[0]


class Einsum(nnx.Module):
  # [CLASS DOC]
  # Parameterized einsum: weight w of the given shape (normal init in
  # param_dtype, given sharding) contracted with the input by einsum_str.
  """Einsum is a convenience module for parameterized tensor multiplication."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    # [FUNCTION DOC]
    # Stores einsum_str and shape and creates the weight parameter w.
    self.einsum_str = einsum_str
    self.shape = shape
    self.w = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape),
        sharding=sharding,
    )

  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Returns einsum(einsum_str, x, w).
    return jnp.einsum(self.einsum_str, x, self.w.value)


@jax.named_scope('rope')
def apply_rope(
    inputs: jaxtyping.Array,  # [B, L]
    positions: jaxtyping.Array,  # [B, L]
    *,
    head_dim: int,
    base_frequency: int,
    scale_factor: float = 1.0,
) -> jaxtyping.Array:
  # [FUNCTION DOC]
  # Applies half-split rotary embeddings to inputs [B, L, N, H] at positions
  # [B, L]. Angle = position / base_frequency^(2i/H) for i < H/2, then divided
  # by scale_factor (linear position interpolation). Rotation:
  # first*cos - second*sin, second*cos + first*sin; output cast to input dtype.
  # Raises ValueError if scale_factor < 1.0.
  """Applies RoPE."""
  fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
  timescale = base_frequency**fraction

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


K_MASK = -2.3819763e38  # Set to a large negative number.


class AttentionType(enum.Enum):
  # [CLASS DOC]
  # Enum of attention kinds: GLOBAL (=1) and LOCAL_SLIDING (=2). The module
  # constant GEMMA3_ATTENTION_PATTERN defines five LOCAL_SLIDING followed by one
  # GLOBAL; layers cycle through it.
  GLOBAL = 1
  LOCAL_SLIDING = 2


GEMMA3_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def find_last_one_index(attn_mask: jnp.ndarray) -> jnp.ndarray:
  # [FUNCTION DOC]
  # For attn_mask [B, T, S], returns the index of the rightmost true/1 entry
  # along the last axis (0 for all-zero rows), computed via argmax on the
  # reversed rows; the trailing T axis is squeezed, so the result is [B].
  # Raises an error if T != 1 (squeeze of a non-unit axis); intended only for
  # single-token decoding.
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
  # HELPER: decode-time sliding window mask. upper = last allowed key index per
  # row (find_last_one_index); returns a [B, 1, S] mask true for key indices in
  # [upper - sliding_window_size + 1, upper].
  # KNOWN ISSUE: works on cache slot indices, so it assumes the cache has not
  # wrapped around; requires seq_len == 1.
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


class Attention(nnx.Module):
  # [CLASS DOC]
  # Gemma 3 attention: like Gemma 2 attention (MHA fused qkv / GQA / MQA
  # einsums, left-aligned KV cache, remat option) but with QK RMSNorm
  # (_query_norm, _key_norm over head_dim) before RoPE, per-layer RoPE base
  # frequency and scale factor, configurable query scaling, no logit
  # soft-capping, and a separate sliding mask for prefill vs decode.
  """Attention module."""

  def __init__(
      self,
      *,
      num_heads: int,
      num_kv_heads: int,
      features: int,
      head_dim: int,
      attn_type: AttentionType,
      rngs: nnx.Rngs,
      sliding_window_size: int | None,
      rope_base_frequency: int,
      rope_scale_factor: float,
      query_pre_attn_norm: QueryPreAttentionNormalisation,
      shd_config: ShardingConfig,
      remat_config: RematConfig,
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    # [FUNCTION DOC]
    # Keyword-only inputs: num_heads, num_kv_heads, features (D), head_dim,
    # attn_type, rngs, sliding_window_size, rope_base_frequency,
    # rope_scale_factor, query_pre_attn_norm, shd_config, remat_config,
    # param_dtype. Raises ValueError for LOCAL_SLIDING without a window size.
    # Creates attn_vec_einsum ([N, H, D]); fused qkv_einsum ([3, N, D, H]) when
    # num_heads == num_kv_heads, else q_einsum ([N, D, H]) and kv_einsum
    # ([2, K, D, H], sharding (None, None, 'fsdp', None) for one kv head); and
    # unsharded RMSNorms _query_norm and _key_norm of size head_dim.
    if attn_type == AttentionType.LOCAL_SLIDING and sliding_window_size is None:
      raise ValueError(
          '`sliding_window_size` must be set if `attn_type` is Local Sliding.'
      )

    self.sliding_window_size = sliding_window_size
    self.attn_type = attn_type
    self.rope_base_frequency = rope_base_frequency
    self.rope_scale_factor = rope_scale_factor
    self.query_pre_attn_norm = query_pre_attn_norm
    self.shd_config = shd_config
    self.remat_config = remat_config
    self.attn_vec_einsum = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(num_heads, head_dim, features),
        rngs=rngs,
        sharding=shd_config.o_weight_nhd,
        param_dtype=param_dtype,
    )
    if num_heads == num_kv_heads:
      self.qkv_einsum = Einsum(
          einsum_str='BTD,SNDH->SBTNH',
          shape=(3, num_heads, features, head_dim),
          rngs=rngs,
          sharding=shd_config.qkv_weight_cndh,
          param_dtype=param_dtype,
      )
    else:
      self.q_einsum = Einsum(
          einsum_str='BTD,NDH->BTNH',
          shape=(num_heads, features, head_dim),
          rngs=rngs,
          sharding=shd_config.q_weight_ndh,
          param_dtype=param_dtype,
      )
      self.kv_einsum = Einsum(
          einsum_str='BSD,CKDH->CBSKH',
          shape=(2, num_kv_heads, features, head_dim),
          rngs=rngs,
          sharding=(None, None, 'fsdp', None)
          if num_kv_heads == 1
          else shd_config.kv_weight_cndh,
          param_dtype=param_dtype,
      )
    # No sharding on head_dim.
    self._query_norm = RMSNorm(head_dim, rngs=rngs, param_dtype=param_dtype)
    self._key_norm = RMSNorm(head_dim, rngs=rngs, param_dtype=param_dtype)

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Attention computation. Inputs: x [B, T, D], segment_pos [B, T], cache or
    # None, attn_mask bool [B, T, S].
    # Steps: project q/k/v; shard; RMSNorm q and k; RoPE on q and k with this
    # layer's base frequency / scale factor; scale q by query_pre_attn_scalar;
    # if cached, write k/v at slot end_index[0] modulo cache length and use the
    # whole cache; compute logits (grouped for GQA). For LOCAL_SLIDING: when T is
    # 1 multiply mask by create_sliding_window_mask; otherwise (prefill) by a band
    # mask keeping |query_index - key_index| < window (assumes queries start at
    # cache slot 0). Mask with K_MASK, softmax, weighted sum of values, output
    # projection and act_btd sharding.
    # Output: (new cache with end_index + T, or None; output [B, T, D]).
    # KNOWN ISSUE: a single end_index (row 0) is used for the whole batch.
    seq_len = x.shape[1]

    if self.use_qkv_einsum:
      query_proj, key_proj, value_proj = self.qkv_einsum(x)
    else:
      query_proj = self.q_einsum(x)
      key_proj, value_proj = self.kv_einsum(x)

    query_proj = sharding_utils.shard(query_proj, self.shd_config.act_btnh)
    key_proj = sharding_utils.shard(key_proj, self.shd_config.act_btnh)
    value_proj = sharding_utils.shard(value_proj, self.shd_config.act_btnh)

    query_proj = self._query_norm(query_proj)
    key_proj = self._key_norm(key_proj)

    query_proj = apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )
    query_scaled = query_proj * self.query_pre_attn_scalar
    key_proj = apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
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

    if self.attn_type == AttentionType.LOCAL_SLIDING:
      if segment_pos.shape[1] == 1:  # for decoding
        sliding_mask = create_sliding_window_mask(
            attn_mask,
            sliding_window_size=self.sliding_window_size,
        )
      else:  # for prefill
        all_ones = jnp.ones_like(attn_mask)
        sliding_mask = jnp.triu(
            all_ones, -1 * self.sliding_window_size + 1
        ) * jnp.tril(all_ones, self.sliding_window_size - 1)
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
    attn_output = sharding_utils.shard(attn_output, self.shd_config.act_btd)

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
    # Calls block via nnx.remat when remat_config is BLOCK (enum or its value),
    # otherwise directly.
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
    # Property: head_dim from attn_vec_einsum.shape[1].
    return self.attn_vec_einsum.shape[1]

  @property
  def features(self):
    # [FUNCTION DOC]
    # Property: model width D from attn_vec_einsum.shape[2].
    return self.attn_vec_einsum.shape[2]

  @property
  def query_pre_attn_scalar(self):
    # [FUNCTION DOC]
    # Property: query scaling factor per query_pre_attn_norm: head_dim^-0.5 or
    # (features // num_heads)^-0.5.
    # KNOWN ISSUE: returns None implicitly for any other enum value.
    match self.query_pre_attn_norm:
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM:
        return self.head_dim**-0.5
      case (
          QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS
      ):
        return (self.features // self.num_heads) ** -0.5

  @property
  def num_heads(self):
    # [FUNCTION DOC]
    # Property: number of query heads (from qkv_einsum or q_einsum shape).
    return (
        self.qkv_einsum.shape[1]
        if self.use_qkv_einsum
        else self.q_einsum.shape[0]
    )

  @property
  def num_kv_heads(self):
    # [FUNCTION DOC]
    # Property: number of kv heads (from qkv_einsum or kv_einsum shape).
    return (
        self.qkv_einsum.shape[1]
        if self.use_qkv_einsum
        else self.kv_einsum.shape[1]
    )

  @property
  def use_qkv_einsum(self):
    # [FUNCTION DOC]
    # Property: True when a fused qkv_einsum exists.
    return hasattr(self, 'qkv_einsum') and self.qkv_einsum is not None

  @property
  def use_gqa(self):
    # [FUNCTION DOC]
    # Property: True for GQA (kv heads != query heads and kv heads > 1).
    return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1


class FeedForward(nnx.Module):
  # [CLASS DOC]
  # Gated GELU MLP: down_proj(gelu(gate_proj(x)) * up_proj(x)) with bias-free
  # zero-initialized nnx.Linear layers in config.param_dtype.
  """Feed forward module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Creates gate_proj and up_proj (embed_dim -> hidden_dim, sharding
    # ffw_weight_df) and down_proj (hidden_dim -> embed_dim, ffw_weight_fd).
    self.config = config
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, config.shd_config.ffw_weight_df
        ),
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, config.shd_config.ffw_weight_df
        ),
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, config.shd_config.ffw_weight_fd
        ),
    )

  def block(
      self,
      x: jaxtyping.Array,
  ) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Computes the gated activation, applies act_btf sharding and projects down.
    # [B, T, D] -> [B, T, D].
    ff_gate = self.gate_proj(x)
    gate_value = nnx.gelu(ff_gate)
    ff1 = self.up_proj(x)
    activations = gate_value * ff1
    activations = sharding_utils.shard(
        activations, self.config.shd_config.act_btf
    )
    outputs = self.down_proj(activations)
    return outputs

  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Uses nnx.remat around block when remat_config is BLOCK, else calls block.
    if self.config.remat_config == RematConfig.BLOCK:
      return nnx.remat(self.block.__func__)(self, x)
    else:
      return self.block(x)


class DecoderLayer(nnx.Module):
  # [CLASS DOC]
  # Gemma 3 decoder layer: pre-attn norm, attention, post-attn norm, residual,
  # pre-FFW norm, MLP, post-FFW norm, residual. All four norms always present.
  """Transformer block."""

  def __init__(
      self,
      config: ModelConfig,
      attn_type: AttentionType,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Builds the four RMSNorms (sharding rms_norm_weight, param_dtype), the
    # Attention (local base frequency and scale for LOCAL_SLIDING layers, global
    # ones otherwise) and the FeedForward from config.
    self.config = config
    self.pre_attention_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config.rms_norm_weight,
        param_dtype=config.param_dtype,
    )
    self.attn = Attention(
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        features=config.embed_dim,
        head_dim=config.head_dim,
        attn_type=attn_type,
        sliding_window_size=config.sliding_window_size,
        rope_base_frequency=config.local_base_frequency
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_base_frequency,
        rope_scale_factor=config.local_scale_factor
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_scale_factor,
        query_pre_attn_norm=config.query_pre_attn_norm,
        rngs=rngs,
        shd_config=config.shd_config,
        remat_config=config.remat_config,
        param_dtype=config.param_dtype,
    )
    self.post_attention_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config.rms_norm_weight,
        param_dtype=config.param_dtype,
    )
    self.pre_ffw_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config.rms_norm_weight,
        param_dtype=config.param_dtype,
    )
    self.mlp = FeedForward(
        config=config,
        rngs=rngs,
    )
    self.post_ffw_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config.rms_norm_weight,
        param_dtype=config.param_dtype,
    )

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # h = post_attention_norm(attn(pre_attention_norm(x))) + x;
    # out = post_ffw_norm(mlp(pre_ffw_norm(h))) + h. Returns (cache, out).
    inputs_normalized = self.pre_attention_norm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
    )
    attn_output = self.post_attention_norm(attn_output)

    attn_output += x

    outputs = self.pre_ffw_norm(attn_output)
    outputs = self.mlp(outputs)
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
    # Uses nnx.remat around block when remat_config is DECODER, else calls block.
    if self.config.remat_config == RematConfig.DECODER:
      return nnx.remat(self.block.__func__)(
          self, x, segment_pos, cache, attn_mask
      )
    else:
      return self.block(x, segment_pos, cache, attn_mask)


class RMSNorm(nnx.Module):
  # [CLASS DOC]
  # RMSNorm with zero-initialized scale applied as (1 + scale), epsilon 1e-6.
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      rngs: nnx.Rngs,
      sharding: tuple[str, ...] = (),
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    # [FUNCTION DOC]
    # Creates scale [dim] of zeros in param_dtype with the given sharding
    # (default unsharded).
    self.scale = nnx.Param(
        nnx.initializers.zeros_init()(rngs.params(), dim).astype(param_dtype),
        sharding=sharding,
    )

  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Returns x * rsqrt(mean(x^2, last axis) + 1e-6) * (1 + scale).
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

    # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
    # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
    # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
    scale = jnp.expand_dims(self.scale.value, axis=range(len(x.shape) - 1))
    normed_inputs = normed_inputs * (1 + scale)
    return normed_inputs


class Gemma3(BackendMappingMixin, nnx.Module):
  # [CLASS DOC]
  # Gemma 3 model: optional SigLIP vision encoder (when config.vision_config
  # is set), Embedder (with vision projection sized by the vision width),
  # num_layers DecoderLayers following GEMMA3_ATTENTION_PATTERN cyclically, and
  # a final RMSNorm with tied unembedding. Mixes in BackendMappingMixin.
  """Gemma3 transformer."""

  BACKEND_PACKAGE_PATH = __name__

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    # [FUNCTION DOC]
    # Builds the submodules described in the class doc from config; sets
    # vision_encoder to None for text-only configs.
    self.config = config

    if config.vision_config is not None:
      self.vision_encoder = vision.SigLiP(
          config=config.vision_config,
          shd_config=config.shd_config.siglip,
          rngs=rngs,
      )
    else:
      self.vision_encoder = None

    self.embedder = Embedder(
        vocab_size=config.num_embed,
        embed_dim=config.embed_dim,
        vision_proj_dim=self.vision_encoder.siglip_encoder.width
        if self.vision_encoder
        else None,
        rngs=rngs,
        shd_config=config.shd_config,
        param_dtype=config.param_dtype,
    )
    self.layers = compat.ModuleList([
        DecoderLayer(
            config=config,
            attn_type=attn_type,
            rngs=rngs,
        )
        for _, attn_type in zip(
            range(config.num_layers), itertools.cycle(GEMMA3_ATTENTION_PATTERN)
        )
    ])
    self.final_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config.rms_norm_weight,
        param_dtype=config.param_dtype,
    )

  def __call__(
      self,
      last_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array | None = None,  # [B, L]
      cache: Cache | None = None,  # (sequence length L')
      attention_mask: jaxtyping.Array | None = None,  # [B, L, L']
      output_hidden_states: bool = False,
      *,
      images: jaxtyping.Array | None = None,  # [B, H, W, C] or [B, N, H, W, C]
  ) -> tuple[jaxtyping.Array, Cache | None]:
    # [FUNCTION DOC]
    # Forward pass. Inputs: last_tokens [B, L]; positions [B, L]; cache or None;
    # attention_mask [B, L, S]; output_hidden_states; keyword images
    # ([B, H, W, C] or [B, N, H, W, C], optional).
    # Embeds tokens (merging image embeddings when images are given), runs every
    # layer inside a named scope 'layer_<i>' collecting per-layer caches, applies
    # final norm, optionally sows the hidden states as nnx.Intermediate
    # 'all_hidden_states' (side effect), unembeds.
    # Output: (logits [B, L, vocab], new cache or None).
    # KNOWN ISSUE: positions and attention_mask default to None but are passed
    # straight to the layers, so calling without them fails.
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      last_tokens: input sequence of tokens.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.
      output_hidden_states: whether to output the hidden states.
      images: Input images. If None, the model will not process images.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    new_cache = None if cache is None else {}
    x = self._encode_and_get_inputs(
        tokens=last_tokens,
        images=images,
    )

    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      with jax.named_scope(layer_name):
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

    return logits, new_cache  # pytype: disable=bad-return-type

  def _encode_and_get_inputs(
      self,
      *,
      tokens: jaxtyping.Array,  # (B, L)
      images: jaxtyping.Array | None = None,  # (B, H, W, C) or (B, N, H, W, C)
  ) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # HELPER: embeds text tokens; if the model has a vision config and images are
    # given, checks multimodal support and adds a num_images axis to 4D images;
    # when images are given, merges vision embeddings via _merge_mm_embeddings.
    # Returns [B, L, D].
    """Encode the text tokens, eventually including the vision embeddings."""
    if self.config.vision_config is not None and images is not None:
      self._assert_support_mm()
      if len(images.shape) == 4:  # If num_images is 1, add an axis.
        images = einops.rearrange(images, 'b h w c -> b 1 h w c')

    # Encode the text tokens
    x = self.embedder.encode(tokens)

    # Encode the vision tokens and merge them with the text embeddings.
    if images is not None:
      x = self._merge_mm_embeddings(tokens=tokens, embeddings=x, images=images)
    return x  # pytype: disable=bad-return-type  # jax-arraylike

  def _assert_support_mm(self) -> None:
    # [FUNCTION DOC]
    # HELPER: raises ValueError if the model has no vision encoder (mentioning
    # text_only when such an attribute is truthy).
    if self.vision_encoder is None:
      msg = ''
      if getattr(self, 'text_only', False):
        msg = ' The model was created with `text_only=True`.'
      raise ValueError(
          f'The model {type(self).__name__!r} does not have vision encoder,'
          ' yet images are provided.'
          + msg
      )

  def _merge_mm_embeddings(
      self,
      *,
      tokens: jaxtyping.ArrayLike,  # B L
      embeddings: jaxtyping.ArrayLike,  # B L D
      images: jaxtyping.ArrayLike,  # B N H W C
  ) -> jaxtyping.ArrayLike:  # B L D
    # [FUNCTION DOC]
    # HELPER: encodes images to soft embeddings [B, N, P, D] and scatters them
    # into embeddings at positions where tokens equal
    # vision_config.soft_token_placeholder (merge_embeddings). Raises ValueError
    # if vision_config is None (only after encoding was attempted).
    """Update the embeddings to include the vision embeddings."""
    # Encode the images
    soft_embeddings = self._encode_vision(images)

    # Merge the soft tokens back with the text embeddings.
    if self.config.vision_config is None:
      raise ValueError(
          '`vision_config` is required for `_merge_mm_embeddings`. Received: '
          f'{self.config.vision_config=}'
      )
    merged_embeddings = merge_embeddings_lib.merge_embeddings(
        text_embeddings=embeddings,
        vision_embeddings=soft_embeddings,
        mask=tokens == self.config.vision_config.soft_token_placeholder,
    )

    return merged_embeddings

  def _encode_vision(
      self, images: jaxtyping.ArrayLike  # B N H W C
  ) -> jaxtyping.ArrayLike:  # B N P D
    # [FUNCTION DOC]
    # HELPER: runs the vision encoder on images [B, N, H, W, C] and projects the
    # result with embedder.encode_vision. Raises ValueError if no vision encoder.
    """Encode the images into the same space as the text embeddings."""
    if self.vision_encoder is None:
      raise ValueError('`vision_encoder` is None, cannot encode images.')

    # TODO(abheesht): Should the vision_encoder have `is_training = False`?
    soft_embeddings = self.vision_encoder(images=images)
    soft_embeddings = self.embedder.encode_vision(soft_embeddings)
    return soft_embeddings

  def get_model_input(self):
    # [FUNCTION DOC]
    # Returns dummy inputs for tracing: last_tokens and positions int32 ones
    # [2, 1], cache None, attention_mask bool ones [2, 1, 1], plus images float32
    # ones [2, 1, 896, 896, 3] when a vision encoder exists.
    """Returns a dummy model input for the transformer.

    This dummy input has a batch size compatible with FSDP sharding on a
    2-device axis.
    """
    dummy_batch_size = 2
    dummy_seq_len = 1
    inputs = {
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

    # Add images. Assume just one image per batch.
    if self.vision_encoder is not None:
      inputs['images'] = jnp.ones(
          (dummy_batch_size, 1, 896, 896, 3), dtype=jnp.float32
      )
    return inputs

  @property
  def embed_dim(self) -> int:
    # [FUNCTION DOC]
    # Property: embedding dimension.
    return self.embedder.embed_dim

  @property
  def num_embed(self) -> int:
    # [FUNCTION DOC]
    # Property: vocabulary size.
    return self.embedder.num_embed

  def get_attention_mask(
      self,
      tokens: jaxtyping.ArrayLike,  # (B, L)
      *,
      inputs_mask: jaxtyping.ArrayLike | None = None,  # (B, L)
  ):
    # [FUNCTION DOC]
    # Returns the [B, L, L] attention mask from utils.get_attention_mask, passing
    # the vision soft-token placeholder id (bidirectional image blocks) when a
    # vision config exists.
    # KNOWN ISSUE: the docstring says it returns positions too, but only the mask
    # is returned.
    """Returns the positions and attention mask for the transformer."""
    token_placeholder_id = (
        None if self.config.vision_config is None else
        self.config.vision_config.soft_token_placeholder
    )
    return utils.get_attention_mask(
        tokens,
        inputs_mask=inputs_mask,
        token_placeholder_id=token_placeholder_id,
    )

  @property
  def num_layers(self) -> int:
    # [FUNCTION DOC]
    # Property: number of decoder layers.
    return len(self.layers)
