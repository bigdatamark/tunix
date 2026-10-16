# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License');
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
# Flax NNX implementation of the Llama 3 decoder-only transformer (Llama 3,
# 3.1, 3.2 sizes) used by tunix for training and sampling. Components: sharding
# config, model config with named presets, an activation-sharding helper,
# Einsum projection, token embedder, rotary position embedding, RMSNorm,
# grouped-query attention with optional KV cache, gated SiLU MLP, decoder layer
# (pre-norm, residual), and the top-level Llama3 model.
# Module-level definitions: K_MASK (large negative constant used to mask
# attention logits), LayerCache (dict of name to array with keys 'k', 'v',
# 'end_index'), Cache (dict of 'layer_<i>' to LayerCache).
# Side effect at import: calls env_utils.setup_sharding_environment() to
# configure the JAX sharding environment.
# Shape letters used in names: B batch, T query length, S key length, D embed
# dim, N query heads, K kv heads, H head dim, F MLP hidden dim, V vocab.
"""LLama3 model."""

import dataclasses
import enum
from typing import Tuple

import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping
from tunix.generate.mappings import BackendMappingMixin
from tunix.utils import compat
from tunix.utils import env_utils

K_MASK = -2.3819763e38

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


env_utils.setup_sharding_environment()


class RematConfig(enum.Enum):
  # [CLASS DOC]
  # Enum of rematerialization (activation checkpointing) policies: NONE (store
  # all activations), BLOCK (remat each attention and MLP block separately),
  # DECODER (remat each whole decoder layer). Compared both against members and
  # their integer values by the modules below.
  NONE = enum.auto()  # No remat, all activations will be stored in HBM.
  BLOCK = enum.auto()  # Remat the entire attn block.
  DECODER = enum.auto()  # Remat the entire decoder layer.


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  # [CLASS DOC]
  # Frozen, slotted dataclass of PartitionSpec tuples (mesh axis name or None per
  # dimension) for each weight/activation: emb_vd (embedding [V,D]), emb_dv
  # (lm_head [D,V]), q_weight_dnh, kv_weight_dnh, o_weight_nhd, ffw_weight_df
  # (gate/up), ffw_weight_fd (down), rms_norm_weight, act_btd, act_btf, act_btnh.
  """Sharding configuration for Llama3 model."""

  emb_vd: Tuple[str | None, ...]
  emb_dv: Tuple[str | None, ...]
  q_weight_dnh: Tuple[str | None, ...]
  kv_weight_dnh: Tuple[str | None, ...]
  o_weight_nhd: Tuple[str | None, ...]
  ffw_weight_df: Tuple[str | None, ...]
  ffw_weight_fd: Tuple[str | None, ...]
  rms_norm_weight: Tuple[str | None, ...]
  act_btd: Tuple[str | None, ...]
  act_btf: Tuple[str | None, ...]
  act_btnh: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    # [FUNCTION DOC]
    # Static factory for the default FSDP + tensor-parallel layout over mesh axes
    # 'fsdp' and 'tp'. When is_sampling is True the 'fsdp' axis is dropped from
    # weight specs (set to None) and act_btd no longer shards D on 'tp'. Weights
    # shard their head/hidden/vocab dims on 'tp' and model dim on 'fsdp';
    # activations shard batch on 'fsdp' and heads/hidden on 'tp'; rms norm weight on
    # 'tp'. Returns a ShardingConfig. No exceptions or side effects.
    fsdp = 'fsdp' if not is_sampling else None

    return ShardingConfig(
        emb_vd=('tp', fsdp),
        emb_dv=(fsdp, 'tp'),
        q_weight_dnh=(fsdp, 'tp', None),
        kv_weight_dnh=(fsdp, 'tp', None),
        o_weight_nhd=('tp', None, fsdp),
        ffw_weight_df=(fsdp, 'tp'),
        ffw_weight_fd=('tp', fsdp),
        rms_norm_weight=('tp',),
        act_btd=('fsdp', None, None if is_sampling else 'tp'),
        act_btf=('fsdp', None, 'tp'),
        act_btnh=('fsdp', None, 'tp', None),
    )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
  # [CLASS DOC]
  # Frozen dataclass of Llama3 hyperparameters: num_layers, vocab_size,
  # embed_dim, hidden_dim (MLP), num_heads, head_dim, num_kv_heads, rope_theta,
  # norm_eps, shd_config (default training sharding), weight_tying (reuse input
  # embedding as LM head, used by Llama 3.2), remat_config. Classmethod presets
  # return configs matching published checkpoints; preset method names are the
  # config ids produced by tunix.models.naming.
  """Configuration for the Llama3 model."""

  num_layers: int
  vocab_size: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  rope_theta: int
  norm_eps: float
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  weight_tying: bool = False  # Llama3.2 features
  remat_config: RematConfig = RematConfig.NONE

  @classmethod
  def llama3p2_1b(cls):
    # [FUNCTION DOC]
    # Preset for Llama 3.2 1B: 16 layers, vocab 128256, embed 2048, MLP 8192, 32
    # heads of dim 64, 8 KV heads, eps 1e-5, rope theta 500000, tied embeddings.
    return cls(
        num_layers=16,
        vocab_size=128256,
        embed_dim=2048,
        hidden_dim=8192,
        num_heads=32,
        head_dim=64,
        num_kv_heads=8,
        norm_eps=1e-05,
        rope_theta=500_000,
        weight_tying=True,
    )

  @classmethod
  def llama3p2_1b_instruct(cls):
    # [FUNCTION DOC]
    # Preset alias: instruct variant uses the same architecture as llama3p2_1b.
    return cls.llama3p2_1b()

  # Llama3.2 3B
  @classmethod
  def llama3p2_3b(cls):
    # [FUNCTION DOC]
    # Preset for Llama 3.2 3B: 28 layers, embed 3072, MLP 8192, 24 heads of dim
    # 128, 8 KV heads, vocab 128256, eps 1e-5, rope theta 500000, tied embeddings.
    return cls(
        num_layers=28,  # ← from num_hidden_layers
        vocab_size=128256,  # ← from vocab_size
        embed_dim=3072,  # ← from hidden_size
        hidden_dim=8192,  # ← from intermediate_size
        num_heads=24,  # ← from num_attention_heads
        head_dim=128,  # ← from head_dim
        num_kv_heads=8,  # ← from num_key_value_heads
        norm_eps=1e-05,  # ← from rms_norm_eps
        rope_theta=500_000,  # ← from rope_theta
        weight_tying=True,
    )

  @classmethod
  def llama3p2_3b_instruct(cls):
    # [FUNCTION DOC]
    # Preset alias: instruct variant uses the same architecture as llama3p2_3b.
    return cls.llama3p2_3b()

  @classmethod
  def llama3p1_8b(cls):
    # [FUNCTION DOC]
    # Preset for Llama 3.1 8B: 32 layers, embed 4096, MLP 14336, 32 heads of dim
    # 128, 8 KV heads, vocab 128256, eps 1e-5, rope theta 500000, untied head.
    return cls(
        num_layers=32,
        vocab_size=128256,
        embed_dim=4096,
        hidden_dim=14336,
        num_heads=32,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-05,
        rope_theta=500_000,
        weight_tying=False,
    )

  @classmethod
  def llama3_70b(cls):
    # [FUNCTION DOC]
    # Preset for Llama 3 70B: 80 layers, embed 8192, MLP 28672, 64 heads of dim
    # 128, 8 KV heads, vocab 128256, eps 1e-5, rope theta 500000, untied head.
    return cls(
        num_layers=80,
        vocab_size=128256,
        embed_dim=8192,
        hidden_dim=28672,
        num_heads=64,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-05,
        rope_theta=500_000,
    )

  @classmethod
  def llama3p1_70b(cls):
    # [FUNCTION DOC]
    # Preset alias: Llama 3.1 70B uses the llama3_70b architecture.
    # KNOWN ISSUE: Llama 3.1 uses rope scaling (and the code ignores rope_theta
    # anyway, see apply_rope), so long-context behavior differs from the reference.
    return cls.llama3_70b()

  @classmethod
  def llama3p1_405b(cls):
    # [FUNCTION DOC]
    # Preset for Llama 3.1 405B: 126 layers, embed 16384, MLP 53248, 128 heads of
    # dim 128, 8 KV heads, vocab 128256, eps 1e-5, rope theta 500000.
    return cls(
        num_layers=126,
        vocab_size=128256,
        embed_dim=16384,
        hidden_dim=53248,
        num_heads=128,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-05,
        rope_theta=500_000,
    )


def shard(x: jnp.ndarray, s: Tuple[str, ...]):
  # [FUNCTION DOC]
  # HELPER: Applies a sharding constraint to activation x using the current
  # thread's physical mesh and PartitionSpec built from s. Returns x unchanged if
  # there is no active mesh or the first JAX device is a CPU. Returns the
  # constrained array (same values).
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )


class Einsum(nnx.Module):
  # [CLASS DOC]
  # Parameterized einsum: holds a weight 'w' of a given shape and an einsum
  # equation; calling it contracts the input with the weight.
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
    # Stores einsum_str and shape and creates parameter w initialized from a
    # standard-normal initializer using rngs.params(), annotated with the given
    # sharding spec. Side effect: consumes one RNG key.
    self.einsum_str = einsum_str
    self.shape = shape
    self.w = nnx.Param(
        nnx.initializers.normal()(rngs.params(), shape), sharding=sharding
    )

  @jax.named_scope('einsum')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Returns the einsum of the input x with w using einsum_str (named scope
    # 'einsum'). Shape errors raise from jnp.einsum.
    return jnp.einsum(self.einsum_str, x, self.w.value)


class Embedder(nnx.Module):
  # [CLASS DOC]
  # Token embedding table input_embedding of shape [vocab, embed] with encode
  # (lookup) and decode (tied-weight logits) operations.
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
    # Creates input_embedding with normal init and the emb_vd sharding, and stores
    # shd_config for activation sharding. Consumes one RNG key.
    self.input_embedding = nnx.Param(
        nnx.initializers.normal()(rngs.params(), (vocab_size, embed_dim)),
        sharding=shd_config.emb_vd,
    )
    self.shd_config = shd_config

  @jax.named_scope('embedder_encode')
  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Looks up embeddings for integer token ids x ([B, L]) returning [B, L, D],
    # sharded with act_btd. No scaling is applied.
    x = self.input_embedding[(x,)]
    x = shard(x, self.shd_config.act_btd)
    return x

  @jax.named_scope('embedder_decode')
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Computes logits by multiplying x ([..., D]) with the transposed embedding
    # table, returning [..., V]. Used when weight tying is enabled.
    return jnp.dot(x, self.input_embedding.value.T)


def apply_rope(
    inputs: jaxtyping.Array,  # [B, L]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = 1_000_000,
) -> jaxtyping.Array:
  # [FUNCTION DOC]
  # Rotary position embedding (non-interleaved, "rotate half" form). Inputs:
  # inputs [B, L, N, H] (the inline comment saying [B, L] is wrong), positions
  # [B, L], head_dim, rope_theta. Computes inverse frequencies with exponent
  # 2*i/head_dim for i in [0, head_dim/2), angles = position / timescale
  # broadcast to [B, L, 1, H/2], splits the last dim into two halves x1, x2 and
  # returns concat(x1*cos - x2*sin, x2*cos + x1*sin) cast to the input dtype.
  # KNOWN ISSUE: the default rope_theta is 1,000,000 and Attention.block calls
  # this without passing config.rope_theta, so Llama3's configured theta (500000)
  # is ignored and rotary frequencies differ from the pretrained model. Llama 3.1
  # rope scaling is also not implemented.
  """Applies RoPE."""
  fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
  timescale = rope_theta**fraction

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


class RMSNorm(nnx.Module):
  # [CLASS DOC]
  # Root-mean-square layer norm with a learned scale w of size dim.
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    # [FUNCTION DOC]
    # Creates w initialized to ones with rms_norm_weight sharding and stores
    # norm_eps. Consumes one RNG key (unused by ones init).
    self.w = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), dim),
        sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps

  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Computes rms = sqrt(mean(x^2 over last axis, in float32) + eps) and returns
    # w * x / rms cast back to x's original dtype.
    dtype = x.dtype
    rms = jnp.sqrt(
        jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True)
        + self.norm_eps
    )
    return jnp.astype(self.w * x / rms, dtype)


class Attention(nnx.Module):
  # [CLASS DOC]
  # Grouped-query self-attention with rotary embeddings and an optional
  # fixed-size KV cache. Parameters: q_proj [D,N,H], k_proj and v_proj [D,K,H],
  # o_proj [N,H,D] (all Einsum). No biases.
  """Attention module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Builds the four projections with shapes from config and sharding from
    # config.shd_config, stores config, n_rep = num_heads / num_kv_heads (unused)
    # and scale = head_dim^-0.5 (via the head_dim property). Consumes RNG keys.
    self.config = config
    self.shd_config = config.shd_config
    self.q_proj = Einsum(
        einsum_str='BTD,DNH->BTNH',
        shape=(config.embed_dim, config.num_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.q_weight_dnh,
    )
    self.k_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
    )
    self.v_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
    )
    self.o_proj = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, config.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=self.shd_config.o_weight_nhd,
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = self.head_dim**-0.5

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Core attention computation. Inputs: x [B,T,D], segment_pos [B,T] absolute
    # positions, cache (dict with 'k' and 'v' [B,S,K,H] and 'end_index' [B]) or
    # None, attn_mask boolean [B,T,S] (True = attend) or None.
    # Steps: project to q [B,T,N,H], k and v [B,T,K,H]; shard with act_btnh; apply
    # RoPE to q and k. If cache is given, write the new k/v into the cache buffers
    # at sequence offset end_index[0] modulo cache length (all batch rows share the
    # first row's end index) and use the full cache buffers as keys/values. Reshape
    # q to [B,T,K,N/K,H] for GQA, compute logits q·k times scale, reshape to
    # [B,N,T,S], replace masked positions with K_MASK (mask broadcast over heads),
    # softmax in float32 then cast to k dtype, weight values, reshape to [B,T,N,H]
    # and apply o_proj, shard as act_btd.
    # Returns (new_cache, output [B,T,D]) where new_cache has updated 'k', 'v' and
    # end_index + T, or None when no cache was given.
    # KNOWN ISSUE: rope_theta from config is not passed to apply_rope (see there).
    """Attention block."""
    seq_len = x.shape[1]

    query_proj = self.q_proj(x)
    key_proj = self.k_proj(x)
    value_proj = self.v_proj(x)

    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)

    query_proj = apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
    )
    key_proj = apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
    )

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

    b, t, qh, d = query_proj.shape
    _, s, kh, _ = key_proj.shape

    # GQA
    query_proj = query_proj.reshape((b, t, kh, qh // kh, d))
    attn = jnp.einsum('BTHGD,BSHD->BHGTS', query_proj, key_proj) * self.scale
    attn = attn.reshape((b, qh, t, s))

    if attn_mask is not None:
      attn = jnp.where((jnp.expand_dims(attn_mask, -3)), attn, K_MASK)

    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
        key_proj.dtype
    )

    attn = attn.reshape((b, kh, qh // kh, t, s))
    qkv = jnp.einsum('BHGTS,BSHD->BTHGD', attn, value_proj)
    qkv = qkv.reshape((b, t, qh, d))

    outputs = self.o_proj(qkv)
    outputs = shard(outputs, self.shd_config.act_btd)

    if cache is not None:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, outputs

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Runs block, wrapped in nnx.remat (on the unbound function with self as first
    # arg) when remat_config is BLOCK (member or its value). Same inputs/outputs as
    # block.
    if (
        self.config.remat_config == RematConfig.BLOCK
        or self.config.remat_config == RematConfig.BLOCK.value
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
    # Returns head dim H read from o_proj's shape (index 1).
    return self.o_proj.shape[1]

  @property
  def num_heads(self):
    # [FUNCTION DOC]
    # Returns number of query heads read from q_proj's shape.
    # KNOWN ISSUE: reads q_proj.shape[0], which is embed_dim D, not N (shape is
    # [D, N, H]); should be index 1.
    return self.q_proj.shape[0]

  @property
  def num_kv_heads(self):
    # [FUNCTION DOC]
    # Returns number of KV heads read from k_proj's shape (index 1).
    return self.k_proj.shape[1]


class MLP(nnx.Module):
  # [CLASS DOC]
  # Gated feed-forward network: down_proj(silu(gate_proj(x)) * up_proj(x)) using
  # bias-free nnx.Linear layers.
  """MLP module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Creates gate_proj and up_proj (D -> F, sharding ffw_weight_df) and down_proj
    # (F -> D, sharding ffw_weight_fd), kernels initialized to zeros (weights are
    # expected to be loaded from a checkpoint). Stores config and shd_config.
    self.config = config
    self.shd_config = config.shd_config
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_fd
        ),
    )

  def block(
      self,
      x: jaxtyping.Array,
  ) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Computes silu(gate(x)) * up(x), shards the activations with act_btf, applies
    # down_proj and returns [B,T,D].
    activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
    activations = shard(activations, self.shd_config.act_btf)
    outputs = self.down_proj(activations)
    return outputs

  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Runs block, under nnx.remat when remat_config is BLOCK. Returns the output.
    if (
        self.config.remat_config == RematConfig.BLOCK
        or self.config.remat_config == RematConfig.BLOCK.value
    ):
      return nnx.remat(self.block.__func__)(self, x)
    else:
      return self.block(x)


class DecoderLayer(nnx.Module):
  # [CLASS DOC]
  # Pre-norm transformer decoder layer: attention and MLP each preceded by an
  # RMSNorm and followed by a residual add.
  """DecoderLayer."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Creates input_layernorm, attn, mlp and post_attention_layernorm from config.
    self.config = config
    self.input_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
    )
    self.attn = Attention(
        config=config,
        rngs=rngs,
    )
    self.mlp = MLP(
        config=config,
        rngs=rngs,
    )
    self.post_attention_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
    )

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Computes h = x + attn(input_layernorm(x)) and out = h +
    # mlp(post_attention_layernorm(h)). Inputs as for Attention.block. Returns
    # (updated layer cache or None, out [B,T,D]).
    inputs_normalized = self.input_layernorm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
    )
    attn_output += x
    residual = attn_output
    attn_output = self.post_attention_layernorm(attn_output)
    outputs = residual + self.mlp(attn_output)
    return cache, outputs

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Runs block, under nnx.remat when remat_config is DECODER. Same signature.
    if (
        self.config.remat_config == RematConfig.DECODER
        or self.config.remat_config == RematConfig.DECODER.value
    ):
      return nnx.remat(self.block.__func__)(
          self, x, segment_pos, cache, attn_mask
      )
    else:
      return self.block(x, segment_pos, cache, attn_mask)


class Llama3(BackendMappingMixin, nnx.Module):
  # [CLASS DOC]
  # Top-level Llama3 causal LM. Mixes in BackendMappingMixin, which uses
  # BACKEND_PACKAGE_PATH (this module's name) to locate the package's
  # BACKEND_MAPPINGS for syncing weights to rollout engines. Submodules:
  # embedder, layers (list of DecoderLayer), final_norm, and lm_head (only when
  # weights are not tied).
  """Llama3 model."""

  BACKEND_PACKAGE_PATH = __name__

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Builds the embedder, num_layers decoder layers, the final RMSNorm and, if
    # weight_tying is False, an lm_head Einsum [D,V] with emb_dv sharding. Stores
    # config. Consumes RNG keys.
    self.config = config
    self.embedder = Embedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=config.shd_config,
    )
    self.layers = compat.ModuleList([
        DecoderLayer(config=config, rngs=rngs) for _ in range(config.num_layers)
    ])
    self.final_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        norm_eps=config.norm_eps,
        shd_config=config.shd_config,
    )
    if not config.weight_tying:
      self.lm_head = Einsum(
          einsum_str='BTD,DV->BTV',
          shape=(config.embed_dim, config.vocab_size),
          rngs=rngs,
          sharding=config.shd_config.emb_dv,
      )

  def get_model_input(self):
    # [FUNCTION DOC]
    # Returns a dummy keyword-argument dict for tracing/initializing the model:
    # input_tokens and positions of ones [1,128] int32, cache None, and
    # attention_mask of ones [1,1,128] bool.
    # KNOWN ISSUE: the mask shape [1,1,128] does not match the documented
    # [B, L, L'] (it relies on broadcasting).
    """Returns a dummy model input for the transformer."""
    dummy_batch_size = 1
    dummy_seq_len = 128
    return {
        'input_tokens': jnp.ones(
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

  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jaxtyping.Array,  # [B, L, L']
  ) -> tuple[jaxtyping.Array, Cache | None]:
    # [FUNCTION DOC]
    # Forward pass. Inputs: input_tokens [B,L] int, positions [B,L], cache (dict
    # 'layer_<i>' -> LayerCache) or None, attention_mask [B,L,L'] boolean.
    # Embeds tokens, runs each decoder layer with its cache entry (when cache is
    # truthy), collects updated caches into a new dict when a cache was given,
    # applies final norm and computes logits via tied embedding decode or lm_head.
    # Returns (logits [B,L,V], new_cache or None).
    # KNOWN ISSUE: an empty dict cache is treated as "no per-layer cache" but
    # still produces a dict of None entries.
    """Llama3 model.

    Args:
      input_tokens: input sequence of tokens.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    new_cache = None if cache is None else {}
    x = self.embedder.encode(input_tokens)

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

    if self.config.weight_tying:
      logits = self.embedder.decode(x)
    else:
      logits = self.lm_head(x)

    return logits, new_cache  # pytype: disable=bad-return-type

  @property
  def num_embed(self) -> int:
    # [FUNCTION DOC]
    # Property returning the embedding dimension from config.
    return self.config.embed_dim
