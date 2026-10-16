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
# Flax NNX implementation of the Qwen2 / Qwen2.5 decoder-only transformer (also
# used for DeepSeek-R1-Distill-Qwen). Same overall structure as the Llama3
# module (pre-norm decoder layers with GQA attention, gated SiLU MLP, RMSNorm,
# RoPE) with these differences: q/k/v projections have biases; computation
# dtype and parameter dtype are configurable; RoPE sin/cos are precomputed once
# per forward in high precision; an optional TPU splash (flash) attention path
# via shard_map is available for the no-cache case; sequence-parallel ('sp')
# sharding can be enabled.
# Module-level definitions: K_MASK (large negative attention-logit mask value),
# LayerCache (dict with 'k', 'v', 'end_index'), Cache (dict 'layer_<i>' ->
# LayerCache). Side effect at import: env_utils.setup_sharding_environment().
# Shape letters: B batch, T query length, S key length, D embed, N query heads,
# K kv heads, H head dim, F MLP hidden, V vocab.
"""Qwen2 model."""

import dataclasses
import enum
from functools import partial
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
from tunix.generate.mappings import BackendMappingMixin
from tunix.utils import compat
from tunix.utils import env_utils

env_utils.setup_sharding_environment()

K_MASK = -2.3819763e38

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class RematConfig(enum.Enum):
  # [CLASS DOC]
  # Enum of rematerialization policies: NONE, BLOCK (remat attention and MLP
  # blocks), DECODER (remat whole decoder layers). Compared against members and
  # their values.
  NONE = enum.auto()  #  No remat, all activations will be stored in HBM.
  BLOCK = enum.auto()  # Remat the entire attn block.
  DECODER = enum.auto()  # Remat the entire decoder layer.


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  # [CLASS DOC]
  # Frozen dataclass of PartitionSpec tuples for weights and activations: the
  # Llama3 set (emb_vd, emb_dv, q_weight_dnh, kv_weight_dnh, o_weight_nhd,
  # ffw_weight_df, ffw_weight_fd, rms_norm_weight, act_btd, act_btf, act_btnh)
  # plus exp_weight_cdf / exp_weight_cfd (expert weights, unused by Qwen2) and
  # qkv_bias.
  # KNOWN ISSUE: the docstring says "Qwen3" though this is the Qwen2 module.
  """Sharding configuration for Qwen3 model."""

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
  exp_weight_cdf: Tuple[str | None, ...]
  exp_weight_cfd: Tuple[str | None, ...]
  qkv_bias: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False, enable_sp: bool = False):
    # [FUNCTION DOC]
    # Static factory for the default layout over mesh axes 'fsdp', 'tp' and
    # optionally 'sp'. is_sampling drops 'fsdp' from weight specs and the 'tp'
    # sharding of act_btd's last dim. enable_sp (only when not sampling) makes the
    # model-dim weight axis the tuple ('fsdp', 'sp') and shards activation sequence
    # dims on 'sp'. Expert weights shard expert dim on 'fsdp' and hidden on 'tp';
    # qkv_bias and rms norm weight on 'tp'. Returns a ShardingConfig.
    fsdp = 'fsdp' if not is_sampling else None
    sp = 'sp' if (not is_sampling and enable_sp) else None
    fsdp = (fsdp, sp) if fsdp and sp else fsdp

    return ShardingConfig(
        emb_vd=('tp', fsdp),
        emb_dv=(fsdp, 'tp'),
        q_weight_dnh=(fsdp, 'tp', None),
        kv_weight_dnh=(fsdp, 'tp', None),
        o_weight_nhd=('tp', None, fsdp),
        ffw_weight_df=(fsdp, 'tp'),
        ffw_weight_fd=('tp', fsdp),
        rms_norm_weight=('tp',),
        act_btd=('fsdp', sp, None if is_sampling else 'tp'),
        act_btf=('fsdp', sp, 'tp'),
        act_btnh=('fsdp', sp, 'tp', None),
        exp_weight_cdf=('fsdp', None, 'tp'),
        exp_weight_cfd=('fsdp', 'tp', None),
        qkv_bias=('tp',),
    )


@dataclasses.dataclass(slots=True)
class ModelConfig:
  # [CLASS DOC]
  # Mutable slotted dataclass of Qwen2 hyperparameters: num_layers, vocab_size,
  # embed_dim, hidden_dim, num_heads, head_dim, num_kv_heads, rope_theta,
  # norm_eps, use_tied_embedding (embedding reused as LM head), shd_config,
  # remat_config, dtype (computation dtype), param_dtype, use_flash_attention
  # (splash attention when no cache) and flash_attention_block_size. Classmethod
  # presets are named after tunix config ids.
  """Configuration for the Qwen2 model."""

  num_layers: int
  vocab_size: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  rope_theta: int
  norm_eps: float
  use_tied_embedding: bool = False
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  use_flash_attention: bool = False
  flash_attention_block_size: int = 1024

  # qwen2.5-0.5B and qwen2.5-coder-0.5B share the same config.
  @classmethod
  def qwen2p5_0p5b(cls):  # qwen2.5-0.5B
    # [FUNCTION DOC]
    # Preset for Qwen2.5-0.5B (also coder): 24 layers, vocab 151936, embed 896,
    # MLP 4864, 14 heads of dim 64, 2 KV heads, eps 1e-6, rope theta 1e6, tied.
    return cls(
        num_layers=24,
        vocab_size=151936,
        embed_dim=896,
        hidden_dim=4864,
        num_heads=14,
        head_dim=64,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=True,
    )

  @classmethod
  def qwen2p5_0p5b_instruct(cls):
    # [FUNCTION DOC]
    # Preset alias of qwen2p5_0p5b (instruct).
    return cls.qwen2p5_0p5b()

  @classmethod
  def qwen2p5_coder_0p5b(cls):
    # [FUNCTION DOC]
    # Preset alias of qwen2p5_0p5b (coder).
    return cls.qwen2p5_0p5b()

  # DeepSeek-R1-Distill-Qwen-1.5B
  @classmethod
  def deepseek_r1_distill_qwen_1p5b(cls):
    # [FUNCTION DOC]
    # Preset for DeepSeek-R1-Distill-Qwen-1.5B: 28 layers, vocab 151936, embed
    # 1536, MLP 8960, 12 heads of dim 128, 2 KV heads, eps 1e-6, rope theta 10000,
    # untied LM head.
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=1536,
        hidden_dim=8960,
        num_heads=12,
        head_dim=128,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=10000,
        use_tied_embedding=False,
    )

  @classmethod
  def qwen2p5_1p5b(cls):  # qwen2.5-1.5B
    # [FUNCTION DOC]
    # Preset for Qwen2.5-1.5B: 28 layers, vocab 151936, embed 1536, MLP 8960, 12
    # heads of dim 128, 2 KV heads, eps 1e-6, rope theta 1e6, tied.
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=1536,
        hidden_dim=8960,
        num_heads=12,
        head_dim=128,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=True,
    )

  @classmethod
  def qwen2p5_1p5b_instruct(cls):
    # [FUNCTION DOC]
    # Preset alias of qwen2p5_1p5b (instruct).
    return cls.qwen2p5_1p5b()

  @classmethod
  def qwen2p5_math_1p5b(cls):  # qwen2.5-math-1.5B
    # [FUNCTION DOC]
    # Preset for Qwen2.5-Math-1.5B: as qwen2p5_1p5b but rope theta 10000.
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=1536,
        hidden_dim=8960,
        num_heads=12,
        head_dim=128,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=10000,
        use_tied_embedding=True,
    )

  # qwen2.5-coder-3B and qwen2.5-3B share the same config.
  @classmethod
  def qwen2p5_3b(cls):
    # [FUNCTION DOC]
    # Preset for Qwen2.5-3B (also coder): 36 layers, vocab 151936, embed 2048, MLP
    # 11008, 16 heads of dim 128, 2 KV heads, eps 1e-6, rope theta 1e6, tied.
    return cls(
        num_layers=36,
        vocab_size=151936,
        embed_dim=2048,
        hidden_dim=11008,
        num_heads=16,
        head_dim=128,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=True,
    )

  @classmethod
  def qwen2p5_3b_instruct(cls):
    # [FUNCTION DOC]
    # Preset alias of qwen2p5_3b (instruct).
    return cls.qwen2p5_3b()

  @classmethod
  def qwen2p5_coder_3b(cls):
    # [FUNCTION DOC]
    # Preset alias of qwen2p5_3b (coder).
    return cls.qwen2p5_3b()

  # qwen2.5-7B and qwen2.5-coder-7B share the same config.
  @classmethod
  def qwen2p5_7b(cls):
    # [FUNCTION DOC]
    # Preset for Qwen2.5-7B (also coder): 28 layers, vocab 152064, embed 3584, MLP
    # 18944, 28 heads of dim 128, 4 KV heads, eps 1e-6, rope theta 1e6, untied.
    return cls(
        num_layers=28,
        vocab_size=152064,
        embed_dim=3584,
        hidden_dim=18944,
        num_heads=28,
        head_dim=128,
        num_kv_heads=4,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=False,
    )

  @classmethod
  def qwen2p5_7b_instruct(cls):
    # [FUNCTION DOC]
    # Preset alias of qwen2p5_7b (instruct).
    return cls.qwen2p5_7b()

  @classmethod
  def qwen2p5_coder_7b(cls):
    # [FUNCTION DOC]
    # Preset alias of qwen2p5_7b (coder).
    return cls.qwen2p5_7b()

  # TODO(linchai): add other qwen2.5 model configs.


def shard(x: jnp.ndarray, s: Tuple[str, ...]):
  # [FUNCTION DOC]
  # HELPER: Constrains activation x to PartitionSpec s on the current thread's
  # physical mesh; returns x unchanged when there is no mesh or devices are CPU.
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )


class Einsum(nnx.Module):
  # [CLASS DOC]
  # Parameterized einsum with weight w; casts input and weight to the
  # computation dtype before contracting.
  """Einsum is a convenience module for parameterized tensor multiplication."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    # [FUNCTION DOC]
    # Stores einsum_str, shape, dtype, param_dtype and creates w of the given shape
    # in param_dtype using Glorot-uniform init (one RNG key) with the given sharding.
    self.einsum_str = einsum_str
    self.shape = shape
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.w = nnx.Param(
        nnx.initializers.glorot_uniform()(
            rngs.params(), shape, dtype=param_dtype
        ),
        sharding=sharding,
    )

  @jax.named_scope('einsum')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Casts x and w to self.dtype and returns their einsum per einsum_str.
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(self.w.value, self.dtype)
    return jnp.einsum(self.einsum_str, x, w)


class Embedder(nnx.Module):
  # [CLASS DOC]
  # Token embedding table [V, D] with encode (lookup) and decode (tied logits).
  """Embedder module."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    # [FUNCTION DOC]
    # Creates input_embedding with standard normal values in param_dtype drawn from
    # rngs.params, with out_sharding emb_vd; stores shd_config and dtype.
    self.input_embedding = nnx.Param(
        rngs.params.normal((vocab_size, embed_dim), dtype=param_dtype),
        out_sharding=shd_config.emb_vd,
    )
    self.shd_config = shd_config
    self.dtype = dtype

  @jax.named_scope('embedder_encode')
  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Looks up embeddings for token ids [B,T], casts to dtype, shards with act_btd
    # and returns [B,T,D].
    x = self.input_embedding[(x,)]
    x = jnp.astype(x, self.dtype)
    x = shard(x, self.shd_config.act_btd)
    return x

  @jax.named_scope('embedder_decode')
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Casts x and the table to dtype and returns x times the transposed table,
    # i.e. logits [..., V].
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(self.input_embedding.value, self.dtype)
    return jnp.dot(x, w.T)


def _generate_pos_embeddings(
    positions: jax.Array,
    features: int,
    rope_theta: int,
) -> tuple[jax.Array, jax.Array]:
  # [FUNCTION DOC]
  # HELPER: Precomputes RoPE sinusoids. Inputs: positions [B,T], features (head
  # dim), rope_theta. Inverse frequencies are 1 / theta^(2i/features) for i in
  # [0, features/2). Computes the outer product positions x frequency with
  # HIGHEST precision (bf16 rounding of positions would be catastrophic) giving
  # [B,T,features/2], and returns (sin, cos) of it.
  # KNOWN ISSUE: the docstring mentions min/max timescales and says the outputs
  # are [length, features//2]; actually they are [B, T, features//2]; the arg is
  # misspelled 'postions' in the docstring.
  """Generate Sin/Cos for Rotary Embeddings.

  Generates sinusoids at (features//2) different timescales, where the
  timescales form a geometric series from min_timescale to max_timescale
  (max_timescale is not included, but would be the next element in the series).

  Sinusoids are evaluated at integer positions i in [0, length).

  The outputs are computed as:


  sin[b, t, j] = sin(rope_pos[b, t] / timescale[j])
  cos[b, t, j] = cos(rope_pos[b, t] / timescale[j])

  Args:
      postions: [batch, time]
      features: head_dim.
      rope_theta: the rope_theta parameter.

  Returns:
      sin: a float32 array with shape [length, features // 2]
      cos: a float32 array with shape [length, features // 2]
  """
  # Forked from: flaxformer/components/embedding.py;l=592
  fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
  timescale = rope_theta**fraction
  rotational_frequency = 1.0 / timescale
  # Must use high precision einsum here, since rounding off to a bfloat16 is
  # catastrophic. bfloat16 rounds 257 to 256, but sin(257) is very different
  # from sin(256).
  sinusoid_inp = jnp.einsum(
      'BT,k->BTk',
      positions,
      rotational_frequency,
      precision=jax.lax.Precision.HIGHEST,
  )
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def apply_rotary_embedding(
    x: jax.Array, sin: jax.Array, cos: jax.Array
) -> jax.Array:
  # [FUNCTION DOC]
  # Applies rotate-half RoPE: x is [B,T,N,H] and sin/cos are [B,T,H/2]
  # (asserted 4D and 3D, AssertionError otherwise). Splits x's last dim into
  # halves x1, x2, broadcasts sin/cos over the head axis, and returns
  # concat(x1*cos - x2*sin, x2*cos + x1*sin). The result dtype follows JAX
  # promotion of x with float32 sin/cos (not cast back).
  assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  # [B, T, head_dim] -> [B, h, T, head_dim]
  sin, cos = sin[:, :, None, :], cos[:, :, None, :]
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class RMSNorm(nnx.Module):
  # [CLASS DOC]
  # RMSNorm with learned scale w (ones init) computed in float32.
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    # [FUNCTION DOC]
    # Creates w = ones(dim) in param_dtype with out_sharding rms_norm_weight and
    # stores norm_eps and dtype. Does not use an RNG.
    self.w = nnx.Param(
        jnp.ones(dim, param_dtype),
        out_sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps
    self.dtype = dtype

  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Casts x to float32, divides by sqrt(mean(x^2 over last axis) + eps),
    # multiplies by w (float32) and casts to self.dtype.
    x = jnp.astype(x, jnp.float32)
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.norm_eps)
    return jnp.astype(
        jnp.astype(self.w.value, jnp.float32) * (x / rms),
        self.dtype,
    )


class Attention(nnx.Module):
  # [CLASS DOC]
  # Grouped-query self-attention with q/k/v biases, precomputed RoPE, optional
  # KV cache, optional sequence-packing segment ids and optional TPU splash
  # attention. Parameters: q_proj [D,N,H], k_proj/v_proj [D,K,H], o_proj
  # [N,H,D] (Einsum), q_bias [N*H], k_bias and v_bias [K*H] (flat vectors).
  """Attention module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Creates the projections with config dtypes and shardings, stores config,
    # shd_config, n_rep (unused) and scale = head_dim^-0.5, and creates the three
    # bias vectors initialized to zeros in param_dtype with qkv_bias sharding.
    self.config = config
    self.shd_config = config.shd_config
    self.q_proj = Einsum(
        einsum_str='BTD,DNH->BTNH',
        shape=(config.embed_dim, config.num_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.q_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.k_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.v_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.o_proj = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, config.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=self.shd_config.o_weight_nhd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = self.head_dim**-0.5
    self.q_bias = nnx.Param(
        jnp.zeros(config.num_heads * config.head_dim, dtype=config.param_dtype),
        out_sharding=self.shd_config.qkv_bias,
    )
    self.k_bias = nnx.Param(
        jnp.zeros(
            config.num_kv_heads * config.head_dim, dtype=config.param_dtype
        ),
        out_sharding=self.shd_config.qkv_bias,
    )
    self.v_bias = nnx.Param(
        jnp.zeros(
            config.num_kv_heads * config.head_dim, dtype=config.param_dtype
        ),
        out_sharding=self.shd_config.qkv_bias,
    )

  def block(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Core attention. Inputs: x [B,T,D]; cache (dict 'k','v' [B,S,K,H],
    # 'end_index' [B]) or None; attn_mask boolean [B,T,S] or None; sin, cos
    # [B,T,H/2]; segment_ids [B,T] or None (tokens attend only within equal ids).
    # Steps: project q, k, v; add the flat biases (cast to compute dtype) after
    # flattening heads, then restore [B,T,heads,H]; shard with act_btnh; apply RoPE
    # to q and k. With a cache, write new k/v at offset end_index[0] mod cache
    # length (shared across batch) and use the full buffers as keys/values.
    # Flash path (use_flash_attention and T > 1): transpose to [B,heads,T,H],
    # pre-scale q, build a causal multi-head splash mask of size (T,T) for all
    # query heads, block sizes all equal to flash_attention_block_size, compute
    # head/sequence shard counts from the mesh axes named in act_btnh, build the
    # splash MHA kernel and run it per-batch-element (vmap) inside shard_map with q
    # sharded and k/v unsharded on sequence; with segment_ids, pass them as splash
    # SegmentIds; transpose back to [B,T,N,H].
    # Default path: GQA einsum logits [B,K,N/K,T,S] times scale; mask with K_MASK
    # where attn_mask is False and, if segment ids are given, where q and kv
    # segment ids differ; float32 softmax cast to k dtype; weighted sum of values;
    # reshape to [B,T,N,H].
    # Finally o_proj and shard to act_btd. Returns (new_cache with updated k, v
    # and end_index + T, or None; outputs [B,T,D]).
    # Raises: AssertionError from RoPE on bad ranks; splash/shard_map errors when
    # shapes are not divisible by block sizes or mesh axes.
    # KNOWN ISSUE: the flash path ignores attn_mask (padding masks are not applied,
    # only causal and segment masking) and assumes keys have the same length as the
    # queries, so it is wrong when a cache is present with T > 1 (prefill into a
    # longer cache). In the default path the segment mask is built as [B,T,T] and
    # only matches the logits when S == T (i.e. no longer cache).
    """Attention block."""
    seq_len = x.shape[1]

    query_proj = self.q_proj(x)
    b, t, n, h = query_proj.shape
    query_proj = jnp.reshape(query_proj, (b, t, n * h)) + self.q_bias.astype(
        self.config.dtype
    )
    query_proj = jnp.reshape(query_proj, (b, t, n, h))
    key_proj = self.k_proj(x)
    _, s, k, h = key_proj.shape
    key_proj = jnp.reshape(key_proj, (b, s, k * h)) + self.k_bias.astype(
        self.config.dtype
    )
    key_proj = jnp.reshape(key_proj, (b, s, k, h))
    value_proj = self.v_proj(x)
    value_proj = jnp.reshape(value_proj, (b, s, k * h)) + self.v_bias.astype(
        self.config.dtype
    )
    value_proj = jnp.reshape(value_proj, (b, s, k, h))

    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)

    query_proj = apply_rotary_embedding(
        query_proj,
        sin,
        cos,
    )
    key_proj = apply_rotary_embedding(
        key_proj,
        sin,
        cos,
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
      cache_value_proj = value_proj
      cache_key_proj = key_proj
    else:
      cache_value_proj = value_proj
      cache_key_proj = key_proj

    b, t, qh, d = query_proj.shape
    _, _, kh, _ = key_proj.shape

    # NB: flash attention doesn't work for decoding step
    if self.config.use_flash_attention and seq_len > 1:
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)

      query_proj = query_proj * self.scale

      mesh = pxla.thread_resources.env.physical_mesh
      causal_mask = mask_lib.CausalMask((seq_len, seq_len))
      multi_head_mask = mask_lib.MultiHeadMask([causal_mask for _ in range(qh)])

      block_sizes = splash.BlockSizes(
          block_q=self.config.flash_attention_block_size,
          block_kv=self.config.flash_attention_block_size,
          block_q_dkv=self.config.flash_attention_block_size,
          block_kv_dkv=self.config.flash_attention_block_size,
          block_kv_dkv_compute=self.config.flash_attention_block_size,
          block_q_dq=self.config.flash_attention_block_size,
          block_kv_dq=self.config.flash_attention_block_size,
      )

      shd_b, shd_t, shd_n, shd_h = self.shd_config.act_btnh
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

      # Segment IDs are used to implement sequence packing
      if segment_ids is not None:
        seg_spec = P(shd_b, shd_t)
        unsharded_seg_spec = P(shd_b, None)
        splash_segment_ids = splash.SegmentIds(
            q=segment_ids, kv=segment_ids
        )

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                kernel_spec,
                shd_spec,
                unsharded_seq,
                unsharded_seq,
                seg_spec,
                unsharded_seg_spec,
            ),
            out_specs=shd_spec,
            check_rep=False,
        )
        def sharded_splash_attn(
            kernel, q_block, k_block, v_block, q_seg_block, kv_seg_block
        ):
          # [FUNCTION DOC] HELPER: Per-shard flash (splash) attention used when segment ids are present.
          # Behavior: wraps the per-device query/key/value blocks and the query/key-value segment-id blocks into a splash SegmentIds
          # record, then maps the precompiled splash kernel over the batch dimension of the blocks.
          # Inputs: kernel (precompiled splash multi-head attention kernel), q_block/k_block/v_block (per-shard [batch, heads, seq, head_dim]
          # arrays), q_seg_block and kv_seg_block (per-shard [batch, seq] segment ids; the key/value ids are unsharded along sequence).
          # Outputs: per-shard attention output with the same layout as q_block. Exceptions: none raised directly. Side effects: none.
          # Decorated with a shard_map over the mesh so it runs once per device shard.
          seg_ids = splash.SegmentIds(q=q_seg_block, kv=kv_seg_block)
          return jax.vmap(kernel)(
              q_block, k_block, v_block, segment_ids=seg_ids
          )

        qkv = sharded_splash_attn(
            splash_attn_kernel,
            query_proj,
            key_proj,
            value_proj,
            splash_segment_ids.q,
            splash_segment_ids.kv,
        )
      else:

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(kernel_spec, shd_spec, unsharded_seq, unsharded_seq),
            out_specs=shd_spec,
            check_rep=False,
        )
        def sharded_splash_attn(kernel, q_block, k_block, v_block):
          # [FUNCTION DOC] HELPER: Per-shard flash (splash) attention used when no segment ids are given.
          # Behavior: maps the precompiled splash kernel over the batch dimension of the per-device query/key/value blocks.
          # Inputs: kernel (precompiled splash attention kernel), q_block (sharded queries), k_block/v_block (keys/values unsharded along sequence).
          # Outputs: per-shard attention output with the same layout as q_block. Exceptions: none raised directly. Side effects: none.
          # KNOWN ISSUE: no attention mask is applied on this path, so padding positions are not masked (attn_mask is ignored).
          return jax.vmap(kernel)(q_block, k_block, v_block)

        qkv = sharded_splash_attn(
            splash_attn_kernel, query_proj, key_proj, value_proj
        )

      # Transpose back
      qkv = qkv.transpose(0, 2, 1, 3)  # pytype: disable=attribute-error
    else:
      # GQA
      query_proj = query_proj.reshape((b, t, kh, qh // kh, d))
      attn = jnp.einsum('BTHGD,BSHD->BHGTS', query_proj, key_proj) * self.scale

      if attn_mask is not None:
        attn = jnp.where(attn_mask[:, None, None, :, :], attn, K_MASK)

      if segment_ids is not None:
        seg_mask = (
            segment_ids[:, :, None] == segment_ids[:, None, :]
        )
        attn = jnp.where(seg_mask[:, None, None, :, :], attn, K_MASK)

      attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
          key_proj.dtype
      )

      qkv = jnp.einsum('BHGTS,BSHD->BTHGD', attn, value_proj)
      qkv = qkv.reshape((b, t, qh, d))

    outputs = self.o_proj(qkv)
    outputs = shard(outputs, self.shd_config.act_btd)

    if cache is not None:
      new_cache = {
          'v': cache_value_proj,
          'k': cache_key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, outputs

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Runs block, under nnx.remat (unbound function, self first) when
    # remat_config is BLOCK. Same inputs/outputs as block.
    if (
        self.config.remat_config == RematConfig.BLOCK
        or self.config.remat_config == RematConfig.BLOCK.value
    ):
      # nnx.remat needs to be applied to the unbound function and take self
      # as the first argument.
      return nnx.remat(self.block.__func__)(
          self, x, cache, attn_mask, sin, cos, segment_ids
      )
    else:
      return self.block(x, cache, attn_mask, sin, cos, segment_ids)

  @property
  def head_dim(self):
    # [FUNCTION DOC]
    # Returns head dim H from o_proj's shape (index 1).
    return self.o_proj.shape[1]

  @property
  def num_heads(self):
    # [FUNCTION DOC]
    # Returns number of query heads from q_proj's shape.
    # KNOWN ISSUE: uses index 0, which is embed_dim D for shape [D,N,H].
    return self.q_proj.shape[0]

  @property
  def num_kv_heads(self):
    # [FUNCTION DOC]
    # Returns number of KV heads from k_proj's shape (index 1).
    return self.k_proj.shape[1]


class MLP(nnx.Module):
  # [CLASS DOC]
  # Gated SiLU feed-forward network: down(silu(gate(x)) * up(x)) with bias-free
  # nnx.Linear layers in config dtype/param_dtype.
  """MLP module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Creates gate_proj, up_proj (D -> F, ffw_weight_df) and down_proj (F -> D,
    # ffw_weight_fd) with zero-initialized kernels (to be overwritten by loading).
    self.config = config
    self.shd_config = config.shd_config
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_fd
        ),
    )

  def block(
      self,
      x: jaxtyping.Array,
  ) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Computes silu(gate(x)) * up(x), shards with act_btf, returns down_proj of it.
    activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
    activations = shard(activations, self.shd_config.act_btf)
    outputs = self.down_proj(activations)
    return outputs

  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # [FUNCTION DOC]
    # Runs block, under nnx.remat when remat_config is BLOCK.
    if (
        self.config.remat_config == RematConfig.BLOCK
        or self.config.remat_config == RematConfig.BLOCK.value
    ):
      return nnx.remat(self.block.__func__)(self, x)
    else:
      return self.block(x)


class DecoderLayer(nnx.Module):
  # [CLASS DOC]
  # Pre-norm decoder layer: residual attention followed by residual MLP, each
  # preceded by an RMSNorm.
  """DecoderLayer."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Creates input_layernorm, attn, post_attention_layernorm and mlp from config
    # (norms use no RNG).
    self.config = config
    self.input_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.attn = Attention(
        config=config,
        rngs=rngs,
    )
    self.post_attention_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.mlp = MLP(
        config=config,
        rngs=rngs,
    )

  def block(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      sin,
      cos,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Computes h = x + attn(input_layernorm(x), ...) and out = h +
    # mlp(post_attention_layernorm(h)), forwarding cache, mask, sin/cos and
    # segment ids. Returns (updated cache or None, out [B,T,D]).
    inputs_normalized = self.input_layernorm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        cache,
        attn_mask=attn_mask,
        sin=sin,
        cos=cos,
        segment_ids=segment_ids,
    )
    attn_output += x
    residual = attn_output
    attn_output = self.post_attention_layernorm(attn_output)
    outputs = self.mlp(attn_output)
    outputs = residual + outputs
    return cache, outputs

  def __call__(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      sin,
      cos,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    # [FUNCTION DOC]
    # Runs block, under nnx.remat when remat_config is DECODER. Same signature.
    if (
        self.config.remat_config == RematConfig.DECODER
        or self.config.remat_config == RematConfig.DECODER.value
    ):
      return nnx.remat(self.block.__func__)(
          self,
          x,
          cache,
          attn_mask=attn_mask,
          sin=sin,
          cos=cos,
          segment_ids=segment_ids,
      )
    else:
      return self.block(
          x,
          cache,
          attn_mask=attn_mask,
          sin=sin,
          cos=cos,
          segment_ids=segment_ids,
      )


class Qwen2(BackendMappingMixin, nnx.Module):
  # [CLASS DOC]
  # Top-level Qwen2/Qwen2.5 causal LM. BackendMappingMixin uses
  # BACKEND_PACKAGE_PATH (this module's name) to find the package's
  # BACKEND_MAPPINGS for rollout weight sync. Submodules: embedder, layers,
  # final_norm, and lm_head when embeddings are not tied.
  """Qwen2.5 model."""

  BACKEND_PACKAGE_PATH = __name__

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Builds embedder, num_layers DecoderLayers, final RMSNorm and, unless
    # use_tied_embedding, an lm_head Einsum [D,V] with emb_dv sharding. Consumes
    # RNG keys.
    self.config = config
    self.embedder = Embedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.layers = compat.ModuleList([
        DecoderLayer(config=config, rngs=rngs) for _ in range(config.num_layers)
    ])
    self.final_norm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    if not self.config.use_tied_embedding:
      self.lm_head = Einsum(
          einsum_str='BTD,DV->BTV',
          shape=(config.embed_dim, config.vocab_size),
          rngs=rngs,
          sharding=config.shd_config.emb_dv,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jaxtyping.Array | None = None,  # [B, L, L']
      output_hidden_states: bool = False,
      segment_ids: jaxtyping.Array | None = None,  # [B, L]
  ) -> tuple[jaxtyping.Array, Cache | None]:
    # [FUNCTION DOC]
    # Forward pass. Inputs: input_tokens [B,L], positions [B,L], cache or None,
    # attention_mask [B,L,L'] or None, output_hidden_states (if True, sows the
    # final-normed hidden states as an nnx.Intermediate named
    # 'all_hidden_states'), segment_ids [B,L] or None for packed sequences.
    # Embeds tokens, computes RoPE sin/cos from positions (cast to activation
    # dtype), runs each layer with its 'layer_<i>' cache when cache is truthy and
    # collects updated caches into a new dict when a cache was given, applies the
    # final norm, computes logits via tied embedding or lm_head and returns
    # (logits cast to float32 [B,L,V], new cache or None).
    # Side effects: sow when output_hidden_states.
    """Qwen2 model.

    Args:
      input_tokens: input sequence of tokens.
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
    x = self.embedder.encode(input_tokens)
    sin, cos = _generate_pos_embeddings(
        positions, self.config.head_dim, self.config.rope_theta
    )
    sin, cos = sin.astype(x.dtype), cos.astype(x.dtype)

    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(
          x,
          layer_cache,
          attention_mask,
          sin,
          cos,
          segment_ids=segment_ids,
      )
      if cache is not None:
        new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)
    # Qwen2.5 0.5B-3B uses tied embedding, sharing weights for input and output.
    if self.config.use_tied_embedding:
      logits = self.embedder.decode(x)
    else:
      logits = self.lm_head(x)

    return jnp.astype(logits, jnp.float32), new_cache  # pytype: disable=bad-return-type

  def get_model_input(self):
    # [FUNCTION DOC]
    # Returns dummy forward kwargs: tokens and positions of ones [2,1] int32
    # (batch 2 for FSDP on two devices), cache None, attention_mask ones [2,1,1].
    """Returns a dummy model input for the transformer.

    This dummy input has a batch size compatible with FSDP sharding on a
    2-device axis.
    """
    dummy_batch_size = 2
    dummy_seq_len = 1
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
