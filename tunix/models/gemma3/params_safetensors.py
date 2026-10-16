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
# Loads Gemma 3 weights from Hugging Face safetensors checkpoints (text-only
# or multimodal, with an optional 'language_model.' key prefix) into the NNX
# model.Gemma3 via the generic safetensors_loader. Provides the regex key
# mapping (with transpose/reshape rules), a stateful q/k/v fusing
# preprocess function and the create_model_from_safe_tensors entry point.
"""Loads Gemma3 parameters from safetensors files."""

import re

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models.gemma3 import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  # [FUNCTION DOC]
  # HELPER: returns {HF key regex: (NNX dotted target, transform)} where the
  # transform is None or (axis permutation, reshape shape or None).
  # Text weights: embed_tokens -> embedder.input_embedding; q/k/v_proj ->
  # temporary 'tmp.layers.N.attn.q|k|v' transposed and reshaped to
  # [D, heads, head_dim]; o_proj -> attn_vec_einsum.w transposed and reshaped
  # to [N, H, D]; gate/up/down_proj -> mlp kernels transposed; input_layernorm
  # -> pre_attention_norm; post_attention_layernorm -> post_attention_norm;
  # pre/post feedforward layernorm (several spellings) -> pre/post_ffw_norm;
  # q_norm/k_norm -> attn._query_norm/_key_norm; model.norm -> final_norm.
  # lm_head, attention/norm biases, rotary buffers and fused qkv_proj weights
  # are routed to 'unused.*' targets (ignored).
  # When cfg.vision_config is set, adds SigLIP mappings: patch embedding
  # weight permuted [out, in, kh, kw] -> [kh, kw, in, out]; position
  # embedding; per-layer q/k/v/out projection kernels (transposed) and biases;
  # layer_norm1/2 -> ln1/ln2; mlp fc1/fc2 (kernels transposed); post_layernorm
  # -> encoder_norm; multi-modal projector weight and soft-embedding norm ->
  # embedder.mm_input_projection.w / mm_soft_embedding_norm.scale.
  # KNOWN ISSUE: the HF position embedding is [num_patches, width] while the
  # NNX parameter is [1, num_patches, width], and no reshape rule is given.
  """Mapping of torch_keys to (nnx_keys, (permute_rule, reshape_rule))."""
  mapping = {
      r"(?:language_model\.)?model\.embed_tokens\.weight": (
          "embedder.input_embedding", None
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"tmp.layers.\1.attn.q",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"tmp.layers.\1.attn.k",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"tmp.layers.\1.attn.v",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
          r"layers.\1.attn.attn_vec_einsum.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
          r"layers.\1.mlp.gate_proj.kernel",
          ((1, 0), None),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
          r"layers.\1.mlp.up_proj.kernel",
          ((1, 0), None),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
          r"layers.\1.mlp.down_proj.kernel",
          ((1, 0), None),
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\1.pre_attention_norm.scale",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attention_norm.scale",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.(post_feedforward_layernorm|post_ffn_layernorm|post_ffw_layernorm)\.weight": (
          r"layers.\1.post_ffw_norm.scale",
          None,
      ),
      r"(?:language_model\.)?model\.norm\.weight": ("final_norm.scale", None),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
          r"layers.\1.attn._query_norm.scale",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
          r"layers.\1.attn._key_norm.scale",
          None,
      ),
      r"lm_head\.weight": ("unused.lm_head.weight", None),
      r"lm_head\.bias": ("unused.lm_head.bias", None),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.bias": (
          r"unused.layers.\1.attn.\2.bias",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.input_layernorm\.bias": (
          r"unused.layers.\1.input_layernorm.bias",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.post_attention_layernorm\.bias": (
          r"unused.layers.\1.post_attention_layernorm.bias",
          None,
      ),
      r"(?:language_model\.)?model\.rotary_emb\..*": (
          "unused.rotary_emb", None
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.rotary_emb\..*": (
          r"unused.layers.\1.attn.rotary_emb",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.self_attn\.qkv_proj\.weight": (
          r"unused.layers.\1.attn.qkv_proj.weight",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.pre_feedforward_layernorm\.weight": (
          r"layers.\1.pre_ffw_norm.scale",
          None,
      ),
      r"(?:language_model\.)?model\.layers\.([0-9]+)\.(pre_ffn_layernorm|pre_ffw_layernorm)\.weight": (
          r"layers.\1.pre_ffw_norm.scale",
          None,
      ),
  }

  # Vision Tower (SigLIP).
  if cfg.vision_config is not None:
    mapping.update({
        r"vision_tower\.vision_model\.embeddings\.patch_embedding\.weight": (
            "vision_encoder.siglip_encoder.embedding.kernel",
            ((2, 3, 1, 0), None),
        ),
        r"vision_tower\.vision_model\.embeddings\.patch_embedding\.bias": (
            "vision_encoder.siglip_encoder.embedding.bias",
            None,
        ),
        r"vision_tower\.vision_model\.embeddings\.position_embedding\.weight": (
            "vision_encoder.siglip_encoder.pos_embedding",
            None,
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.attn.query_proj.kernel",
            ((1, 0), None),
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.attn.key_proj.kernel",
            ((1, 0), None),
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.attn.value_proj.kernel",
            ((1, 0), None),
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.weight": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.attn.out_proj.kernel",
            ((1, 0), None),
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.attn.query_proj.bias",
            None,
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.attn.key_proj.bias",
            None,
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.attn.value_proj.bias",
            None,
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.bias": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.attn.out_proj.bias",
            None,
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm1\.weight": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.ln1.scale",
            None,
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm1\.bias": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.ln1.bias",
            None,
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm2\.weight": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.ln2.scale",
            None,
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm2\.bias": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.ln2.bias",
            None,
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc1\.weight": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.mlp.fc1.kernel",
            ((1, 0), None),
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc2\.weight": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.mlp.fc2.kernel",
            ((1, 0), None),
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc1\.bias": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.mlp.fc1.bias",
            None,
        ),
        r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc2\.bias": (
            r"vision_encoder.siglip_encoder.transformer.blocks.\1.mlp.fc2.bias",
            None,
        ),
        r"vision_tower\.vision_model\.post_layernorm\.weight": (
            "vision_encoder.siglip_encoder.transformer.encoder_norm.scale",
            None,
        ),
        r"vision_tower\.vision_model\.post_layernorm\.bias": (
            "vision_encoder.siglip_encoder.transformer.encoder_norm.bias",
            None,
        ),
        # Multi-modal Projector
        r"multi_modal_projector\.mm_input_projection_weight": (
            "embedder.mm_input_projection.w",
            None,
        ),
        r"multi_modal_projector\.mm_soft_emb_norm\.weight": (
            "embedder.mm_soft_embedding_norm.scale",
            None,
        ),
    })
  return mapping


def _make_preprocess_fn(cfg: model_lib.ModelConfig):
  # [FUNCTION DOC]
  # HELPER: builds the stateful preprocess closure that fuses the temporary
  # q/k/v tensors into qkv_einsum.w ([3, N, D, H], when num_heads ==
  # num_kv_heads) or q_einsum.w ([N, D, H]) plus kv_einsum.w ([2, K, D, H]).
  # Raises ValueError if cfg.head_dim is odd. The pending dict persists across
  # calls so a layer's q/k/v may arrive in different calls.
  """Creates a tensor preprocessing function for Gemma3 safetensors, fusing q, k, and v projections."""
  q_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.q$")
  k_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.k$")
  v_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.v$")

  fused_qkv = cfg.num_heads == cfg.num_kv_heads
  pending: dict[str, dict[str, jnp.ndarray]] = {}

  if cfg.head_dim % 2 != 0:
    raise ValueError(
        f"Gemma3 head_dim must be even for RoPE, got {cfg.head_dim}"
    )

  def _to_ndh(q: jnp.ndarray) -> jnp.ndarray:
    # [FUNCTION DOC]
    # HELPER: returns q in [N, D, H] layout, accepting [N, D, H], [D, N, H] or
    # [N, H, D]; raises ValueError otherwise.
    # Expected shape: (N, D, H)
    if q.shape == (cfg.num_heads, cfg.embed_dim, cfg.head_dim):
      return q
    if q.shape == (cfg.embed_dim, cfg.num_heads, cfg.head_dim):  # D,N,H
      return jnp.transpose(q, (1, 0, 2))  # -> N,D,H
    if q.shape == (cfg.num_heads, cfg.head_dim, cfg.embed_dim):  # N,H,D
      return jnp.transpose(q, (0, 2, 1))  # -> N,D,H
    raise ValueError(f"[gemma3 preprocess] unexpected q shape: {q.shape}")

  def _to_kdh(x: jnp.ndarray) -> jnp.ndarray:
    # [FUNCTION DOC]
    # HELPER: returns a k or v weight in [K, D, H] layout, accepting [K, D, H],
    # [D, K, H] or [K, H, D]; raises ValueError otherwise.
    if x.shape == (cfg.num_kv_heads, cfg.embed_dim, cfg.head_dim):
      return x
    if x.shape == (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim):  # D,K,H
      return jnp.transpose(x, (1, 0, 2))  # -> K,D,H
    if x.shape == (cfg.num_kv_heads, cfg.head_dim, cfg.embed_dim):  # K,H,D
      return jnp.transpose(x, (0, 2, 1))  # -> K,D,H
    raise ValueError(f"[gemma3 preprocess] unexpected kv shape: {x.shape}")

  def preprocess(tensors: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    # [FUNCTION DOC]
    # Copies the input dict, moves temporary q/k/v entries into pending, then
    # for each pending layer writes the fused weights as soon as the needed
    # slots are present (fused: all three, validated to [N, D, H] each, else
    # ValueError; non-fused: q alone, and k with v together) and removes
    # completed layers from pending. Returns the new dict. Side effect: mutates
    # the closure pending dict.
    out = dict(tensors)

    for key in list(out):
      m = q_pat.fullmatch(key) or k_pat.fullmatch(key) or v_pat.fullmatch(key)
      if not m:
        continue
      layer_id = m.group(1)
      arr = out.pop(key)
      slot = "q" if key.endswith(".q") else ("k" if key.endswith(".k") else "v")
      pending.setdefault(layer_id, {})[slot] = arr

    for layer_id, slots in list(pending.items()):
      q = slots.get("q")
      k = slots.get("k")
      v = slots.get("v")

      if fused_qkv:
        if (q is not None) and (k is not None) and (v is not None):
          q = _to_ndh(q)
          k = _to_kdh(k)
          v = _to_kdh(v)
          exp = (cfg.num_heads, cfg.embed_dim, cfg.head_dim)
          if not (q.shape == k.shape == v.shape == exp):
            raise ValueError(
                f"[gemma3 preprocess] layer {layer_id}: fused q/k/v shape"
                f" mismatch: q={getattr(q, 'shape', None)},"
                f" k={getattr(k, 'shape', None)},"
                f" v={getattr(v, 'shape', None)}, expected={exp}"
            )
          out[f"layers.{layer_id}.attn.qkv_einsum.w"] = jnp.stack(
              [q, k, v], axis=0
          )  # (3,N,D,H)
          del pending[layer_id]
      else:
        wrote = False
        if q is not None:
          q = _to_ndh(q)
          out[f"layers.{layer_id}.attn.q_einsum.w"] = q
          slots.pop("q", None)
          wrote = True

        if (k is not None) and (v is not None):
          k = _to_kdh(k)
          v = _to_kdh(v)
          out[f"layers.{layer_id}.attn.kv_einsum.w"] = jnp.stack(
              [k, v], axis=0
          )  # (2,K,D,H)
          slots.pop("k", None)
          slots.pop("v", None)
          wrote = True

        if wrote and not slots:
          del pending[layer_id]

    return out

  return preprocess


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
):
  # [FUNCTION DOC]
  # Public entry point: loads safetensors from file_dir into a model.Gemma3
  # built from config, optionally sharded on mesh and cast to dtype, using the
  # key mapping above and a fresh preprocess function; mode is forwarded to
  # safetensors_loader.load_and_create_model. Returns the populated model.
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Gemma3,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=_make_preprocess_fn(config),
      dtype=dtype,
      mode=mode,
  )
