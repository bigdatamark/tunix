# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

# [MODULE DOC]
# Loads Gemma (v1 / Gemma 2) weights from Hugging Face-style safetensors files
# into the NNX model.Gemma via the generic safetensors_loader.
# Provides: the HF-key-regex to NNX-key mapping with transpose/reshape rules,
# a stateful preprocess function that fuses separate q/k/v projections into
# the qkv_einsum / q_einsum + kv_einsum weights, a vocab-size probe, and the
# public create_model_from_safe_tensors entry point.
"""Loads Gemma2 parameters from safetensors files."""

import dataclasses
import os
import re

import jax
import jax.numpy as jnp
import safetensors
from tunix.models import safetensors_loader
from tunix.models.gemma import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  # [FUNCTION DOC]
  # HELPER: returns the key mapping consumed by safetensors_loader. Keys are
  # regexes over HF tensor names; values are (target NNX dotted path, which may
  # contain backreference \1 for the layer index, and a transform that is either
  # None or a (permute axes, reshape shape or None) tuple applied in that order).
  # Mappings: embed_tokens -> embedder.input_embedding; q/k/v_proj ->
  # temporary keys 'tmp.layers.N.attn.q|k|v', transposed then reshaped to
  # [D, heads, head_dim] (later fused by preprocess); o_proj ->
  # layers.N.attn.attn_vec_einsum.w transposed and reshaped to [N, H, D];
  # gate/up/down_proj -> mlp.*.kernel transposed; layer norms -> the matching
  # RMSNorm scale (input_layernorm -> pre_attention_norm, post_attention_layernorm
  # -> post_attn_norm, pre/post_feedforward_layernorm -> pre/post_ffw_norm);
  # model.norm -> final_norm.scale; lm_head weight/bias, rotary buffers and any
  # bias are routed to 'unused.*' keys (ignored).
  """Mapping of torch_keys to (nnx_keys, (permute_rule, reshape_rule))."""
  mapping = {
      r"model\.embed_tokens\.weight": ("embedder.input_embedding", None),
      r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"tmp.layers.\1.attn.q",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"tmp.layers.\1.attn.k",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"tmp.layers.\1.attn.v",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
          r"layers.\1.attn.attn_vec_einsum.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
          r"layers.\1.mlp.gate_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
          r"layers.\1.mlp.up_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
          r"layers.\1.mlp.down_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\1.pre_attention_norm.scale",
          None,
      ),
      r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attn_norm.scale",
          None,
      ),
      r"model\.layers\.([0-9]+)\.pre_feedforward_layernorm\.weight": (
          r"layers.\1.pre_ffw_norm.scale",
          None,
      ),
      r"model\.layers\.([0-9]+)\.post_feedforward_layernorm\.weight": (
          r"layers.\1.post_ffw_norm.scale",
          None,
      ),
      r"model\.norm\.weight": ("final_norm.scale", None),
      r"lm_head\.weight": ("unused.lm_head.weight", None),
      r"lm_head\.bias": ("unused.lm_head.bias", None),
      r"model\.layers\.([0-9]+)\.self_attn\.rotary_emb\..*": (
          r"unused.rotary.\1",
          None,
      ),
      r".*\.bias": (r"unused.bias", None),
  }
  return mapping


def _make_preprocess_fn(cfg: model_lib.ModelConfig):
  # [FUNCTION DOC]
  # HELPER: builds the preprocess closure used by the loader to assemble
  # attention weights.
  # Input: cfg (ModelConfig). Raises ValueError if cfg.head_dim is odd.
  # Closure state: compiled regexes for the temporary q/k/v keys, fused_qkv flag
  # (num_heads == num_kv_heads), and a pending dict layer_id -> {slot: array}
  # that persists across preprocess calls (tensors of one layer may come from
  # different files/batches).
  # Output: the preprocess function.
  """Creates a preprocess function to reshape and stack Q, K, and V tensors for Gemma safetensors."""
  q_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.q$")
  k_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.k$")
  v_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.v$")
  fused_qkv = cfg.num_heads == cfg.num_kv_heads
  pending: dict[str, dict[str, jnp.ndarray]] = {}
  if cfg.head_dim % 2 != 0:
    raise ValueError(
        f"Gemma2 head_dim must be even for RoPE, got {cfg.head_dim}"
    )

  def _to_ndh(q: jnp.ndarray) -> jnp.ndarray:
    # [FUNCTION DOC]
    # HELPER: normalizes a query weight to [num_heads, embed_dim, head_dim];
    # accepts that layout, [D, N, H] (transposed to N, D, H) or [N, H, D]
    # (transposed to N, D, H). Raises ValueError for any other shape.
    if q.shape == (cfg.num_heads, cfg.embed_dim, cfg.head_dim):
      return q
    if q.shape == (cfg.embed_dim, cfg.num_heads, cfg.head_dim):
      return jnp.transpose(q, (1, 0, 2))
    if q.shape == (cfg.num_heads, cfg.head_dim, cfg.embed_dim):
      return jnp.transpose(q, (0, 2, 1))
    raise ValueError(f"[gemma2 preprocess] unexpected q shape: {q.shape}")

  def _to_kdh(x: jnp.ndarray) -> jnp.ndarray:
    # [FUNCTION DOC]
    # HELPER: same as _to_ndh but for key/value weights with num_kv_heads, producing
    # [K, D, H]. Raises ValueError for unexpected shapes.
    if x.shape == (cfg.num_kv_heads, cfg.embed_dim, cfg.head_dim):
      return x
    if x.shape == (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim):
      return jnp.transpose(x, (1, 0, 2))
    if x.shape == (cfg.num_kv_heads, cfg.head_dim, cfg.embed_dim):
      return jnp.transpose(x, (0, 2, 1))
    raise ValueError(f"[gemma2 preprocess] unexpected kv shape: {x.shape}")

  def preprocess(tensors: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    # [FUNCTION DOC]
    # Takes a dict of tensors (NNX dotted keys) and returns a new dict where the
    # temporary q/k/v entries are removed and stashed into pending. Then for each
    # pending layer: in fused mode, once q, k and v are all present, normalize
    # them, verify all equal [N, D, H] (ValueError otherwise) and write
    # 'layers.N.attn.qkv_einsum.w' = stack([q, k, v]) and drop the layer from
    # pending. In non-fused mode, write q as 'layers.N.attn.q_einsum.w' as soon as
    # available, and 'layers.N.attn.kv_einsum.w' = stack([k, v]) once both k and v
    # are present; the layer is removed from pending when all its slots were
    # consumed. Side effect: mutates the closure pending dict.
    # KNOWN ISSUE: error messages say "gemma2" although the function is used for
    # all Gemma versions; layers still incomplete at the end of loading are
    # silently left in pending.
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
                f"[gemma2 preprocess] layer {layer_id}: fused q/k/v shape"
                f" mismatch: q={getattr(q, 'shape', None)},"
                f" k={getattr(k, 'shape', None)},"
                f" v={getattr(v, 'shape', None)}, expected={exp}"
            )
          out[f"layers.{layer_id}.attn.qkv_einsum.w"] = jnp.stack(
              [q, k, v], axis=0
          )
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
          out[f"layers.{layer_id}.attn.kv_einsum.w"] = jnp.stack([k, v], axis=0)
          slots.pop("k", None)
          slots.pop("v", None)
          wrote = True
        if wrote and not slots:
          del pending[layer_id]
    return out
  return preprocess


def _peek_vocab_size_from_safetensors(file_dir: str) -> int:
  # [FUNCTION DOC]
  # HELPER: scans file_dir (non-recursive) for '.safetensors' files and returns
  # the first dimension (vocab size) of 'model.embed_tokens.weight' from the
  # first file that contains it (loads that tensor fully). Raises
  # FileNotFoundError if no file has that tensor (the message says no
  # .safetensors were found even when files exist but lack the key).
  for fn in os.listdir(file_dir):
    if fn.endswith(".safetensors"):
      path = os.path.join(file_dir, fn)
      with safetensors.safe_open(path, framework="jax") as f:
        if "model.embed_tokens.weight" in f.keys():
          shape = f.get_tensor("model.embed_tokens.weight").shape
          return shape[0]
  raise FileNotFoundError("No .safetensors found to peek vocab size")


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
) -> model_lib.Gemma:
  # [FUNCTION DOC]
  # Public entry point. Inputs: file_dir (local directory of safetensors),
  # config (ModelConfig), optional mesh for sharding, optional dtype to cast,
  # mode passed through to the loader (default 'auto').
  # Behavior: probes the checkpoint vocab size; if it differs from
  # config.num_embed, replaces num_embed in a copied config and prints a notice
  # (side effect: stdout). Then calls safetensors_loader.load_and_create_model
  # with model class model.Gemma, the key mapping function and a fresh
  # preprocess function.
  # Output: a populated model.Gemma. Exceptions propagate from the probe and
  # loader.
  # KNOWN ISSUE: the probe uses os.listdir, so a gs:// path (which the loader
  # itself supports) would fail here.
  v_ckpt = _peek_vocab_size_from_safetensors(file_dir)
  if v_ckpt != config.num_embed:
    config = dataclasses.replace(config, num_embed=v_ckpt)
    print(f"[gemma2] override num_embed -> {v_ckpt} from checkpoint")
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Gemma,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=_make_preprocess_fn(config),
      dtype=dtype,
      mode=mode,
  )
