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
# Checkpoint loading for Llama3: defines the regex mapping from Hugging Face
# safetensors keys to tunix Llama3 parameter paths (with optional axis permutation
# and reshape), and a thin wrapper around safetensors_loader.load_and_create_model.
"""Utils for loading and converting Llama3 PT weights."""

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models.llama3 import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  # [FUNCTION DOC]
  # HELPER: Returns a dict mapping regex patterns of HF checkpoint keys to a tuple
  # (tunix target path with backreference to the layer index, transform). The
  # transform is None or (permutation, reshape-shape-or-None). Mapping: token
  # embedding -> embedder.input_embedding; q/k/v/o projection weights are
  # transposed (HF stores [out, in]) and reshaped to (embed, heads, head_dim) for
  # q, (embed, kv_heads, head_dim) for k/v and (heads, head_dim, embed) for o;
  # MLP gate/up/down weights are transposed into Linear kernels; final norm,
  # input and post-attention layernorm weights copied as-is; lm_head weight is
  # transposed into lm_head.w.
  # Also maps optional per-layer q_norm/k_norm weights, which standard Llama 3
  # checkpoints do not contain (unused patterns are harmless).
  # Input: cfg supplies embed_dim, num_heads, num_kv_heads, head_dim. No side effects.
  # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
  return {
      r"model\.embed_tokens\.weight": ("embedder.input_embedding", None),
      # attention projection weights
      r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"layers.\1.attn.q_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"layers.\1.attn.k_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"layers.\1.attn.v_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
          r"layers.\1.attn.o_proj.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      # mlp
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
      r"model\.norm\.weight": ("final_norm.w", None),
      # norms
      r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
          r"layers.\1.attn.q_norm.w",
          None,
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
          r"layers.\1.attn.k_norm.w",
          None,
      ),
      # layer norms (pre/post attention)
      r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\1.input_layernorm.w",
          None,
      ),
      r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attention_layernorm.w",
          None,
      ),
      r"lm_head\.weight": ("lm_head.w", ((1, 0), None)),
  }


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
) -> model_lib.Llama3:
  # [FUNCTION DOC]
  # Loads safetensors files from file_dir and returns a populated Llama3 model.
  # Delegates to safetensors_loader.load_and_create_model with model_class Llama3,
  # the given config, the key mapping above, optional mesh (sharding), dtype
  # override, no preprocessing, and mode (passed through; 'auto' by default).
  # Raises/side effects: those of the loader (file I/O, device allocation).
  """Load tensors from the safetensors file and create a Llama3 model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Llama3,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=None,
      dtype=dtype,
      mode=mode,
  )
