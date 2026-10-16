# [MODULE DOC]
# Static weight mappings from tunix Llama3 parameters to the vLLM JAX (tpu
# inference) backend. Exposes VLLM_JAX_MAPPING with keys 'to_hf_mappings',
# 'lora_to_hf_mappings', 'to_hf_transpose_keys', 'to_hf_hook_fns', built by
# the helpers below. Sharding specs use the mesh axis name 'model'.
# Mapping entries are (target name pattern, sharding spec), where "*" in a
# tunix path stands for the layer index and is substituted into the target, and
# the sharding spec is a tuple of mesh axis names (or None for replicated) per
# tensor dimension. The consumer (tunix weight-sync utilities) uses these to
# rename, reshard and optionally transpose tunix parameters when pushing them
# into the inference engine. No runtime behavior beyond building these dicts at
# import time; nothing raises.
# KNOWN ISSUE: the 'os' import is unused.
"""Mappings for converting Llama3 weights to the vLLM JAX backend."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


def _to_hf_mappings() -> Dict[str, MappingEntry]:
  # [FUNCTION DOC]
  # HELPER: Returns base mapping to vLLM-JAX's flax naming: lm_head.w ->
  # model.lm_head (sharded dim 1); embedder.input_embedding -> model.embed.embedding
  # (sharded dim 0); layernorm .w -> .scale (replicated); MLP kernels keep
  # 'kernel' names (down sharded on input dim, gate/up on output dim); attention
  # q/k/v .w -> self_attn.*.kernel sharded on heads dim 1, o .w sharded on dim 0;
  # final_norm.w -> model.norm.scale.
  """The parameter key mapping between Tunix vanilla model and vLLM Jax backend"""
  return {
      'lm_head.w': ('model.lm_head', (None, 'model')),
      'embedder.input_embedding': (
          'model.embed.embedding',
          ('model', None),
      ),
      'layers.*.input_layernorm.w': (
          'model.layers.*.input_layernorm.scale',
          (None,),
      ),
      'layers.*.mlp.down_proj.kernel': (
          'model.layers.*.mlp.down_proj.kernel',
          ('model', None),
      ),
      'layers.*.mlp.gate_proj.kernel': (
          'model.layers.*.mlp.gate_proj.kernel',
          (None, 'model'),
      ),
      'layers.*.mlp.up_proj.kernel': (
          'model.layers.*.mlp.up_proj.kernel',
          (None, 'model'),
      ),
      'layers.*.post_attention_layernorm.w': (
          'model.layers.*.post_attention_layernorm.scale',
          (None,),
      ),
      'layers.*.attn.k_proj.w': (
          'model.layers.*.self_attn.k_proj.kernel',
          (None, 'model', None),
      ),
      'layers.*.attn.o_proj.w': (
          'model.layers.*.self_attn.o_proj.kernel',
          ('model', None, None),
      ),
      'layers.*.attn.q_proj.w': (
          'model.layers.*.self_attn.q_proj.kernel',
          (None, 'model', None),
      ),
      'layers.*.attn.v_proj.w': (
          'model.layers.*.self_attn.v_proj.kernel',
          (None, 'model', None),
      ),
      'final_norm.w': ('model.norm.scale', (None,)),
  }


def _lora_to_hf_mappings() -> Dict[str, MappingEntry] | None:
  # [FUNCTION DOC]
  # HELPER: Returns LoRA mapping: MLP and attention '<kernel|w>_lora_a/_lora_b'
  # map to vLLM '*.kernel_lora_a/_lora_b'. gate/up lora_b sharded on dim 1,
  # down lora_a and all attention lora_a sharded on dim 0, others replicated.
  # KNOWN ISSUE: attention lora_a tensors are [embed, rank] for q/k/v and
  # [heads, head_dim, rank] for o, so sharding dim 0 ('model') shards the input
  # dim, and the 2-element spec for o lora_a mismatches its 3D rank.
  """The lora parameter key mapping between Tunix vanilla model and vLLM Jax backend"""
  return {
      'layers.*.mlp.gate_proj.kernel_lora_a': (
          'model.layers.*.mlp.gate_proj.kernel_lora_a',
          (None, None),
      ),
      'layers.*.mlp.gate_proj.kernel_lora_b': (
          'model.layers.*.mlp.gate_proj.kernel_lora_b',
          (None, 'model'),
      ),
      'layers.*.mlp.up_proj.kernel_lora_a': (
          'model.layers.*.mlp.up_proj.kernel_lora_a',
          (None, None),
      ),
      'layers.*.mlp.up_proj.kernel_lora_b': (
          'model.layers.*.mlp.up_proj.kernel_lora_b',
          (None, 'model'),
      ),
      'layers.*.mlp.down_proj.kernel_lora_a': (
          'model.layers.*.mlp.down_proj.kernel_lora_a',
          ('model', None),
      ),
      'layers.*.mlp.down_proj.kernel_lora_b': (
          'model.layers.*.mlp.down_proj.kernel_lora_b',
          (None, None),
      ),
      'layers.*.attn.q_proj.w_lora_a': (
          'model.layers.*.self_attn.q_proj.kernel_lora_a',
          ('model', None),
      ),
      'layers.*.attn.q_proj.w_lora_b': (
          'model.layers.*.self_attn.q_proj.kernel_lora_b',
          (None, None),
      ),
      'layers.*.attn.k_proj.w_lora_a': (
          'model.layers.*.self_attn.k_proj.kernel_lora_a',
          ('model', None),
      ),
      'layers.*.attn.k_proj.w_lora_b': (
          'model.layers.*.self_attn.k_proj.kernel_lora_b',
          (None, None),
      ),
      'layers.*.attn.v_proj.w_lora_a': (
          'model.layers.*.self_attn.v_proj.kernel_lora_a',
          ('model', None),
      ),
      'layers.*.attn.v_proj.w_lora_b': (
          'model.layers.*.self_attn.v_proj.kernel_lora_b',
          (None, None),
      ),
      'layers.*.attn.o_proj.w_lora_a': (
          'model.layers.*.self_attn.o_proj.kernel_lora_a',
          ('model', None),
      ),
      'layers.*.attn.o_proj.w_lora_b': (
          'model.layers.*.self_attn.o_proj.kernel_lora_b',
          (None, None),
      ),
  }


def _to_hf_transpose_keys() -> Dict[str, Tuple[int, int]] | None:
  # [FUNCTION DOC]
  # HELPER: Returns transpose rules: any key containing 'embedding' is
  # transposed (1, 0).
  # KNOWN ISSUE: the type hint says Tuple[int, int] values while generic rules
  # may be longer; and transposing the embedding seems at odds with the mapping
  # of input_embedding to an embedding table of the same layout (verify against
  # the consumer's matching semantics).
  """The parameter key transposition mapping between Tunix vanilla model and vLLM Jax backend"""
  return {
      'embedding': (1, 0),
  }


def _to_hf_hook_fns() -> Dict[str, Any] | None:
  # [FUNCTION DOC]
  # HELPER: Returns None: no hook functions.
  """Additional parameter manipulation hook between Tunix vanilla model and vLLM Jax backend"""
  return None


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': _to_hf_mappings(),
    'lora_to_hf_mappings': _lora_to_hf_mappings(),
    'to_hf_transpose_keys': _to_hf_transpose_keys(),
    'to_hf_hook_fns': _to_hf_hook_fns(),
}

__all__ = [
    'VLLM_JAX_MAPPING',
]
