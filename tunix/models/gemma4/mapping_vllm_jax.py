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
# Weight-mapping specification used to transfer Gemma4 NNX weights into the
# vLLM JAX backend (Hugging Face naming).
# - Sharding / MappingEntry type aliases: a sharding is a tuple of mesh axis
#   names or None per dimension; an entry is (HF key, sharding).
# - TO_HF_MAPPINGS: NNX dotted parameter path pattern ('*' = layer index)
#   -> (HF parameter name pattern, target sharding on the 'model' axis).
#   Covers embeddings, attention q/k/v/o einsums and QK norms, all RMSNorms,
#   dense MLP, layer skip scalar, MoE router/expert weights and norms, and the
#   per-layer-input embedding/projection/gate weights.
# - LORA_TO_HF_MAPPINGS: empty (no LoRA mapping support).
# - TO_HF_TRANSPOSE_KEYS: q/k/v einsum weights are transposed with axes
#   (1, 0, 2), i.e. [N, D, H] -> [D, N, H].
# - preprocess_src_state: splits fused kv_einsum weights into separate
#   k_einsum / v_einsum entries before mapping.
# - VLLM_JAX_MAPPING: dict bundling the above under keys 'to_hf_mappings',
#   'lora_to_hf_mappings', 'to_hf_transpose_keys', 'preprocess_src_state'.
"""vLLM JAX backend mappings for Gemma4 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple
from flax import nnx

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


TO_HF_MAPPINGS = {
    'embedder.input_embedding': ('model.embed_tokens.weight', ('model', None)),
    'layers.*.pre_attention_norm.scale': (
        'model.layers.*.input_layernorm.weight',
        (None,),
    ),
    'layers.*.attn.q_einsum.w': (
        'model.layers.*.self_attn.q_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn._query_norm.scale': (
        'model.layers.*.self_attn.q_norm.weight',
        (None,),
    ),
    'layers.*.attn.k_einsum.w': (
        'model.layers.*.self_attn.k_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn.v_einsum.w': (
        'model.layers.*.self_attn.v_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn._key_norm.scale': (
        'model.layers.*.self_attn.k_norm.weight',
        (None,),
    ),
    'layers.*.attn.attn_vec_einsum.w': (
        'model.layers.*.self_attn.o_proj.weight',
        ('model', None, None),
    ),
    'layers.*.post_attention_norm.scale': (
        'model.layers.*.post_attention_layernorm.weight',
        (None,),
    ),
    'layers.*.pre_ffw_norm.scale': (
        'model.layers.*.pre_feedforward_layernorm.weight',
        (None,),
    ),
    'layers.*.mlp.gate_proj.kernel': (
        'model.layers.*.mlp.gate_proj.weight',
        (None, 'model'),
    ),
    'layers.*.mlp.up_proj.kernel': (
        'model.layers.*.mlp.up_proj.weight',
        (None, 'model'),
    ),
    'layers.*.mlp.down_proj.kernel': (
        'model.layers.*.mlp.down_proj.weight',
        ('model', None),
    ),
    'layers.*.post_ffw_norm.scale': (
        'model.layers.*.post_feedforward_layernorm.weight',
        (None,),
    ),
    'layers.*.skip_scale': (
        'model.layers.*.layer_scalar',
        (None,),
    ),
    'final_norm.scale': ('model.norm.weight', (None,)),
    'layers.*.moe_pre_ffw_norm.scale': (
        'model.layers.*.pre_feedforward_layernorm_2.weight',
        (None,),
    ),
    'layers.*.moe.router_logits': (
        'model.layers.*.router.proj.weight',
        (None, 'model'),
    ),
    'layers.*.moe.router_scale': (
        'model.layers.*.router.scale',
        (None,),
    ),
    'layers.*.moe.per_expert_scale': (
        'model.layers.*.router.per_expert_scale',
        (None,),
    ),
    'layers.*.moe.gating_einsum': (
        'model.layers.*.experts.kernel_gating_upproj_EDF',
        (None, None, 'model'),
    ),
    'layers.*.moe.linear': (
        'model.layers.*.experts.kernel_down_proj_EFD',
        ('model', None, None),
    ),
    'layers.*.dense_post_ffw_norm.scale': (
        'model.layers.*.post_feedforward_layernorm_1.weight',
        (None,),
    ),
    'layers.*.moe_post_ffw_norm.scale': (
        'model.layers.*.post_feedforward_layernorm_2.weight',
        (None,),
    ),
    'embedder.per_layer_input_embedding': (
        'model.embed_tokens_per_layer.weight',
        ('model', None, None),
    ),
    'embedder.per_layer_model_projection.w': (
        'model.per_layer_model_projection.weight',
        (None, None, 'model'),
    ),
    'embedder.per_layer_projection_norm.scale': (
        'model.per_layer_projection_norm.weight',
        (None,),
    ),
    'layers.*.per_layer_input_gate.w': (
        'model.layers.*.per_layer_input_gate.weight',
        (None, 'model'),
    ),
    'layers.*.per_layer_projection.w': (
        'model.layers.*.per_layer_projection.weight',
        ('model', None),
    ),
    'layers.*.post_per_layer_input_norm.scale': (
        'model.layers.*.post_per_layer_input_norm.weight',
        (None,),
    ),
}

LORA_TO_HF_MAPPINGS: Dict[str, MappingEntry] = {}

TO_HF_TRANSPOSE_KEYS = {
    'layers.*.attn.q_einsum.w': (1, 0, 2),
    'layers.*.attn.k_einsum.w': (1, 0, 2),
    'layers.*.attn.v_einsum.w': (1, 0, 2),
}

def preprocess_src_state(src_state: Any) -> Any:
  # [FUNCTION DOC]
  # Splits fused KV weights in an NNX State.
  # Input: src_state (anything; only objects with a flat_state method are
  # processed, others are returned unchanged).
  # Behavior: iterates over (key path, param) pairs of the flat state; for each
  # path whose dotted form contains 'attn.kv_einsum.w', takes the value
  # (param.value if present) [2, K, D, H] and emits two entries with the last
  # two path elements replaced by ('k_einsum', 'w') with val[0] and
  # ('v_einsum', 'w') with val[1]; wrapped in a new nnx.Param if the original
  # had a value attribute, raw arrays otherwise. Other entries are kept. Builds
  # a new state via src_state.from_flat_path.
  # Output: the new (or unchanged) state. No mutation of the input.
  # KNOWN ISSUE: the split weights are always wrapped as nnx.Param, discarding
  # the original variable type and its metadata (e.g. sharding annotations);
  # a fused qkv_einsum (MHA layout) is not split and has no mapping entry.
  if hasattr(src_state, 'flat_state'):
    flat_state = list(src_state.flat_state())
    new_flat_state = []
    for keys, param in flat_state:
      src_key = '.'.join(str(k) for k in keys)
      if 'attn.kv_einsum.w' in src_key:
        val = param.value if hasattr(param, 'value') else param
        k_val = val[0]
        v_val = val[1]
        k_keys = keys[:-2] + ('k_einsum', 'w')
        v_keys = keys[:-2] + ('v_einsum', 'w')
        if hasattr(param, 'value'):
          new_flat_state.append((k_keys, nnx.Param(k_val)))
          new_flat_state.append((v_keys, nnx.Param(v_val)))
        else:
          new_flat_state.append((k_keys, k_val))
          new_flat_state.append((v_keys, v_val))
      else:
        new_flat_state.append((keys, param))
    src_state = src_state.from_flat_path(new_flat_state)
  return src_state


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': LORA_TO_HF_MAPPINGS,
    'to_hf_transpose_keys': TO_HF_TRANSPOSE_KEYS,
    'preprocess_src_state': preprocess_src_state,
}

__all__ = [
    'VLLM_JAX_MAPPING',
]
