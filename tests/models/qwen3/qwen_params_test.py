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
# Tests for Qwen3 parameter handling: LoRA merged-model saving/loading (via the shared
# lora_params_test_base.LoraParamsTestBase, which this module specializes with Qwen3 hooks) and
# tied-embedding configuration behavior. Uses a tiny 2-layer config (vocab 256, embed 64, hidden 128,
# 2 heads of dim 32, 2 kv heads, dense MLP). Side effects: writes safetensors and config.json into the
# base class's checkpoint directory. When run directly in Colab it runs only Qwen3ParamsTest via the
# unittest runner; otherwise absltest.main.

"""Tests for Qwen3 model parameters and LoRA merged model saving."""

import os
import unittest

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as safe_np
from flax import nnx
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import params as qwen3_params
from tunix.tests import lora_params_test_base
from tunix.tests import test_common


class Qwen3ParamsTest(lora_params_test_base.LoraParamsTestBase):
  # [CLASS DOC]
  # Qwen3 specialization of the shared LoRA params test base; supplies config, model class, LoRA target
  # regex, HF projection key names, save/load functions and a checkpoint writer.
  """Tests for Qwen3 model parameters and LoRA merging."""

  def create_config(self):
    # [FUNCTION DOC]
    # HELPER: returns a tiny Qwen3 ModelConfig (2 layers, vocab 256, embed 64, hidden 128, 2 heads, head
    # dim 32, 2 kv heads, rope theta 10000, norm eps 1e-6, no experts).
    """Create Qwen3 model config for testing."""
    return qwen3_model.ModelConfig(
        num_layers=2,
        vocab_size=256,
        embed_dim=64,
        hidden_dim=128,
        num_heads=2,
        head_dim=32,
        num_kv_heads=2,
        rope_theta=10000,
        norm_eps=1e-6,
        num_experts=None,
        num_experts_per_tok=None,
    )

  def get_model_class(self):
    # [FUNCTION DOC]
    # HELPER: returns the Qwen3 model class.
    """Get Qwen3 model class."""
    return qwen3_model.Qwen3

  def get_lora_module_path(self) -> str:
    # [FUNCTION DOC]
    # HELPER: returns a regex alternation matching q/k/v/o attention projections and gate/up/down MLP
    # projections as LoRA targets.
    """Get LoRA target modules for Qwen3."""
    return (
        ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj"
    )

  def get_projection_keys(self, layer_idx: int) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: returns the seven HF safetensors weight names for the given layer index:
    # model.layers.<i>.self_attn.{q,k,v,o}_proj.weight and model.layers.<i>.mlp.{gate,up,down}_proj.weight.
    """Get projection keys for Qwen3."""
    prefix = f"model.layers.{layer_idx}"
    return [
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.mlp.gate_proj.weight",
        f"{prefix}.mlp.up_proj.weight",
        f"{prefix}.mlp.down_proj.weight",
    ]

  def save_merged_model(self, lora_model):
    # [FUNCTION DOC]
    # HELPER: calls qwen3 params save_lora_merged_model_as_safetensors with the base checkpoint dir, merged
    # output dir, the LoRA model, and the base class's rank and alpha. Side effect: writes files.
    """Save Qwen3 LoRA merged model."""
    qwen3_params.save_lora_merged_model_as_safetensors(
        local_model_path=self.base_checkpoint_dir,
        output_dir=self.merged_output_dir,
        lora_model=lora_model,
        rank=self.rank,
        alpha=self.alpha,
    )

  def create_model_from_checkpoint(self, checkpoint_dir: str):
    # [FUNCTION DOC]
    # HELPER: loads a Qwen3 model from a safetensors directory with self.config, no mesh, float32.
    """Load Qwen3 model from checkpoint."""
    return qwen3_params.create_model_from_safe_tensors(
        file_dir=checkpoint_dir,
        config=self.config,
        mesh=None,
        dtype=jnp.float32,
    )

  def create_checkpoint(self, model) -> str:
    # [FUNCTION DOC]
    # HELPER: writes the model's weights as an HF-format safetensors checkpoint. Creates the base checkpoint
    # dir, saves embed_tokens, final norm and transposed lm_head; for each layer saves input/post-attention
    # layernorms and q/k norms; q/k/v projections stored as (heads, embed, head_dim) are permuted to (embed,
    # heads, head_dim), flattened to (embed, heads*head_dim) and transposed; o_proj (heads, head_dim, embed)
    # is permuted to (embed, heads, head_dim), flattened and transposed; MLP kernels are transposed. Each
    # projection is written only if present. Writes model.safetensors and a minimal config.json with
    # model_type "qwen3". Returns the checkpoint directory path.
    # KNOWN ISSUE: lm_head is read unconditionally, so this fails if the config used tied embeddings (no
    # lm_head attribute).
    """Extract model weights and save in safetensors format.

    Uses the model's actual weights and applies inverse transformations
    (from _get_key_and_transform_mapping) to create a valid safetensors file.

    Args:
      model: Base model to extract weights from.

    Returns:
      Path to the created checkpoint directory.
    """
    os.makedirs(self.base_checkpoint_dir, exist_ok=True)

    base_state = {}
    base_state["model.embed_tokens.weight"] = np.array(
        model.embedder.input_embedding.value
    )
    base_state["model.norm.weight"] = np.array(model.final_norm.w.value)
    base_state["lm_head.weight"] = np.array(model.lm_head.w.value).T

    # Extract and transform weights for all layers
    # Based on inverse of _get_key_and_transform_mapping in qwen3/params.py
    for layer_idx, layer in enumerate(model.layers):
      prefix = f"model.layers.{layer_idx}"

      base_state[f"{prefix}.input_layernorm.weight"] = np.array(
          layer.input_layernorm.w.value
      )
      base_state[f"{prefix}.post_attention_layernorm.weight"] = np.array(
          layer.post_attention_layernorm.w.value
      )

      base_state[f"{prefix}.self_attn.q_norm.weight"] = np.array(
          layer.attn.q_norm.w.value
      )
      base_state[f"{prefix}.self_attn.k_norm.weight"] = np.array(
          layer.attn.k_norm.w.value
      )

      # Attention projections
      # nnx: (num_heads, embed_dim, head_dim) → safetensors: (num_heads*head_dim, embed_dim)
      # Transform: transpose(1,0,2) then reshape then transpose
      if hasattr(layer.attn, "q_proj"):
        w = np.array(
            layer.attn.q_proj.w.value
        )  # (num_heads, embed_dim, head_dim)
        w = w.transpose(1, 0, 2)  # (embed_dim, num_heads, head_dim)
        w = w.reshape(
            self.config.embed_dim, -1
        )  # (embed_dim, num_heads*head_dim)
        base_state[f"{prefix}.self_attn.q_proj.weight"] = w.T

      if hasattr(layer.attn, "k_proj"):
        w = np.array(
            layer.attn.k_proj.w.value
        )  # (num_kv_heads, embed_dim, head_dim)
        w = w.transpose(1, 0, 2)  # (embed_dim, num_kv_heads, head_dim)
        w = w.reshape(
            self.config.embed_dim, -1
        )  # (embed_dim, num_kv_heads*head_dim)
        base_state[f"{prefix}.self_attn.k_proj.weight"] = w.T

      if hasattr(layer.attn, "v_proj"):
        w = np.array(
            layer.attn.v_proj.w.value
        )  # (num_kv_heads, embed_dim, head_dim)
        w = w.transpose(1, 0, 2)  # (embed_dim, num_kv_heads, head_dim)
        w = w.reshape(
            self.config.embed_dim, -1
        )  # (embed_dim, num_kv_heads*head_dim)
        base_state[f"{prefix}.self_attn.v_proj.weight"] = w.T

      if hasattr(layer.attn, "o_proj"):
        w = np.array(
            layer.attn.o_proj.w.value
        )  # (num_heads, head_dim, embed_dim)
        w = w.transpose(2, 0, 1)  # (embed_dim, num_heads, head_dim)
        w = w.reshape(
            self.config.embed_dim, -1
        )  # (embed_dim, num_heads*head_dim)
        base_state[f"{prefix}.self_attn.o_proj.weight"] = w.T

      # MLP projections
      # nnx: (in_features, out_features) → safetensors: (out_features, in_features)
      # Transform: just transpose
      if hasattr(layer.mlp, "gate_proj"):
        base_state[f"{prefix}.mlp.gate_proj.weight"] = np.array(
            layer.mlp.gate_proj.kernel.value
        ).T

      if hasattr(layer.mlp, "up_proj"):
        base_state[f"{prefix}.mlp.up_proj.weight"] = np.array(
            layer.mlp.up_proj.kernel.value
        ).T

      if hasattr(layer.mlp, "down_proj"):
        base_state[f"{prefix}.mlp.down_proj.weight"] = np.array(
            layer.mlp.down_proj.kernel.value
        ).T

    safe_np.save_file(
        base_state, os.path.join(self.base_checkpoint_dir, "model.safetensors")
    )

    # Minimal config for file copying test
    with open(os.path.join(self.base_checkpoint_dir, "config.json"), "w") as f:
      f.write('{"model_type": "qwen3"}')

    return self.base_checkpoint_dir



class Qwen3ModelConfigTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests Qwen3 tied-embedding configuration defaults and their effect on model structure.
  """Tests specific to Qwen3 configurations and architectural flags."""

  def test_tied_embeddings_config_values(self):
    # [FUNCTION DOC]
    # Verifies qwen3_0p6b and qwen3_1p7b configs enable use_tied_embedding and qwen3_8b disables it.
    """Verify that smaller Qwen3 models correctly default to tied embeddings."""
    # 0.5B / 0.6B model should use tied embeddings
    config_0p6b = qwen3_model.ModelConfig.qwen3_0p6b()
    self.assertTrue(
        config_0p6b.use_tied_embedding,
        "qwen3_0p6b config should have use_tied_embedding=True"
    )

    # 1.5B / 1.7B model should use tied embeddings
    config_1p7b = qwen3_model.ModelConfig.qwen3_1p7b()
    self.assertTrue(
        config_1p7b.use_tied_embedding,
        "qwen3_1p7b config should have use_tied_embedding=True"
    )

    # 8B model typically does not use tied embeddings
    config_8b = qwen3_model.ModelConfig.qwen3_8b()
    self.assertFalse(
        config_8b.use_tied_embedding,
        "qwen3_8b config should have use_tied_embedding=False"
    )

  def test_model_instantiation_with_tied_embeddings(self):
    # [FUNCTION DOC]
    # Verifies a Qwen3 built with use_tied_embedding True has no lm_head attribute.
    """Verify that the Qwen3 model omits the lm_head when embeddings are tied."""
    rngs = nnx.Rngs(params=0)

    # Create a config WITH tied embeddings
    tied_config = qwen3_model.ModelConfig(
        num_layers=2,
        vocab_size=256,
        embed_dim=64,
        hidden_dim=128,
        num_heads=2,
        head_dim=32,
        num_kv_heads=2,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=True,
    )
    tied_model = qwen3_model.Qwen3(tied_config, rngs=rngs)

    # The model should decode using the embedder, so lm_head shouldn't exist
    self.assertFalse(
        hasattr(tied_model, "lm_head"),
        "Model should not have a separate lm_head when use_tied_embedding is True"
    )

  def test_model_instantiation_without_tied_embeddings(self):
    # [FUNCTION DOC]
    # Verifies a Qwen3 built with use_tied_embedding False has an lm_head attribute.
    """Verify that the Qwen3 model includes the lm_head when embeddings are not tied."""
    rngs = nnx.Rngs(params=0)

    # Create a config WITHOUT tied embeddings
    untied_config = qwen3_model.ModelConfig(
        num_layers=2,
        vocab_size=256,
        embed_dim=64,
        hidden_dim=128,
        num_heads=2,
        head_dim=32,
        num_kv_heads=2,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=False,
    )
    untied_model = qwen3_model.Qwen3(untied_config, rngs=rngs)

    # The model should have a distinct lm_head layer
    self.assertTrue(
        hasattr(untied_model, "lm_head"),
        "Model must have a separate lm_head when use_tied_embedding is False"
    )


if __name__ == "__main__":
  # Check if running in Jupyter/IPython environment
  if test_common.is_running_in_colab():
    # Running in Jupyter/IPython - run tests directly to avoid SystemExit
    suite = unittest.TestLoader().loadTestsFromTestCase(Qwen3ParamsTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
  else:
    # Running as a script - use absltest.main()
    absltest.main()
