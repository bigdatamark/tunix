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
# HELPER: abstract base test case for verifying "save LoRA-merged model" functionality across
# model architectures. Subclasses provide the architecture-specific config, checkpoint writer,
# LoRA module regex, expected projection keys, the merge-and-save call and optionally a loader
# from a safetensors checkpoint. The base supplies fixtures (temp dirs, LoRA hyperparameters),
# deterministic dummy base-model creation, LoRA application with random LoRA weights, an
# expected-merge formula and three shared test methods. Module alias: create_dummy_model from the
# dummy model creator. No side effects at import.

"""Base test class for LoRA merged model saving and loading tests.

This module provides a common base class for testing LoRA merged model
functionality across different model architectures (Qwen3, Gemma3, Llama3,
etc.).
It handles shared test logic including:
- Test setup/teardown (temp directories, config)
- Base model creation
- LoRA application and weight initialization
- Expected merged weight computation
- Common test method templates

Architecture-specific implementations should:
1. Inherit from LoraParamsTestBase
2. Implement abstract methods for checkpoint creation and model loading
3. Optionally override methods for architecture-specific behavior
"""

import abc
import os
import shutil
import tempfile

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import qwix
import safetensors.numpy as safe_np
from tunix.models import dummy_model_creator

create_dummy_model = dummy_model_creator.create_dummy_model


class LoraParamsTestBase(absltest.TestCase, abc.ABC):
  # [CLASS DOC]
  # HELPER: abstract absltest TestCase (also an ABC). Fixture attributes: rank 4, alpha 16, random
  # seed 42, LoRA init scale 0.01, base-model init scale 0.02, a temp directory with "base" and
  # "merged" subpaths, and the subclass-built config. Tests: merged-save produces files, all
  # projection keys present, forward-pass equivalence (skipped unless a loader is provided).
  # KNOWN ISSUE: the class declares abstract methods, so test loaders that collect this base class
  # directly would fail to instantiate it; it is meant to be used only via subclasses.
  """Base test class for LoRA merged model saving and loading.

  Subclasses must implement:
  - create_config(): Create model-specific configuration
  - create_checkpoint(): Create safetensors checkpoint from model
  - get_lora_module_path(): Return regex pattern for LoRA target modules
  - get_projection_keys(): Return list of projection keys to verify
  - save_merged_model(): Call model-specific save function
  - create_model_from_checkpoint(): Load model from safetensors (optional)
  """

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: test fixture setup: sets LoRA hyperparameters and scales, creates a fresh temp dir and
    # derives base checkpoint and merged output dir paths (not created), then builds the config via
    # the subclass hook. Side effects: temp directory creation.
    """Set up test fixtures."""
    super().setUp()

    # LoRA hyperparameters
    self.rank = 4
    self.alpha = 16
    self.random_seed = 42
    self.lora_scale = 0.01
    self.base_model_scale = 0.02

    # Create temporary directories
    self.temp_dir = tempfile.mkdtemp()
    self.base_checkpoint_dir = os.path.join(self.temp_dir, "base")
    self.merged_output_dir = os.path.join(self.temp_dir, "merged")

    # Create model config (architecture-specific)
    self.config = self.create_config()

  def tearDown(self):
    # [FUNCTION DOC]
    # HELPER: removes the temp directory tree if it exists, then calls the parent teardown.
    """Clean up test fixtures."""
    if os.path.exists(self.temp_dir):
      shutil.rmtree(self.temp_dir)
    super().tearDown()

  @abc.abstractmethod
  def create_config(self):
    # [FUNCTION DOC]
    # HELPER: abstract hook returning a small model config for fast tests.
    """Create model-specific configuration.

    Returns:
      Model configuration object with small dimensions for fast testing.
    """
    pass

  @abc.abstractmethod
  def create_checkpoint(self, model) -> str:
    # [FUNCTION DOC]
    # HELPER: abstract hook writing the base model's weights as a safetensors checkpoint plus a
    # minimal config.json and returning the checkpoint directory.
    """Extract model weights and save in safetensors format.

    This method should:
    1. Create checkpoint directory if needed
    2. Extract weights from model
    3. Apply architecture-specific transformations
    4. Save to safetensors file
    5. Create minimal config.json

    Args:
      model: Base model to extract weights from.

    Returns:
      Path to the created checkpoint directory.
    """
    pass

  @abc.abstractmethod
  def get_lora_module_path(self) -> str:
    # [FUNCTION DOC]
    # HELPER: abstract hook returning the qwix LoRA module-path regex.
    """Get regex pattern for LoRA target modules.

    Returns:
      Regex pattern string for qwix.LoraProvider.
      Example:
      ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj"
    """
    pass

  @abc.abstractmethod
  def get_projection_keys(self, layer_idx: int) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: abstract hook returning the safetensors keys of the projection weights in a layer.
    """Get list of projection weight keys for a given layer.

    Args:
      layer_idx: Layer index.

    Returns:
      List of safetensors keys for projections in this layer.
      Example: ["model.layers.0.self_attn.q_proj.weight", ...]
    """
    pass

  @abc.abstractmethod
  def save_merged_model(self, lora_model):
    # [FUNCTION DOC]
    # HELPER: abstract hook that merges LoRA into the base weights and saves to the merged output dir.
    """Save LoRA merged model using architecture-specific function.

    Args:
      lora_model: Model with LoRA applied.
    """
    pass

  def create_model_from_checkpoint(self, checkpoint_dir: str):
    # [FUNCTION DOC]
    # HELPER: optional hook loading a model from a safetensors checkpoint; the default returns None,
    # and its being un-overridden makes the equivalence test skip.
    """Load model from safetensors checkpoint (optional).

    Override this method to enable forward pass equivalence testing.

    Args:
      checkpoint_dir: Path to checkpoint directory.

    Returns:
      Loaded model, or None if not implemented.
    """
    return None

  def get_model_class(self):
    # [FUNCTION DOC]
    # HELPER: hook returning the model class for dummy creation; default raises NotImplementedError.
    """Get model class for dummy model creation.

    Override if needed for architecture-specific model class.

    Returns:
      Model class to use with create_dummy_model.
    """
    # Default implementation - subclasses should override
    raise NotImplementedError("Subclass must implement get_model_class()")

  def _create_base_model(self):
    # [FUNCTION DOC]
    # HELPER: creates the base model via the dummy model creator with the subclass model class, the
    # config, no mesh, float32, the fixture seed and base scale. Output: the model.
    """Create a base model with deterministic weights."""
    model = create_dummy_model(
        model_class=self.get_model_class(),
        config=self.config,
        mesh=None,
        dtype=jnp.float32,
        random_seed=self.random_seed,
        scale=self.base_model_scale,
    )
    return model

  def _apply_lora_to_model(self, model):
    # [FUNCTION DOC]
    # HELPER: applies qwix LoRA (subclass regex, fixture rank and alpha) using the model's example
    # inputs, then randomly initializes the LoRA weights. Output: the LoRA model.
    """Apply LoRA to a model.

    Args:
      model: Base model to apply LoRA to.

    Returns:
      Model with LoRA applied and initialized with random weights.
    """
    # Create LoRA provider
    lora_provider = qwix.LoraProvider(
        module_path=self.get_lora_module_path(),
        rank=self.rank,
        alpha=self.alpha,
    )

    # Get model input and apply LoRA
    model_input = model.get_model_input()
    lora_model = qwix.apply_lora_to_model(model, lora_provider, **model_input)

    # Initialize LoRA weights
    self._initialize_lora_weights(lora_model)

    return lora_model

  def _initialize_lora_weights(self, lora_model):
    # [FUNCTION DOC]
    # HELPER: randomly initializes LoRA weights in place. Starting from a PRNG key seeded with seed
    # + 1000, for each layer in the model's layers list and each of the "attn" and "mlp" submodules
    # present, iterates over all attribute names of the submodule; for each non-None attribute that
    # has w_lora_a, w_lora_b, kernel_lora_a or kernel_lora_b, splits the key and sets that parameter to
    # lora_scale times a standard normal sample of the same shape. Side effects: mutates parameters.
    """Initialize LoRA weights with random values.

    Override this method for architecture-specific weight initialization.

    Args:
      lora_model: Model with LoRA applied.
    """
    lora_rng = jax.random.PRNGKey(self.random_seed + 1000)

    for layer in lora_model.layers:
      # Process all attributes in the layer
      for attr_name in ["attn", "mlp"]:
        if not hasattr(layer, attr_name):
          continue

        module = getattr(layer, attr_name)

        # Find all projections in this module
        for proj_name in dir(module):
          proj = getattr(module, proj_name, None)
          if proj is None:
            continue

          # Initialize w_lora_a and w_lora_b (for attention projections)
          if hasattr(proj, "w_lora_a"):
            lora_rng, key = jax.random.split(lora_rng)
            proj.w_lora_a.value = self.lora_scale * jax.random.normal(
                key, proj.w_lora_a.value.shape
            )
          if hasattr(proj, "w_lora_b"):
            lora_rng, key = jax.random.split(lora_rng)
            proj.w_lora_b.value = self.lora_scale * jax.random.normal(
                key, proj.w_lora_b.value.shape
            )

          # Initialize kernel_lora_a and kernel_lora_b (for MLP projections)
          if hasattr(proj, "kernel_lora_a"):
            lora_rng, key = jax.random.split(lora_rng)
            proj.kernel_lora_a.value = self.lora_scale * jax.random.normal(
                key, proj.kernel_lora_a.value.shape
            )
          if hasattr(proj, "kernel_lora_b"):
            lora_rng, key = jax.random.split(lora_rng)
            proj.kernel_lora_b.value = self.lora_scale * jax.random.normal(
                key, proj.kernel_lora_b.value.shape
            )

  def _compute_expected_merged_weight(self, base_weight, lora_a, lora_b):
    # [FUNCTION DOC]
    # HELPER: computes the expected merged weight. Converts LoRA A and B to numpy; a 3-D A is reshaped
    # by merging its first two dims, a 3-D B by merging its last two dims; delta is A times B scaled
    # by alpha / rank; result is base weight plus the transposed delta (base assumed stored as
    # output-by-input). Output: numpy array. Shape mismatch raises ValueError.
    """Compute expected merged weight: base + (lora_a @ lora_b) * (alpha/rank).

    Args:
      base_weight: Base weight tensor.
      lora_a: LoRA A matrix.
      lora_b: LoRA B matrix.

    Returns:
      Expected merged weight.
    """
    lora_a = np.array(lora_a)
    lora_b = np.array(lora_b)

    # Handle 3D tensors (flatten to 2D for matmul)
    if lora_a.ndim == 3:
      d0, d1, d2 = lora_a.shape
      lora_a = lora_a.reshape(d0 * d1, d2)
    if lora_b.ndim == 3:
      d0, d1, d2 = lora_b.shape
      lora_b = lora_b.reshape(d0, d1 * d2)

    # Compute LoRA delta
    lora_delta = (lora_a @ lora_b) * (self.alpha / self.rank)

    # Merge with base weight
    expected = np.array(base_weight) + lora_delta.T

    return expected

  # ============================================================================
  # Common test methods
  # ============================================================================

  def test_save_lora_merged_model(self):
    # [FUNCTION DOC]
    # Test: creates base model and checkpoint, applies LoRA, verifies at least one LoRA parameter
    # exists in the graph, saves the merged model, and verifies the merged dir, its model.safetensors
    # (non-empty when loaded) and a copied config.json exist.
    """Test LoRA merged model saving and output file creation."""
    # Create base model and checkpoint
    base_model = self._create_base_model()
    self.create_checkpoint(base_model)

    # Apply LoRA
    lora_model = self._apply_lora_to_model(base_model)

    # Verify LoRA was applied
    lora_param_count = 0
    for _, value in nnx.iter_graph(lora_model):
      if isinstance(value, nnx.LoRAParam):
        lora_param_count += 1
    self.assertGreater(
        lora_param_count, 0, "LoRA should be applied to the model"
    )

    # Save merged model
    self.save_merged_model(lora_model)

    # Verify output files exist
    self.assertTrue(os.path.exists(self.merged_output_dir))
    merged_safetensors_path = os.path.join(
        self.merged_output_dir, "model.safetensors"
    )
    self.assertTrue(os.path.exists(merged_safetensors_path))

    # Verify merged state is non-empty
    merged_state = safe_np.load_file(merged_safetensors_path)
    self.assertGreater(len(merged_state), 0)

    # Verify config was copied
    config_path = os.path.join(self.merged_output_dir, "config.json")
    self.assertTrue(os.path.exists(config_path))

  def test_all_projections_merged(self):
    # [FUNCTION DOC]
    # Test: after creating checkpoint, applying LoRA and saving, loads the merged safetensors and
    # verifies each projection key for layer 0 is present, not None and non-empty.
    """Test that all projection types are present in merged state."""
    # Create base model and checkpoint
    base_model = self._create_base_model()
    self.create_checkpoint(base_model)

    # Apply LoRA and save merged model
    lora_model = self._apply_lora_to_model(base_model)
    self.save_merged_model(lora_model)

    # Load merged state
    merged_state = safe_np.load_file(
        os.path.join(self.merged_output_dir, "model.safetensors")
    )

    # Verify all projection keys exist
    projection_keys = self.get_projection_keys(layer_idx=0)
    for key in projection_keys:
      self.assertIn(key, merged_state, f"Missing projection key: {key}")
      self.assertIsNotNone(merged_state[key])
      self.assertGreater(merged_state[key].size, 0)

  def test_forward_pass_equivalence(self):
    # [FUNCTION DOC]
    # Test: skipped when the subclass did not override the loader hook. Otherwise creates base model
    # and checkpoint, applies LoRA, runs a forward pass on fixed random inputs, saves the merged model,
    # reloads it, runs the same forward pass and asserts outputs match within absolute tolerance 1e-2.
    """Test that outputs match between LoRA model and merged-then-reloaded model.

    This test is optional and requires implementing
    create_model_from_checkpoint().
    """
    # Skip if not implemented
    if (
        self.create_model_from_checkpoint.__func__
        is LoraParamsTestBase.create_model_from_checkpoint
    ):
      self.skipTest("create_model_from_checkpoint not implemented")

    # Create base model and checkpoint
    base_model = self._create_base_model()
    self.create_checkpoint(base_model)

    # Apply LoRA
    lora_model = self._apply_lora_to_model(base_model)

    # Create test inputs
    input_tokens, positions, attention_mask = self._create_test_inputs()

    # Get output from LoRA model
    lora_output, _ = self._run_forward_pass(
        lora_model, input_tokens, positions, attention_mask
    )

    # Save merged model
    self.save_merged_model(lora_model)

    # Reload merged model
    reloaded_model = self.create_model_from_checkpoint(self.merged_output_dir)

    # Get output from reloaded model
    merged_output, _ = self._run_forward_pass(
        reloaded_model, input_tokens, positions, attention_mask
    )

    # Compare outputs
    np.testing.assert_allclose(
        lora_output,
        merged_output,
        atol=1e-2,
        err_msg=(
            "Forward pass outputs don't match between LoRA model and merged"
            " model"
        ),
    )

  def _create_test_inputs(self):
    # [FUNCTION DOC]
    # HELPER: builds forward inputs: batch 2, length 10 random int tokens (key 123) in [0, vocab) where
    # vocab is the config's vocab_size or else num_embed; positions 0..9 repeated per row; mask None.
    # Output: tuple (tokens, positions, mask).
    """Create test inputs for forward pass.

    Override this method for architecture-specific input requirements.

    Returns:
      Tuple of (input_tokens, positions, attention_mask).
    """
    batch_size = 2
    seq_len = 10

    input_tokens = jax.random.randint(
        jax.random.PRNGKey(123),
        shape=(batch_size, seq_len),
        minval=0,
        maxval=self.config.vocab_size
        if hasattr(self.config, "vocab_size")
        else self.config.num_embed,
    )
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    attention_mask = None  # Override if needed

    return input_tokens, positions, attention_mask

  def _run_forward_pass(self, model, input_tokens, positions, attention_mask):
    # [FUNCTION DOC]
    # HELPER: calls the model with keyword args input_tokens, positions, cache None and attention_mask;
    # returns whatever it returns (expected (output, cache)).
    """Run forward pass through model.

    Override this method for architecture-specific forward pass parameters.

    Args:
      model: Model to run.
      input_tokens: Input token IDs.
      positions: Position IDs.
      attention_mask: Attention mask (or None).

    Returns:
      Tuple of (output, cache).
    """
    return model(
        input_tokens=input_tokens,
        positions=positions,
        cache=None,
        attention_mask=attention_mask,
    )
