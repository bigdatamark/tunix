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
# External-environment test for loading Llama-3.2-1B weights from HuggingFace safetensors via
# tunix.models.llama3.params.create_model_from_safe_tensors.
# Side effects at import time: prints progress and downloads "meta-llama/Llama-3.2-1B" (ignoring
# .pth files) with huggingface_hub.snapshot_download into the HF cache; on any download failure it
# prints the error and sets the module-level local path to None so the test is skipped.
# When run directly it uses unittest's text runner inside IPython (detected by get_ipython existing)
# or absltest.main otherwise. Outputs: test results and printed messages.

# BEGIN-GOOGLE-INTERNAL
# Tests for Llama3 model parameter loading from safetensors files.
# WARNING: This test is intended for external environments, such as GCE.
# It should not be run as part of a standard internal codebase or Blaze build.

# Setup:
# 1. Run `huggingface-cli login` to authenticate with Hugging Face
# 2. Ensure you have the corresponding model access.

# Usage:
# Script: python params_test.py
# Jupyter: %run params_test.py

# Each test validates model loading, device placement, and display
# functionality.
# Tests are skipped if model paths are not configured.
# END-GOOGLE-INTERNAL

import unittest

from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download
import jax
from tunix.models.llama3 import model
from tunix.models.llama3 import params


# --- Model Download Section ---
print("Downloading Llama-3.2-1B model from Hugging Face...")

# Model ID on Hugging Face
model_id_1b = "meta-llama/Llama-3.2-1B"

# Patterns to ignore during download (e.g., PyTorch .pth weights)
ignore_patterns = [
    "*.pth",
]

# Download the 1B model and get its local path
# pylint: disable=broad-exception-caught
try:
  local_model_path_1b = snapshot_download(
      repo_id=model_id_1b, ignore_patterns=ignore_patterns
  )
  print(f"Llama-3.2-1B model downloaded to: {local_model_path_1b}")
except Exception as e:
  local_model_path_1b = None
  print(f"Failed to download Llama-3.2-1B model: {e}")


class Llama3ParamsTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests Llama3 parameter loading, device placement and display.

  def _test_model_loading(self, model_name, model_path, config_fn, mesh_config):
    # [FUNCTION DOC]
    # HELPER: shared test body. Inputs: model_name (label), model_path (local directory or None),
    # config_fn (config factory), mesh_config (pair of mesh shape and axis names). Skips if model_path is
    # None. Builds config and an Auto-typed mesh, then inside the mesh loads the model from safetensors
    # (failing the test with a message on any exception), asserts it is non-None, calls nnx.display
    # (failing on exception), asserts display succeeded and prints a success message.
    """Common test logic for loading different Llama3 models.

    Args:
      model_name: Name of the model for logging (e.g., "1B")
      model_path: Path to the model directory
      config_fn: Function to create model config (e.g.,
        model.ModelConfig.llama3p2_1b)
      mesh_config: Mesh configuration tuple (e.g., [(1, 4), ("fsdp", "tp")])
    """
    if model_path is None:
      self.skipTest(
          "No local model path available. Please download"
          f" Llama3-{model_name} model first."
      )

    config = config_fn()
    mesh = jax.make_mesh(*mesh_config, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_config[0]))

    with mesh:
      # Test that model loading completes without exceptions
      # pylint: disable=broad-exception-caught
      try:
        llama3 = params.create_model_from_safe_tensors(model_path, config, mesh)
      except Exception as e:
        self.fail(
            f"create_model_from_safe_tensors failed for {model_name} model: {e}"
        )

      # Test that the model was created successfully
      self.assertIsNotNone(llama3)

      # Test model display completes without exceptions
      try:
        nnx.display(llama3)
        display_success = True
      except Exception as e:
        self.fail(f"nnx.display failed for {model_name} model: {e}")

      self.assertTrue(
          display_success, f"{model_name} model display should succeed"
      )

      print(f"Llama3-{model_name} model loaded and displayed successfully")

  def test_create_model_from_safe_tensors_1b(self):
    # [FUNCTION DOC]
    # Runs the shared loading check for the downloaded 1B model with config llama3p2_1b on a (1, N) mesh
    # with axes ("fsdp", "tp").
    # Use the globally downloaded path for the 1B model
    self._test_model_loading(
        model_name="1B",
        model_path=local_model_path_1b,
        config_fn=model.ModelConfig.llama3p2_1b,
        mesh_config=[(1, len(jax.devices())), ("fsdp", "tp")],
    )


if __name__ == "__main__":
  # Check if running in Jupyter/IPython environment
  try:
    get_ipython()
    # Running in Jupyter/IPython - run tests directly to avoid SystemExit
    suite = unittest.TestLoader().loadTestsFromTestCase(Llama3ParamsTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
  except NameError:
    # Running as a script - use absltest.main()
    absltest.main()
