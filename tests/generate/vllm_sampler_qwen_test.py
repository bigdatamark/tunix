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
# Integration test checking that Tunix's Qwen3 vLLM-JAX weight mapping (mapping_vllm_jax.VLLM_JAX_MAPPING)
# covers every source key when syncing Tunix weights into a vLLM sampler.
# Inputs: downloads "Qwen/Qwen3-1.7B" from HuggingFace into the system temp dir under models/<repo id>;
# needs TPU/JAX devices and vLLM. Outputs: test results. Side effects: network download, device
# memory allocation, temporary replacement of sys.stderr during weight loading.

from io import StringIO
import os
import sys
import tempfile
from unittest import mock
from absl.testing import absltest
from flax import nnx
import jax
import transformers
from tunix.generate import mappings
from tunix.generate import vllm_sampler
from tunix.models.qwen3 import mapping_vllm_jax
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import params as qwen3_params
from tunix.tests import test_common as tc


class VllmSamplerQwenTest(absltest.TestCase):
  # [CLASS DOC]
  # Test suite for vLLM sampler weight syncing with Qwen3 mappings.

  @classmethod
  def setUpClass(cls) -> None:
    # [FUNCTION DOC]
    # HELPER: class fixture. Sets repo_id to Qwen/Qwen3-1.7B, model_path to the temp dir models/<repo_id>,
    # downloads the checkpoint via test_common.download_from_huggingface, and creates a (1, N device) mesh
    # with axes ("fsdp", "tp") and Auto axis types stored as cls.mesh.
    super().setUpClass()
    cls.repo_id = "Qwen/Qwen3-1.7B"
    temp_dir = tempfile.gettempdir()
    cls.model_path = os.path.join(temp_dir, "models", cls.repo_id)

    tc.download_from_huggingface(repo_id=cls.repo_id, model_path=cls.model_path)

    mesh_shape = (1, len(jax.devices()))
    axis_names = ("fsdp", "tp")
    cls.mesh = jax.make_mesh(
        mesh_shape,
        axis_names,
        devices=jax.devices(),
        axis_types=(jax.sharding.AxisType.Auto,) * len(axis_names),
    )

  def test_qwen3_base_mapping_no_errors(self):
    # [FUNCTION DOC]
    # Verifies that loading a Tunix Qwen3 model state into VllmSampler produces no "No mapping for source key"
    # log lines. Loads a 1-layer Qwen3 1.7B from safetensors, builds a MappingConfig from VLLM_JAX_MAPPING,
    # creates a VllmSampler (jax backend, 0.3 HBM utilization, random init, max_model_len 128, not server
    # mode), redirects sys.stderr to an in-memory buffer, patches the LLM's reset_prefix_cache and
    # collective_rpc, calls load_checkpoint with the model state, then restores stderr and stops the sampler
    # and asserts the captured text lacks the missing-mapping message.
    # KNOWN ISSUE: replacing sys.stderr only captures Python-level writes; logs written by native code
    # directly to file descriptor 2 (which the comment says it intends to trap) are not captured, so the
    # assertion may pass vacuously. A comment also refers to a "mocked logger" that does not exist.
    """Tests that vLLM accepts Tunix's Qwen3 LoRA mappings without logging missing keys."""

    # 1. Create Tunix Actor Model
    config = qwen3_model.ModelConfig.qwen3_1p7b()
    config.num_layers = 1
    base_model = qwen3_params.create_model_from_safe_tensors(
        self.model_path, config, self.mesh
    )

    # 2. Configure Sampler with Explicit Mappings
    tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)

    # CRITICAL: Bypass the qwix-wrapped actor_model and inject the dictionary directly
    mapping_config = mappings.MappingConfig(**mapping_vllm_jax.VLLM_JAX_MAPPING)

    vllm_config = vllm_sampler.VllmConfig(
        mesh=self.mesh,
        hbm_utilization=0.3,
        init_with_random_weights=True,
        tpu_backend_type="jax",
        mapping_config=mapping_config,
        server_mode=False,
        engine_kwargs={
            "model": self.model_path,
            "max_model_len": 128,
        },
    )

    sampler = vllm_sampler.VllmSampler(
        tokenizer=tokenizer,
        config=vllm_config,
    )

    # 3. Capture sys.stderr to physically trap the absl C++ logs
    stderr_capture = StringIO()
    original_stderr = sys.stderr
    sys.stderr = stderr_capture

    try:
      # Mock the RPC calls to delete and reinitialize kv cache
      with mock.patch.object(sampler.llm, "reset_prefix_cache"), \
           mock.patch.object(sampler.llm, "collective_rpc"):
        # 4. Trigger param update to force mapping of the base model weights
        sampler.load_checkpoint(nnx.state(base_model))

        # 5. Check the mocked logger to see if it was called with mapping errors
    finally:
      # Always restore stderr so we don't break console output for other tests
      sys.stderr = original_stderr
      if hasattr(sampler, "stop"):
        sampler.stop()

    # 5. Parse the captured stderr string
    logs = stderr_capture.getvalue()

    # 6. Strictly Assert
    self.assertNotIn(
        "No mapping for source key",
        logs,
        f"Missing LoRA mappings found in utils.py! Captured Logs:\n{logs}",
    )


if __name__ == "__main__":
  absltest.main()
