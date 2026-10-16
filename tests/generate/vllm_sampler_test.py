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
# Tests for tunix.generate.vllm_sampler.VllmSampler. Integration tests (VllmSamplerTest) download
# meta-llama/Llama-3.2-1B-Instruct to the temp dir, compare greedy outputs of the vanilla Tunix sampler
# and the vLLM sampler (batch, data-parallel and server modes), check concurrent requests from an executor
# complete out of order, and check sampling kwargs from config and call are merged into vLLM
# SamplingParams. Unit tests (VllmSamplerConfigTest) check parallelism plumbing with a mocked LLM.
# Side effects at import: sets environment variable SKIP_JAX_PRECOMPILE to "1". Other side effects:
# network downloads, device memory, prints.

import asyncio
import os
import tempfile
import time
from unittest import mock
from absl.testing import absltest
from flax import nnx
import jax
import numpy as np
import transformers
from tunix.generate import mappings
from tunix.generate import sampler as vanilla_sampler
from tunix.generate import vllm_sampler
from tunix.models.dummy_model_creator import create_dummy_model
from tunix.models.llama3 import model as llama_lib
from tunix.models.llama3 import params as llama_params
from tunix.sft import utils as base_utils
from tunix.tests import test_common as tc
import asyncio

os.environ["SKIP_JAX_PRECOMPILE"] = "1"


class VllmSamplerTest(absltest.TestCase):
  # [CLASS DOC]
  # Hardware integration tests for VllmSampler with a Llama 3.2 1B Instruct model.

  @classmethod
  def setUpClass(cls) -> None:
    # [FUNCTION DOC]
    # HELPER: class fixture downloading the model to the temp dir under models/<repo id>, disabling LoRA
    # (enable_lora False pending vLLM support), and creating a (1, N device) Auto mesh with axes ("fsdp", "tp").
    super().setUpClass()
    cls.repo_id = "meta-llama/Llama-3.2-1B-Instruct"
    temp_dir = tempfile.gettempdir()
    cls.model_path = os.path.join(temp_dir, "models", cls.repo_id)

    tc.download_from_huggingface(repo_id=cls.repo_id, model_path=cls.model_path)

    # TODO(b/432096319): Enable after LoRA support in vLLM
    cls.enable_lora = False

    mesh_shape = (1, len(jax.devices()))  # e.g., (1, 8) for v2-8
    axis_names = ("fsdp", "tp")
    cls.mesh = jax.make_mesh(
        mesh_shape,
        axis_names,
        devices=jax.devices(),
        axis_types=(jax.sharding.AxisType.Auto,) * len(axis_names),
    )

  def load_llama3_model(self, model_version: str, enable_lora: bool = False):
    # [FUNCTION DOC]
    # HELPER: loads the Tunix Llama3 model for a supported repo id (3.2-1B or 3.1-8B Instruct; otherwise
    # AssertionError) from self.model_path on self.mesh. If enable_lora, wraps it with LoRA (rank 64, alpha 64)
    # on all attention and MLP projections and prints a message. Returns (model, model config).
    model_config = {
        "meta-llama/Llama-3.2-1B-Instruct": llama_lib.ModelConfig.llama3p2_1b,
        "meta-llama/Llama-3.1-8B-Instruct": llama_lib.ModelConfig.llama3p1_8b,
    }
    assert (
        model_version in model_config
    ), f"Invalid model version: {model_version}"
    model_config = model_config[model_version]()

    llama3 = llama_params.create_model_from_safe_tensors(
        self.model_path, model_config, self.mesh
    )
    if enable_lora:
      llama3 = tc.get_lora_model(
          llama3,
          model_path=".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
          rank=64,
          alpha=64.0,
          mesh=self.mesh,
      )
      print(f"Loaded LoRA model: {model_version} with LoRA enabled")
    # nnx.display(llama3)
    return llama3, model_config

  # Parametized test always fails on vLLM HBM usage exceeding limit, no matter how much HBM we allocated to it, and no matter how we clear the Jax cache (delete all the live arrays, gc collect, clear cache, clear test cache). vLLM will allocate all the assigned HBM to weights + KV cache. The conclusion is parametized test doesn't reset Jax properly, therefore the 2nd test adds on top of the previous HBM usage. This is the workaround for that.
  def test_vllm_sampler_batch_mode(self):
    # [FUNCTION DOC]
    # Runs the shared vLLM comparison in batch (non-server) mode. Separate test methods are used instead of
    # parameterization because HBM is not released between parameterized runs.
    self._run_vllm_sampler(server_mode=False)

  def test_vllm_sampler_batch_mode_with_data_parallel(self):
    # [FUNCTION DOC]
    # Runs the shared vLLM comparison in batch mode with data_parallel_size 2.
    self._run_vllm_sampler(server_mode=False, data_parallel_size=2)

  def test_vllm_sampler_server_mode(self):
    # [FUNCTION DOC]
    # Runs the shared vLLM comparison in server (async driver) mode.
    self._run_vllm_sampler(server_mode=True)

  def _run_vllm_sampler(self, server_mode, data_parallel_size: int = -1):
    # [FUNCTION DOC]
    # HELPER: shared comparison body. Inputs: server_mode (bool) and data_parallel_size (default -1, meaning
    # inferred). Loads the Tunix model and tokenizer, templatizes three prompts, runs the vanilla sampler
    # (cache 512, greedy top_k 1, 128 steps, padded), builds a MappingConfig from the model and a VllmConfig
    # (jax backend, 0.2 HBM, random init, optional LoRA config, max_model_len 512, prefix caching), creates
    # the VllmSampler and asserts its mesh differs from the test mesh, loads the Tunix state with
    # reset_prefix_cache and collective_rpc patched on the engine (driver engine in server mode, else llm),
    # samples with the same settings, validates both outputs against expected keywords, asserts the embedding
    # equals vLLM's embed table, and stops the sampler in server mode.
    # KNOWN ISSUE: the tokenizer is loaded twice (the first result is overwritten), and the sampler is not
    # stopped if an assertion fails first in server mode.
    tunix_model, model_config = self.load_llama3_model(
        self.repo_id, enable_lora=self.enable_lora
    )

    base_utils.show_hbm_usage("After loading tunix model")

    model_tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.model_path
    )

    lora_config = None
    if self.enable_lora:
      lora_config = {
          "rank": 64,
          "alpha": 64.0,
          "module_path": ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
          # "dropout": 0.0,
          # "bias": "none",
      }

    # Sampler setup
    model_tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.model_path
    )

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    prompts = [
        "Hello, my name is Tom.",
        "The capital of France is",
        "why is sky blue?",
    ]

    inputs = tc.batch_templatize(prompts, model_tokenizer)

    vn_sampler = vanilla_sampler.Sampler(
        transformer=tunix_model,
        tokenizer=model_tokenizer,
        cache_config=vanilla_sampler.CacheConfig(
            cache_size=512,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )
    vanilla_output = vn_sampler(
        input_strings=inputs,
        max_generation_steps=128,  # Changed from 768 to 128 for vLLM
        max_prompt_length=None,  # Use default max prompt length
        temperature=0.0,
        # top_p=0.9,
        top_k=1,
        seed=0,
        echo=False,
        pad_output=True,  # Use padding for output
    )

    mapping_config = mappings.MappingConfig.build(tunix_model)

    vllm_config = vllm_sampler.VllmConfig(
        mesh=self.mesh,
        hbm_utilization=0.2,
        init_with_random_weights=True,
        tpu_backend_type="jax",
        mapping_config=mapping_config,
        lora_config=lora_config,
        server_mode=server_mode,
        data_parallel_size=data_parallel_size,
        engine_kwargs={
            "model": self.model_path,
            "max_model_len": 512,
            "enable_prefix_caching": True,
        },  # Test kwargs forwarding
    )

    vl_sampler = vllm_sampler.VllmSampler(
        tokenizer=model_tokenizer,
        config=vllm_config,
    )
    # vLLM construct its own mesh
    self.assertNotEqual(vl_sampler.mesh, self.mesh)
    state = nnx.state(tunix_model)
    # Mock the RPC calls to delete and reinitialize kv cache
    mock_llm = vl_sampler._driver.llm_engine if server_mode else vl_sampler.llm
    with mock.patch.object(mock_llm, "reset_prefix_cache"), \
        mock.patch.object(mock_llm, "collective_rpc"):
        vl_sampler.load_checkpoint(state)

    base_utils.show_hbm_usage("After loading vLLM sampler")

    vllm_output = vl_sampler(
        input_strings=inputs,
        max_generation_steps=128,  # Changed from 768 to 128 for vLLM
        max_prompt_length=None,  # Use default max prompt length
        temperature=0.0,
        # top_p=0.9,
        top_k=1,
        seed=0,
        echo=False,
        pad_output=True,  # Use padding for output
    )

    expected_output_pattern = [
        (prompts[0], ["Tom", "Hello"]),
        (prompts[1], ["Paris"]),
        (prompts[2], ["Rayleigh", "scattering"]),
    ]

    print("-" * 50)
    print(f"Vanilla Generated text: {vanilla_output.text}")

    tc.validate_llm_outputs(expected_output_pattern, vanilla_output.text)

    print("-" * 50)
    print(f"vLLM Generated text: {vllm_output.text}")

    tc.validate_llm_outputs(expected_output_pattern, vllm_output.text)

    _, tunix_state = nnx.split(tunix_model)
    vllm_state = vl_sampler._model_runner.state

    self.assertTrue(
        np.allclose(
            tunix_state["embedder"]["input_embedding"].value,
            vllm_state["model"]["embed"]["embedding"].value,
        )
    )
    if vllm_config.server_mode:
      vl_sampler.stop()

  def test_vllm_sampler_run_in_executor_concurrency(self):
    # [FUNCTION DOC]
    # Verifies concurrent calls in server mode: loads the model into a server-mode VllmSampler (stopped on
    # cleanup), submits ten templated prompts from an asyncio loop via run_in_executor with decreasing delays
    # (later prompts sleep less), collects results by index, validates each output against expected keywords
    # (empty list for unlisted prompts) and asserts completion order is a permutation different from
    # submission order.
    tunix_model, _ = self.load_llama3_model(
        self.repo_id, enable_lora=self.enable_lora
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)

    mapping_config = mappings.MappingConfig.build(tunix_model)
    vllm_config = vllm_sampler.VllmConfig(
        mesh=self.mesh,
        hbm_utilization=0.2,
        init_with_random_weights=True,
        tpu_backend_type="jax",
        mapping_config=mapping_config,
        server_mode=True,
        engine_kwargs={
            "model": self.model_path,
            "max_model_len": 512,
            "enable_prefix_caching": True,
        },  # Test kwargs forwarding
    )

    vl_sampler = vllm_sampler.VllmSampler(
        tokenizer=tokenizer,
        config=vllm_config,
    )
    self.addCleanup(vl_sampler.stop)

    state = nnx.state(tunix_model)
    # Mock the RPC calls to delete and reinitialize kv cache
    with mock.patch.object(vl_sampler._driver.llm_engine, "reset_prefix_cache"), \
        mock.patch.object(vl_sampler._driver.llm_engine, "collective_rpc"):
        vl_sampler.load_checkpoint(state)

    base_prompts = [
        "Hello, my name is Tom.",
        "The capital of France is",
        "why is sky blue?",
        "Explain the theory of relativity in simple terms.",
        "List three benefits of regular exercise.",
        "Write a haiku about winter.",
        "Summarize the plot of Romeo and Juliet.",
        "Give me a recipe for pancakes.",
        "What is the boiling point of water at sea level?",
        "Share a motivational quote about perseverance.",
    ]
    prompts = list(base_prompts)
    templated_prompts = tc.batch_templatize(prompts, tokenizer)

    expected_keywords = {
        base_prompts[0]: ["Tom", "help"],
        base_prompts[1]: ["Paris"],
        base_prompts[2]: ["Rayleigh", "scattering"],
        base_prompts[3]: ["relativity", "physics"],
        base_prompts[4]: ["health", "can", "regular"],
        base_prompts[5]: ["winter"],
        base_prompts[6]: ["romeo", "juliet"],
        base_prompts[7]: ["pancake"],
        base_prompts[8]: ["100", "celsius"],
        base_prompts[9]: ["seven", "eight"],
    }
    prompt_expectations = [
        (prompt, expected_keywords.get(prompt, [])) for prompt in prompts
    ]

    delays = [0.05 * (len(prompts) - idx) for idx in range(len(prompts))]

    def _call_sampler(templated_prompt: str, delay: float):
      # [FUNCTION DOC]
      # HELPER: sleeps for delay seconds then calls the sampler on one templated prompt with greedy settings and
      # returns the sampler output.
      time.sleep(delay)
      return vl_sampler(
          input_strings=[templated_prompt],
          max_generation_steps=128,
          max_prompt_length=None,
          temperature=0.0,
          top_k=1,
          seed=0,
          echo=False,
          pad_output=True,
      )

    async def __call_sampler_async(
        index: int, templated_prompt: str, delay: float
    ):
      # [FUNCTION DOC]
      # HELPER: coroutine running the blocking call helper in the default executor and returning
      # (index, result).
      loop = asyncio.get_running_loop()
      result = await loop.run_in_executor(
          None,
          _call_sampler,
          templated_prompt,
          delay,
      )
      return index, result

    async def dispatch_requests():
      # [FUNCTION DOC]
      # HELPER: coroutine creating one task per templated prompt, awaiting them as they complete while
      # recording completion order and results by index, and returning (results ordered by index, completion
      # order list).
      loop = asyncio.get_running_loop()
      tasks = []
      for idx, templated_prompt in enumerate(templated_prompts):
        task = loop.create_task(
            __call_sampler_async(idx, templated_prompt, delays[idx])
        )

        tasks.append(task)

      completion_order = []
      results_by_idx = {}
      for task in asyncio.as_completed(tasks):
        idx, result = await task
        completion_order.append(idx)
        results_by_idx[idx] = result

      ordered_results = [results_by_idx[i] for i in range(len(tasks))]
      return ordered_results, completion_order

    results, completion_order = asyncio.run(dispatch_requests())

    self.assertLen(results, len(prompts))

    for (prompt, expectations), sampler_output in zip(
        prompt_expectations, results
    ):
      tc.validate_llm_outputs([(prompt, expectations)], sampler_output.text)

    expected_order = list(range(len(prompts)))
    self.assertCountEqual(completion_order, expected_order)
    self.assertNotEqual(
        completion_order,
        expected_order,
        msg=(
            "Responses returned strictly in submission order; "
            "expected out-of-order completions."
        ),
    )

  def test_vllm_sampler_sampling_kwargs(self):
    # [FUNCTION DOC]
    # Verifies config sampling_kwargs (frequency_penalty 0.5, presence_penalty 0.3) and per-call kwargs
    # (min_tokens 10) are both applied to the SamplingParams passed to llm.generate, by wrapping generate to
    # capture its sampling_params. Uses a dummy-weight Llama 3.2 1B model in non-server mode.
    # KNOWN ISSUE: llm.generate is replaced by assignment and never restored.
    """Test that sampling kwargs are correctly applied to sampling_params."""
    tunix_model = create_dummy_model(
          model_class=llama_lib.Llama3,
          config=llama_lib.ModelConfig.llama3p2_1b(),
          mesh=self.mesh,
          random_seed=3,
    )

    model_tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.model_path
    )

    prompts = ["Hello, my name is Tom."]
    inputs = tc.batch_templatize(prompts, model_tokenizer)

    mapping_config = mappings.MappingConfig.build(tunix_model)

    # Test 1: Config sampling_kwargs are applied
    config_sampling_kwargs = {
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
    }

    vllm_config = vllm_sampler.VllmConfig(
        mesh=self.mesh,
        hbm_utilization=0.2,
        init_with_random_weights=True,
        tpu_backend_type="jax",
        mapping_config=mapping_config,
        server_mode=False,
        sampling_kwargs=config_sampling_kwargs,
        engine_kwargs={
            "model": self.model_path,
            "max_model_len": 512,
            "enable_prefix_caching": True,
        },
    )

    vl_sampler = vllm_sampler.VllmSampler(
        tokenizer=model_tokenizer,
        config=vllm_config,
    )

    state = nnx.state(tunix_model)
    # Mock the RPC calls to delete and reinitialize kv cache
    with mock.patch.object(vl_sampler.llm, "reset_prefix_cache"), \
        mock.patch.object(vl_sampler.llm, "collective_rpc"):
        vl_sampler.load_checkpoint(state)

    # Mock the generate method to capture sampling_params
    original_generate = vl_sampler.llm.generate
    captured_sampling_params = []

    def mock_generate(prompts, sampling_params, **kwargs):
      # [FUNCTION DOC]
      # HELPER: records the sampling_params and delegates to the original generate.
      captured_sampling_params.append(sampling_params)
      return original_generate(prompts, sampling_params, **kwargs)

    vl_sampler.llm.generate = mock_generate

    # Call with additional method kwargs
    method_sampling_kwargs = {"min_tokens": 10}
    vl_sampler(
        input_strings=inputs,
        max_generation_steps=128,
        max_prompt_length=None,
        temperature=0.0,
        top_k=1,
        seed=0,
        echo=False,
        pad_output=True,
        **method_sampling_kwargs,
    )

    # Verify that both config and method kwargs were applied
    self.assertLen(captured_sampling_params, 1)
    sampling_params = captured_sampling_params[0]

    # Check config kwargs
    self.assertEqual(sampling_params.frequency_penalty, 0.5)
    self.assertEqual(sampling_params.presence_penalty, 0.3)

    # Check method kwargs
    self.assertEqual(sampling_params.min_tokens, 10)

  def test_vllm_sampler_sampling_kwargs_override(self):
    # [FUNCTION DOC]
    # Verifies a per-call kwarg (frequency_penalty 0.8) overrides the config value 0.5 while other config
    # kwargs (presence_penalty 0.3) still apply.
    """Test that method kwargs override config sampling_kwargs."""
    tunix_model = create_dummy_model(
          model_class=llama_lib.Llama3,
          config=llama_lib.ModelConfig.llama3p2_1b(),
          mesh=self.mesh,
          random_seed=3,
    )

    model_tokenizer = transformers.AutoTokenizer.from_pretrained(
        self.model_path
    )

    prompts = ["Hello, my name is Tom."]
    inputs = tc.batch_templatize(prompts, model_tokenizer)

    mapping_config = mappings.MappingConfig.build(tunix_model)

    # Config has frequency_penalty = 0.5
    config_sampling_kwargs = {
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
    }

    vllm_config = vllm_sampler.VllmConfig(
        mesh=self.mesh,
        hbm_utilization=0.2,
        init_with_random_weights=True,
        tpu_backend_type="jax",
        mapping_config=mapping_config,
        server_mode=False,
        sampling_kwargs=config_sampling_kwargs,
        engine_kwargs={
            "model": self.model_path,
            "max_model_len": 512,
            "enable_prefix_caching": True,
        },
    )

    vl_sampler = vllm_sampler.VllmSampler(
        tokenizer=model_tokenizer,
        config=vllm_config,
    )

    state = nnx.state(tunix_model)
    # Mock the RPC calls to delete and reinitialize kv cache
    with mock.patch.object(vl_sampler.llm, "reset_prefix_cache"), \
        mock.patch.object(vl_sampler.llm, "collective_rpc"):
        vl_sampler.load_checkpoint(state)

    # Mock the generate method to capture sampling_params
    original_generate = vl_sampler.llm.generate
    captured_sampling_params = []

    def mock_generate(prompts, sampling_params, **kwargs):
      # [FUNCTION DOC]
      # HELPER: records the sampling_params and delegates to the original generate.
      captured_sampling_params.append(sampling_params)
      return original_generate(prompts, sampling_params, **kwargs)

    vl_sampler.llm.generate = mock_generate

    # Call with method kwargs that override config kwargs
    method_sampling_kwargs = {"frequency_penalty": 0.8}  # Override from 0.5 to 0.8
    vl_sampler(
        input_strings=inputs,
        max_generation_steps=128,
        max_prompt_length=None,
        temperature=0.0,
        top_k=1,
        seed=0,
        echo=False,
        pad_output=True,
        **method_sampling_kwargs,
    )

    # Verify that method kwargs override config kwargs
    self.assertLen(captured_sampling_params, 1)
    sampling_params = captured_sampling_params[0]

    # Check that method kwarg overrides config kwarg
    self.assertEqual(sampling_params.frequency_penalty, 0.8)

    # Check that other config kwargs are still applied
    self.assertEqual(sampling_params.presence_penalty, 0.3)


class VllmSamplerConfigTest(absltest.TestCase):
  # [CLASS DOC]
  # Hardware-free tests of how VllmConfig parallelism settings are translated into vLLM engine args.
  """Unit tests for VllmSampler config plumbing (no hardware required)."""

  def _make_mock_mesh(self, total_devices):
    # [FUNCTION DOC]
    # HELPER: returns a MagicMock mesh whose shape is one axis of total_devices and whose flattened device ids
    # list is 0..total_devices-1.
    mesh = mock.MagicMock()
    mesh.shape = {"axis": total_devices}
    mesh.device_ids.flatten.return_value.tolist.return_value = list(
        range(total_devices)
    )
    return mesh

  def _make_sampler(self, config):
    # [FUNCTION DOC]
    # HELPER: constructs a VllmSampler with a mock tokenizer while vllm_sampler.LLM is patched, and returns it.
    with mock.patch("tunix.generate.vllm_sampler.LLM"):
      return vllm_sampler.VllmSampler(
          tokenizer=mock.MagicMock(), config=config
      )

  def test_expert_parallel_size_plumbed_to_sharding(self):
    # [FUNCTION DOC]
    # Verifies expert_parallel_size 2 on 8 devices sets expert_parallelism 2 in
    # additional_config.sharding.sharding_strategy and infers tensor_parallel_size 4 and data_parallel_size 1.
    mesh = self._make_mock_mesh(8)
    config = vllm_sampler.VllmConfig(
        mesh=mesh,
        expert_parallel_size=2,
        init_with_random_weights=False,
    )
    sampler = self._make_sampler(config)

    sharding_strategy = sampler.args["additional_config"]["sharding"][
        "sharding_strategy"
    ]
    # EP=2 should appear in the sharding strategy passed to vLLM.
    self.assertEqual(sharding_strategy["expert_parallelism"], 2)
    # With 8 total devices and EP=2, TP should be inferred as 4 and DP as 1.
    self.assertEqual(sampler.args["tensor_parallel_size"], 4)
    self.assertEqual(sampler.args["data_parallel_size"], 1)

  def test_reserved_keys_in_engine_kwargs_raise_value_error(self):
    # [FUNCTION DOC]
    # Verifies passing expert_parallel_size, tensor_parallel_size or data_parallel_size in engine_kwargs
    # raises ValueError naming the key when constructing VllmConfig.
    # Reserved VllmConfig fields (e.g. tp, dp, ep) must be set directly on
    # VllmConfig, not smuggled through engine_kwargs. Passing them via
    # engine_kwargs should raise a ValueError at config construction time
    # before any vLLM engine args are assembled.
    mesh = self._make_mock_mesh(8)
    for key in ("expert_parallel_size", "tensor_parallel_size", "data_parallel_size"):
      with self.subTest(key=key):
        with self.assertRaisesRegex(ValueError, key):
          vllm_sampler.VllmConfig(
              mesh=mesh,
              init_with_random_weights=False,
              engine_kwargs={key: 2},
          )

  def test_default_expert_parallel_size_is_one(self):
    # [FUNCTION DOC]
    # Verifies defaults on 8 devices give expert_parallelism 1, tensor_parallel_size 8, data_parallel_size 1.
    mesh = self._make_mock_mesh(8)
    config = vllm_sampler.VllmConfig(
        mesh=mesh,
        init_with_random_weights=False,
    )
    sampler = self._make_sampler(config)

    sharding_strategy = sampler.args["additional_config"]["sharding"][
        "sharding_strategy"
    ]
    self.assertEqual(sharding_strategy["expert_parallelism"], 1)
    self.assertEqual(sampler.args["tensor_parallel_size"], 8)
    self.assertEqual(sampler.args["data_parallel_size"], 1)


if __name__ == "__main__":
  absltest.main()
