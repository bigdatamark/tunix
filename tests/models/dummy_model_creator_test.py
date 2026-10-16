# [MODULE DOC]
# Tests for tunix.models.dummy_model_creator.create_dummy_model: builds randomly-initialised
# Llama3 1B and Qwen3 0.6B models on a JAX mesh over all local devices, checks nnx.display works,
# and runs a forward pass checking logits shape [batch, seq, vocab].
# Inputs: available JAX devices. Outputs: test results. Side effects: allocates full-size model
# weights in bfloat16 on devices. When executed directly in Colab/IPython it runs the suite via
# unittest's text runner instead of absltest.main to avoid SystemExit.
import unittest

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp

from tunix.models.dummy_model_creator import create_dummy_model
from tunix.models.llama3 import model as llama3_model
from tunix.models.qwen3 import model as qwen3_model
from tunix.tests import test_common


class DummyModelCreatorTest(absltest.TestCase):
  # [CLASS DOC]
  # Test suite exercising dummy model creation for several model families.

  def _test_dummy_model_creation(self, *, model_name, model_class, config_fn, mesh_config):
    # [FUNCTION DOC]
    # HELPER: shared test body. Inputs (keyword-only): model_name (string for messages), model_class
    # (model constructor), config_fn (zero-arg factory returning a model config), mesh_config (a pair
    # of mesh shape tuple and axis-name tuple). Steps: build the config; create a mesh with Auto axis
    # types for every axis; inside the mesh context call create_dummy_model with bfloat16 dtype and
    # seed 0, failing the test with a message if it raises; assert the model is non-None; call
    # nnx.display, failing on exception; get example inputs from model.get_model_input(); call the
    # model with keyword inputs and, on TypeError, fall back to positional (tokens, positions, cache,
    # attention_mask) where tokens is "last_tokens" if present else "input_tokens". Then assert logits
    # dims equal batch size, sequence length (from the tokens array) and vocab size (config.vocab_size
    # if present, else config.num_embed). Raises test failures via self.fail / assertions.
    config = config_fn()
    mesh = jax.make_mesh(*mesh_config, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_config[0]))

    with mesh:
      # pylint: disable=broad-exception-caught
      try:
        model = create_dummy_model(
            model_class,
            config,
            mesh=mesh,
            dtype=jnp.bfloat16,
            random_seed=0,
        )
      except Exception as e:
        self.fail(f"create_dummy_model failed for {model_name}: {e}")

      self.assertIsNotNone(model)

      # Validate nnx.display works
      try:
        nnx.display(model)
        display_success = True
      except Exception as e:
        self.fail(f"nnx.display failed for {model_name}: {e}")
      self.assertTrue(display_success, f"{model_name} model display should succeed")

      # Minimal forward to ensure parameters are materialized and callable
      inputs = model.get_model_input()
      try:
        logits, _ = model(**inputs)
      except TypeError:
        # If the model does not accept kwargs, fall back to positional ordering
        if 'last_tokens' in inputs:
          logits, _ = model(inputs['last_tokens'], inputs['positions'], inputs['cache'], inputs['attention_mask'])
        else:
          logits, _ = model(inputs['input_tokens'], inputs['positions'], inputs['cache'], inputs['attention_mask'])

      # Shape checks: [B, T, V]
      if 'last_tokens' in inputs:
        bsz, seqlen = inputs['last_tokens'].shape
      else:
        bsz, seqlen = inputs['input_tokens'].shape

      if hasattr(config, 'vocab_size'):
        vocab = config.vocab_size
      else:
        vocab = config.num_embed

      self.assertEqual(logits.shape[0], bsz, f"{model_name}: batch dimension mismatch")
      self.assertEqual(logits.shape[1], seqlen, f"{model_name}: seq length mismatch")
      self.assertEqual(logits.shape[2], vocab, f"{model_name}: vocab size mismatch")

  def test_llama3p2_1b(self):
    # [FUNCTION DOC]
    # Runs the shared dummy-creation check for Llama3 with the llama3p2_1b config on a (1, N) mesh
    # with axes ("fsdp", "tp"), N being the number of devices.
    self._test_dummy_model_creation(
        model_name="llama3p2_1b",
        model_class=llama3_model.Llama3,
        config_fn=llama3_model.ModelConfig.llama3p2_1b,
        mesh_config=[(1, len(jax.devices())), ("fsdp", "tp")],
    )

  def test_qwen3_0p6b(self):
    # [FUNCTION DOC]
    # Runs the shared dummy-creation check for Qwen3 with the qwen3_0p6b config on a (1, N) mesh
    # with axes ("fsdp", "tp").
    self._test_dummy_model_creation(
        model_name="qwen3_0p6b",
        model_class=qwen3_model.Qwen3,
        config_fn=qwen3_model.ModelConfig.qwen3_0p6b,
        mesh_config=[(1, len(jax.devices())), ("fsdp", "tp")],
    )

if __name__ == "__main__":
  # Check if running in Jupyter/IPython environment
  if test_common.is_running_in_colab():
    # Running in Jupyter/IPython - run tests directly to avoid SystemExit
    suite = unittest.TestLoader().loadTestsFromTestCase(DummyModelCreatorTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
  else:
    # Running as a script - use absltest.main()
    absltest.main()
