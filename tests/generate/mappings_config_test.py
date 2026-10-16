# [MODULE DOC]
# Unit tests for tunix.generate.mappings.MappingConfig.build, which resolves weight-name mapping
# configuration (to_hf_mappings, lora_to_hf_mappings, to_hf_hook_fns, to_hf_transpose_keys) from
# an existing MappingConfig, a dict of values or zero-arg callables, or a model instance.
# Inputs: a Llama3 1B model built once per class. Outputs: absltest results.
# Side effects: allocates model parameters in memory (JAX/flax nnx).
"""Tests for tunix.generate.mappings.MappingConfig utilities."""

from absl.testing import absltest

from tunix.generate import mappings
from tunix.models.llama3 import model as model_lib
from flax import nnx

class MappingConfigTest(absltest.TestCase):
  # [CLASS DOC]
  # Test suite for MappingConfig.build covering error, passthrough, dict, model and override inputs.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # HELPER: class-level fixture. Creates nnx.Rngs with params seed 0 and builds a Llama3 model using
    # ModelConfig.llama3p2_1b, storing it as the class attribute "model". Side effect: allocates weights.
    rng = nnx.Rngs(params=0)
    cls.model = model_lib.Llama3(model_lib.ModelConfig.llama3p2_1b(), rngs=rng)

  def test_from_object_with_none_errors_out(self):
    # [FUNCTION DOC]
    # Verifies that MappingConfig.build called with no mapping object and no model raises an
    # AssertionError whose message contains "Either mapping_obj or model must be provided".
    with self.assertRaisesRegex(
        AssertionError,
        "Either mapping_obj or model must be provided"
    ):
      cfg = mappings.MappingConfig.build()

  def test_from_object_with_mapping_config_instance(self):
    # [FUNCTION DOC]
    # Verifies that passing an existing MappingConfig instance to build returns the identical object.
    original = mappings.MappingConfig(
        to_hf_mappings={'foo': ('bar', ('tp',))},
        lora_to_hf_mappings={'lora': ('baz', ('fsdp',))},
        to_hf_hook_fns={'foo': lambda x: x},
        to_hf_transpose_keys={'foo': (1, 0)},
    )
    cfg = mappings.MappingConfig.build(original)
    self.assertIs(cfg, original)

  def test_from_object_with_dict_and_callables(self):
    # [FUNCTION DOC]
    # Verifies that build accepts a dict whose values are zero-argument callables, invokes each callable,
    # and stores the resulting dicts in the corresponding MappingConfig fields.
    mapping_obj = {
        'to_hf_mappings': lambda: {'a': ('b', ('tp',))},
        'lora_to_hf_mappings': lambda: {'c': ('d', ('fsdp',))},
        'to_hf_hook_fns': lambda: {'a': lambda x: x},
        'to_hf_transpose_keys': lambda: {'embedding': (1, 0)},
    }
    cfg = mappings.MappingConfig.build(mapping_obj)
    self.assertEqual(cfg.to_hf_mappings, {'a': ('b', ('tp',))})
    self.assertEqual(cfg.lora_to_hf_mappings, {'c': ('d', ('fsdp',))})
    self.assertEqual(
        cfg.to_hf_transpose_keys,
        {'embedding': (1, 0)},
    )
    self.assertIn('a', cfg.to_hf_hook_fns)

  def test_build_mapping_config_with_model_only(self):
    # [FUNCTION DOC]
    # Verifies that building from the Llama3 model yields model-provided mappings: the embedder input
    # embedding and a LoRA gate_proj entry exist, and to_hf_transpose_keys equals embedding -> (1, 0).
    # KNOWN ISSUE: the first two checks use assertTrue with two arguments, so the expected tuple is
    # treated as a message and only truthiness of the mapping value is checked, not equality.
    cfg = mappings.MappingConfig.build(self.model)
    self.assertTrue(
        cfg.to_hf_mappings['embedder.input_embedding'],
        (
          'model.embed.embedding',
          ('model', None),
        )
    )

    self.assertTrue(
        cfg.lora_to_hf_mappings['layers.*.mlp.gate_proj.kernel_lora_a'],
        (
          'model.layers.*.mlp.gate_proj.kernel_lora_a',
          (None, None),
        )
    )

    self.assertEqual(cfg.to_hf_transpose_keys, {'embedding': (1, 0)},)

  def test_build_mapping_config_with_overrides(self):
    # [FUNCTION DOC]
    # Verifies that a dict containing plain (non-callable) values for to_hf_mappings and to_hf_hook_fns
    # is used as-is: the mapping equals the override and the hook entry "override" exists and is callable.
    override = {'embedder.input_embedding': (
          'fake.path.embedding',
          ('fake_dim', None),
      )}
    cfg = mappings.MappingConfig.build(
        {"to_hf_mappings":override,
        "to_hf_hook_fns":{'override': lambda x: x},
        }
    )
    self.assertEqual(cfg.to_hf_mappings, override)
    self.assertIn('override', cfg.to_hf_hook_fns)
    self.assertTrue(callable(cfg.to_hf_hook_fns['override']))


if __name__ == '__main__':
  absltest.main()
