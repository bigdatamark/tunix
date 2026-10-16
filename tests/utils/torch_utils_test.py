"""Tests for torch_utils."""

# [MODULE DOC]
# Unit tests for tunix.utils.torch_utils.torch_key_to_jax_key, which maps a
# PyTorch checkpoint key to a JAX key using a dict of regex pattern ->
# (replacement template, transform) entries. No license header present.
from absl.testing import absltest
from absl.testing import parameterized

from tunix.utils import torch_utils


# [CLASS DOC]
# Parameterized test case for torch_key_to_jax_key success and failure modes.
class TorchUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='simple_case',
          mapping={
              'layer.(\\d+).mlp': (r'jax_layer.\1.jax_mlp', None),
          },
          source_key='layer.12.mlp',
          expected=('jax_layer.12.jax_mlp', None),
      ),
  )
  # [FUNCTION DOC]
  # Verifies that a single matching regex is substituted using backreferences
  # (e.g. 'layer.12.mlp' -> 'jax_layer.12.jax_mlp') and the transform (None)
  # is returned alongside it. Inputs: mapping, source_key, expected tuple.
  def test_torch_key_to_jax_key_success(self, mapping, source_key, expected):
    got = torch_utils.torch_key_to_jax_key(mapping, source_key)
    self.assertEqual(expected, got)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_match',
          mapping={
              'layer.(\\d+).mlp': ('jax_layer.{}.jax_mlp', None),
          },
          source_key='other_layer.12.mlp',
      ),
      dict(
          testcase_name='multiple_matches',
          mapping={
              'layer.(\\d+).mlp': ('jax_layer.{}.jax_mlp', None),
              'layer.*': ('jax_layer.all', None),
          },
          source_key='layer.12.mlp',
      ),
  )
  # [FUNCTION DOC]
  # Verifies ValueError is raised when no pattern matches the key or when
  # more than one pattern matches. Inputs: mapping and source_key.
  def test_torch_key_to_jax_key_failure(self, mapping, source_key):
    with self.assertRaises(ValueError):
      torch_utils.torch_key_to_jax_key(mapping, source_key)


if __name__ == '__main__':
  absltest.main()
