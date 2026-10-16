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
# Unit tests for tunix.models.gemma4.params: map_from_upstream_checkpoint (converts an upstream Gemma 4
# checkpoint, in semi-flat "transformer/..." path-string keys or fully nested form, to the Tunix nnx
# parameter tree: layer_N becomes layers/N, gating_einsum of shape (2, F, D) is split into transposed
# gate_proj and up_proj kernels, linear becomes down_proj unchanged, query_norm/key_norm are renamed to
# _query_norm/_key_norm, per_layer_embeddings becomes per_layer_input_embedding, non-text modules such as
# audio_encoder are skipped, and a gating_einsum whose first dim is not 2 raises ValueError),
# _prune_to_model_keys, _validate_param_shapes, create_model_from_checkpoint and create_tokenizer.
# Fixtures use tiny dimensions: vocab 5, embed 3, head dim 2, kv heads 1, heads 4, ffw 6, per-layer
# embedding 5. No side effects beyond mocks.

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax.traverse_util import flatten_dict
import numpy as np
from tunix.models.gemma4 import params

# Small array dimensions used across all fixtures.
_V, _D, _H, _KV, _N, _F, _PLE = 5, 3, 2, 1, 4, 6, 5


def _layer_arrays(offset: int | float = 0) -> dict[str, np.ndarray]:
  # [FUNCTION DOC]
  # HELPER: returns a dict of float32 per-layer fixture arrays (gate_up (2,F,D), down (F,D), q_w (N,D,H),
  # kv_w (2,KV,D,H), o_w (N,H,D), four norm scales of size D, skip_scale of size 1 equal to 0.5 plus
  # offset, query/key norm scales of size H), each filled with a range plus the given offset so layers can
  # be told apart.
  """Per-layer arrays with a numeric offset to distinguish layers."""
  o = np.float32(offset)
  return {
      'gate_up': (
          np.arange(2 * _F * _D, dtype=np.float32).reshape(2, _F, _D) + o
      ),
      'down': np.arange(_F * _D, dtype=np.float32).reshape(_F, _D) + o,
      'q_w': np.arange(_N * _D * _H, dtype=np.float32).reshape(_N, _D, _H) + o,
      'kv_w': (
          np.arange(2 * _KV * _D * _H, dtype=np.float32).reshape(2, _KV, _D, _H)
          + o
      ),
      'o_w': np.arange(_N * _H * _D, dtype=np.float32).reshape(_N, _H, _D) + o,
      'pre_attn': np.arange(_D, dtype=np.float32) + o,
      'post_attn': np.arange(_D, dtype=np.float32) + o,
      'pre_ffw': np.arange(_D, dtype=np.float32) + o,
      'post_ffw': np.arange(_D, dtype=np.float32) + o,
      'skip_scale': np.array([0.5 + offset], dtype=np.float32),
      'query_norm': np.arange(_H, dtype=np.float32) + o,
      'key_norm': np.arange(_H, dtype=np.float32) + o,
  }


def _semiflat_layer(idx: int, arrs: dict[str, np.ndarray]) -> dict[str, Any]:
  # [FUNCTION DOC]
  # HELPER: builds semi-flat upstream entries for one layer index: keys "transformer/layer_<idx>/..." for
  # attention norms and einsums, MLP gating_einsum and linear, the four layer norms, and the bare layer key
  # holding skip_scale, each mapping to a small dict of leaf name to array.
  """Builds semi-flat entries for one layer."""
  p = f'transformer/layer_{idx}'
  return {
      f'{p}/attn/_key_norm': {'scale': arrs['key_norm']},
      f'{p}/attn/_query_norm': {'scale': arrs['query_norm']},
      f'{p}/attn/attn_vec_einsum': {'w': arrs['o_w']},
      f'{p}/attn/kv_einsum': {'w': arrs['kv_w']},
      f'{p}/attn/q_einsum': {'w': arrs['q_w']},
      f'{p}/mlp/gating_einsum': {'w': arrs['gate_up']},
      f'{p}/mlp/linear': {'w': arrs['down']},
      f'{p}/post_attention_norm': {'scale': arrs['post_attn']},
      f'{p}/post_ffw_norm': {'scale': arrs['post_ffw']},
      f'{p}/pre_attention_norm': {'scale': arrs['pre_attn']},
      f'{p}/pre_ffw_norm': {'scale': arrs['pre_ffw']},
      f'{p}': {'skip_scale': arrs['skip_scale']},
  }


def _make_upstream_semiflat() -> dict[str, Any]:
  # [FUNCTION DOC]
  # HELPER: returns a semi-flat upstream checkpoint with embedder input_embedding (V,D), per-layer embeddings
  # (D,PLE), final norm scale, and two layers (offsets 0 and 100).
  """Builds a multi-layer upstream checkpoint in semi-flat 2-tuple format.

  Semi-flat keys: ('transformer/layer_0/attn/q_einsum', 'w')
  Includes layer_0 and layer_1 to exercise multi-layer index parsing.
  """
  embed = np.arange(_V * _D, dtype=np.float32).reshape(_V, _D)
  final_scale = np.arange(_D, dtype=np.float32)
  per_layer_emb = np.arange(_D * _PLE, dtype=np.float32).reshape(_D, _PLE)

  d = {
      'transformer/embedder': {'input_embedding': embed},
      'transformer/embedder/per_layer_embeddings': {'w': per_layer_emb},
      'transformer/final_norm': {'scale': final_scale},
  }
  d.update(_semiflat_layer(0, _layer_arrays(offset=0)))
  d.update(_semiflat_layer(1, _layer_arrays(offset=100)))
  return d


def _make_upstream_nested() -> dict[str, Any]:
  # [FUNCTION DOC]
  # HELPER: converts the semi-flat checkpoint into a fully nested dict by splitting each key on "/" and
  # nesting the leaf dicts, merging into an existing dict when a parent path (e.g. the bare layer key holding
  # skip_scale) collides with sub-paths.
  """Builds the same checkpoint in genuinely nested N-tuple format.

  Nested keys: ('transformer', 'layer_0', 'attn', 'q_einsum', 'w')
  """
  sf = _make_upstream_semiflat()
  nested = {}
  for key, sub_dict in sf.items():
    parts = tuple(key.split('/'))
    current = nested
    for part in parts[:-1]:
      current = current.setdefault(part, {})
    leaf = parts[-1]
    # Merge into existing dict to avoid overwriting sibling sub-paths
    # (e.g., 'transformer/layer_0' shouldn't clobber
    # 'transformer/layer_0/attn').
    if leaf in current and isinstance(current[leaf], dict):
      current[leaf].update(sub_dict)
    else:
      current[leaf] = sub_dict
  return nested


def _expected_keys_and_shapes() -> dict[tuple[str, ...], tuple[int, ...]]:
  # [FUNCTION DOC]
  # HELPER: returns the mapping from expected flattened output key tuples to shapes for embedder, final norm
  # and both layers (including split gate/up kernels of shape (D,F) and skip_scale of shape (1,)).
  """Returns expected output keys and shapes after mapping (both layers)."""
  result = {
      ('embedder', 'input_embedding'): (_V, _D),
      ('embedder', 'per_layer_input_embedding'): (_D, _PLE),
      ('final_norm', 'scale'): (_D,),
  }
  for i in range(2):
    result.update({
        ('layers', i, 'attn', '_key_norm', 'scale'): (_H,),
        ('layers', i, 'attn', '_query_norm', 'scale'): (_H,),
        ('layers', i, 'attn', 'attn_vec_einsum', 'w'): (_N, _H, _D),
        ('layers', i, 'attn', 'kv_einsum', 'w'): (2, _KV, _D, _H),
        ('layers', i, 'attn', 'q_einsum', 'w'): (_N, _D, _H),
        ('layers', i, 'mlp', 'down_proj', 'kernel'): (_F, _D),
        ('layers', i, 'mlp', 'gate_proj', 'kernel'): (_D, _F),
        ('layers', i, 'mlp', 'up_proj', 'kernel'): (_D, _F),
        ('layers', i, 'post_attention_norm', 'scale'): (_D,),
        ('layers', i, 'post_ffw_norm', 'scale'): (_D,),
        ('layers', i, 'pre_attention_norm', 'scale'): (_D,),
        ('layers', i, 'pre_ffw_norm', 'scale'): (_D,),
        ('layers', i, 'skip_scale'): (1,),
    })
  return result


class MapFromUpstreamCheckpointTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests map_from_upstream_checkpoint on both checkpoint formats and edge cases.

  @parameterized.named_parameters(
      dict(testcase_name='semi_flat', make_fn=_make_upstream_semiflat),
      dict(testcase_name='nested', make_fn=_make_upstream_nested),
  )
  def test_keys_and_shapes(self, make_fn):
    # [FUNCTION DOC]
    # Verifies every expected key exists in the mapped output with the expected shape.
    """Verifies all expected keys exist with correct shapes."""
    upstream = make_fn()
    mapped = params.map_from_upstream_checkpoint(upstream)
    flat = flatten_dict(mapped)

    expected = _expected_keys_and_shapes()
    for key, shape in expected.items():
      with self.subTest(key=key):
        self.assertIn(key, flat, msg=f'Missing key {key}')
        self.assertEqual(
            flat[key].shape,
            shape,
            msg=(
                f'Shape mismatch for {key}: got {flat[key].shape}, want {shape}'
            ),
        )

  @parameterized.named_parameters(
      dict(testcase_name='semi_flat', make_fn=_make_upstream_semiflat),
      dict(testcase_name='nested', make_fn=_make_upstream_nested),
  )
  def test_mlp_gating_transpose(self, make_fn):
    # [FUNCTION DOC]
    # Verifies gate_proj and up_proj kernels of layer 0 equal the transposes of gating_einsum[0] and [1].
    # KNOWN ISSUE: the source gating tensor is the first flattened key containing "gating_einsum", which
    # relies on dict insertion order to pick layer 0.
    """Verifies MLP gating_einsum is split and transposed correctly."""
    upstream = make_fn()
    mapped = params.map_from_upstream_checkpoint(upstream)
    flat = flatten_dict(mapped)

    # Get the original gating_einsum value (layer 0).
    upstream_flat = flatten_dict(upstream)
    gate_up_key = [k for k in upstream_flat if 'gating_einsum' in str(k)][0]
    gate_up = upstream_flat[gate_up_key]

    np.testing.assert_array_equal(
        flat[('layers', 0, 'mlp', 'gate_proj', 'kernel')],
        gate_up[0].T,
    )
    np.testing.assert_array_equal(
        flat[('layers', 0, 'mlp', 'up_proj', 'kernel')],
        gate_up[1].T,
    )

  @parameterized.named_parameters(
      dict(testcase_name='semi_flat', make_fn=_make_upstream_semiflat),
      dict(testcase_name='nested', make_fn=_make_upstream_nested),
  )
  def test_passthrough_values(self, make_fn):
    # [FUNCTION DOC]
    # Verifies the layer 0 down_proj kernel equals the original linear weights (no transpose).
    """Verifies pass-through params (e.g., MLP down_proj) are not modified."""
    upstream = make_fn()
    mapped = params.map_from_upstream_checkpoint(upstream)
    flat = flatten_dict(mapped)

    # The down_proj kernel is passed through without transpose. Verify by
    # reconstructing the expected value from the fixture constants.
    expected_down = _layer_arrays(offset=0)['down']
    np.testing.assert_array_equal(
        flat[('layers', 0, 'mlp', 'down_proj', 'kernel')],
        expected_down,
    )

  def test_format_parity(self):
    # [FUNCTION DOC]
    # Verifies semi-flat and nested inputs produce identical key sets and identical values.
    """Both checkpoint formats must produce identical output."""
    sf_mapped = params.map_from_upstream_checkpoint(_make_upstream_semiflat())
    nested_mapped = params.map_from_upstream_checkpoint(_make_upstream_nested())

    sf_flat = flatten_dict(sf_mapped)
    nested_flat = flatten_dict(nested_mapped)

    self.assertEqual(set(sf_flat.keys()), set(nested_flat.keys()))
    for key, sf_value in sf_flat.items():
      with self.subTest(key=key):
        np.testing.assert_array_equal(
            sf_value,
            nested_flat[key],
            err_msg=f'Value mismatch for {key} between formats',
        )

  def test_non_layer_modules_skipped(self):
    # [FUNCTION DOC]
    # Verifies an extra transformer/audio_encoder module is dropped from the mapped output.

    upstream = _make_upstream_semiflat()
    upstream['transformer/audio_encoder'] = {
        'weight': np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }

    mapped = params.map_from_upstream_checkpoint(upstream)
    flat = flatten_dict(mapped)

    audio_keys = [k for k in flat if 'audio_encoder' in str(k)]
    self.assertEmpty(audio_keys)

  @parameterized.named_parameters(
      dict(
          testcase_name='query_norm_no_underscore',
          upstream={
              'transformer/layer_0/attn/query_norm': {
                  'scale': np.arange(2, dtype=np.float32),
              },
          },
          expected_key=('layers', 0, 'attn', '_query_norm', 'scale'),
          expected_shape=(2,),
      ),
      dict(
          testcase_name='key_norm_no_underscore',
          upstream={
              'transformer/layer_0/attn/key_norm': {
                  'scale': np.arange(2, dtype=np.float32),
              },
          },
          expected_key=('layers', 0, 'attn', '_key_norm', 'scale'),
          expected_shape=(2,),
      ),
      dict(
          testcase_name='embedder_per_layer_as_leaf',
          upstream={
              'transformer/embedder': {
                  'per_layer_input_embedding': np.ones(
                      (_D, _PLE), dtype=np.float32
                  ),
              },
          },
          expected_key=('embedder', 'per_layer_input_embedding'),
          expected_shape=(_D, _PLE),
      ),
      dict(
          testcase_name='embedder_mm_submodule',
          upstream={
              'transformer/embedder/mm_input_projection': {
                  'w': np.ones((4, 8), dtype=np.float32),
              },
          },
          expected_key=('embedder', 'mm_input_projection', 'w'),
          expected_shape=(4, 8),
      ),
  )
  def test_mapper_edge_cases(self, upstream, expected_key, expected_shape):
    # [FUNCTION DOC]
    # Verifies single-entry edge cases: query_norm and key_norm without underscore are mapped to _query_norm
    # and _key_norm, a per_layer_input_embedding leaf on the embedder is kept, and an embedder
    # mm_input_projection submodule maps to embedder/mm_input_projection/w with shapes preserved.

    mapped = params.map_from_upstream_checkpoint(upstream)
    flat = flatten_dict(mapped)
    self.assertIn(expected_key, flat, msg=f'Missing key {expected_key}')
    self.assertEqual(flat[expected_key].shape, expected_shape)

  def test_gating_einsum_bad_shape_raises(self):
    # [FUNCTION DOC]
    # Verifies a gating_einsum whose first dimension is 3 raises ValueError mentioning "gating_einsum shape[0]=2".

    upstream = {
        'transformer/layer_0/mlp/gating_einsum': {
            'w': np.ones((3, _F, _D), dtype=np.float32),
        },
    }
    with self.assertRaisesRegex(ValueError, r'gating_einsum shape\[0\]=2'):
      params.map_from_upstream_checkpoint(upstream)


class PruneToModelKeysTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests _prune_to_model_keys, which keeps only checkpoint leaves whose flattened key exists in the model state.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: fixture patching nnx.to_pure_dict in the params module to the identity function.
    super().setUp()
    self.enter_context(
        mock.patch.object(
            params.nnx, 'to_pure_dict', side_effect=lambda x: x, autospec=True
        )
    )

  def test_extra_keys_pruned(self):
    # [FUNCTION DOC]
    # Verifies keys absent from the model state are removed.

    checkpoint = {'a': {'x': np.array(1.0)}, 'b': {'y': np.array(2.0)}}
    model_state = {'a': {'x': np.array(0.0)}}

    pruned = params._prune_to_model_keys(checkpoint, model_state)  # pylint: disable=protected-access
    flat = flatten_dict(pruned)

    self.assertIn(('a', 'x'), flat)
    self.assertNotIn(('b', 'y'), flat)

  def test_expected_keys_preserved(self):
    # [FUNCTION DOC]
    # Verifies keys present in both are kept.

    shared = {'a': {'x': np.array(1.0)}, 'b': {'y': np.array(2.0)}}

    pruned = params._prune_to_model_keys(shared, shared)  # pylint: disable=protected-access
    flat = flatten_dict(pruned)

    self.assertIn(('a', 'x'), flat)
    self.assertIn(('b', 'y'), flat)

  def test_deep_nesting_with_integer_keys(self):
    # [FUNCTION DOC]
    # Verifies pruning works with nested dicts containing integer keys (round-trip of flatten/unflatten).
    """Flatten/unflatten round-trip preserves deeply nested integer keys."""
    checkpoint = {
        'layers': {0: {'mlp': {'gate_proj': {'kernel': np.array(1.0)}}}},
        'extra': {'junk': np.array(2.0)},
    }
    model_state = {
        'layers': {0: {'mlp': {'gate_proj': {'kernel': np.array(0.0)}}}},
    }

    pruned = params._prune_to_model_keys(checkpoint, model_state)  # pylint: disable=protected-access
    flat = flatten_dict(pruned)

    self.assertIn(('layers', 0, 'mlp', 'gate_proj', 'kernel'), flat)
    self.assertNotIn(('extra', 'junk'), flat)


class ValidateParamShapesTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests _validate_param_shapes, which raises on missing keys or shape mismatches and warns on extra keys
  # or non-array values.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: fixture patching nnx.to_pure_dict in the params module to the identity function.
    super().setUp()
    self.enter_context(
        mock.patch.object(
            params.nnx, 'to_pure_dict', side_effect=lambda x: x, autospec=True
        )
    )

  def test_missing_keys_raises(self):
    # [FUNCTION DOC]
    # Verifies a model key missing from the checkpoint raises ValueError mentioning "missing keys".

    checkpoint = {'a': {'x': np.array([1.0])}}
    model_state = {
        'a': {'x': np.array([0.0])},
        'b': {'y': np.array([0.0])},
    }

    with self.assertRaisesRegex(ValueError, 'missing keys'):
      params._validate_param_shapes(checkpoint, model_state)  # pylint: disable=protected-access

  def test_shape_mismatch_raises(self):
    # [FUNCTION DOC]
    # Verifies differing shapes raise ValueError mentioning "Shape mismatch".

    checkpoint = {'a': {'x': np.array([1.0, 2.0])}}  # shape (2,)
    model_state = {'a': {'x': np.array([0.0])}}  # shape (1,)

    with self.assertRaisesRegex(ValueError, 'Shape mismatch'):
      params._validate_param_shapes(checkpoint, model_state)  # pylint: disable=protected-access

  def test_matching_shapes_passes(self):
    # [FUNCTION DOC]
    # Verifies identical structures with matching shapes do not raise.

    data = {'a': {'x': np.array([1.0, 2.0])}}
    params._validate_param_shapes(data, data)  # pylint: disable=protected-access

  def test_extra_keys_warns(self):
    # [FUNCTION DOC]
    # Verifies checkpoint keys not in the model produce a WARNING log mentioning "extra keys".
    """Extra checkpoint keys (superset of model) should log a warning."""
    checkpoint = {
        'a': {'x': np.array([1.0])},
        'b': {'y': np.array([2.0])},
    }
    model_state = {'a': {'x': np.array([0.0])}}

    with self.assertLogs(level='WARNING') as cm:
      params._validate_param_shapes(checkpoint, model_state)  # pylint: disable=protected-access

    self.assertTrue(
        any('extra keys' in msg.lower() for msg in cm.output),
        msg=f'Expected warning about extra keys, got: {cm.output}',
    )

  def test_non_array_values_warns(self):
    # [FUNCTION DOC]
    # Verifies non-array leaf values produce a WARNING log mentioning "non-array".

    data = {'a': {'x': 'not_an_array'}}

    with self.assertLogs(level='WARNING') as cm:
      params._validate_param_shapes(data, data)  # pylint: disable=protected-access

    self.assertTrue(
        any('non-array' in msg.lower() for msg in cm.output),
        msg=f'Expected warning about non-array values, got: {cm.output}',
    )


class CreateModelTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests create_model_from_checkpoint orchestration with all heavy dependencies mocked.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: fixture patching map_from_upstream_checkpoint, _prune_to_model_keys, _validate_param_shapes,
    # nnx.update, nnx.state, nnx.eval_shape and ocp.StandardCheckpointer. eval_shape returns a fake model;
    # restore, map, prune and state return a small fake params dict; config is an autospec ModelConfig and the
    # checkpoint path is "/fake/checkpoint".
    super().setUp()
    self.mock_map = self.enter_context(
        mock.patch.object(params, 'map_from_upstream_checkpoint', autospec=True)
    )
    self.mock_prune = self.enter_context(
        mock.patch.object(params, '_prune_to_model_keys', autospec=True)
    )
    self.mock_validate = self.enter_context(
        mock.patch.object(params, '_validate_param_shapes', autospec=True)
    )
    self.mock_update = self.enter_context(
        mock.patch.object(params.nnx, 'update', autospec=True)
    )
    self.mock_state = self.enter_context(
        mock.patch.object(params.nnx, 'state', autospec=True)
    )
    self.mock_eval_shape = self.enter_context(
        mock.patch.object(params.nnx, 'eval_shape', autospec=True)
    )
    self.mock_ckptr_cls = self.enter_context(
        mock.patch.object(params.ocp, 'StandardCheckpointer', autospec=True)
    )

    self.fake_model = mock.create_autospec(object, instance=True)
    self.mock_eval_shape.return_value = self.fake_model
    self.fake_params = {'layers': {0: {'w': np.array([1.0])}}}
    self.mock_ckptr_cls.return_value.restore.return_value = self.fake_params
    self.mock_map.return_value = self.fake_params
    self.mock_prune.return_value = self.fake_params
    self.mock_state.return_value = self.fake_params
    self.config = mock.create_autospec(
        params.model_lib.ModelConfig, instance=True
    )
    self.checkpoint_path = '/fake/checkpoint'

  def test_create_model_restores_checkpoint(self):
    # [FUNCTION DOC]
    # Verifies the checkpointer restores exactly once from the given checkpoint path.
    params.create_model_from_checkpoint(
        self.checkpoint_path, self.config, mesh=None
    )
    self.mock_ckptr_cls.return_value.restore.assert_called_once_with(
        self.checkpoint_path
    )

  def test_create_model_processes_params(self):
    # [FUNCTION DOC]
    # Verifies mapping, pruning and shape validation are each invoked once.
    params.create_model_from_checkpoint(
        self.checkpoint_path, self.config, mesh=None
    )
    self.mock_map.assert_called_once()
    self.mock_prune.assert_called_once()
    self.mock_validate.assert_called_once()

  def test_create_model_updates_nnx_state(self):
    # [FUNCTION DOC]
    # Verifies nnx.update is called once to load the params into the model.
    params.create_model_from_checkpoint(
        self.checkpoint_path, self.config, mesh=None
    )
    self.mock_update.assert_called_once()

  def test_create_model_returns_model_instance(self):
    # [FUNCTION DOC]
    # Verifies the function returns the model produced by nnx.eval_shape.
    result = params.create_model_from_checkpoint(
        self.checkpoint_path, self.config, mesh=None
    )
    self.assertIs(result, self.fake_model)

  @mock.patch.object(params.spm, 'SentencePieceProcessor', autospec=True)
  @mock.patch.object(params.epath, 'Path', autospec=True)
  def test_create_tokenizer(self, mock_path_cls, mock_spm_cls):
    # [FUNCTION DOC]
    # Verifies create_tokenizer reads the file bytes via epath.Path and loads them into a
    # SentencePieceProcessor with LoadFromSerializedProto, returning that processor.
    """Verifies tokenizer loads bytes and initializes processor."""
    fake_bytes = b'fake-sentencepiece-model'
    mock_path_cls.return_value.read_bytes.return_value = fake_bytes
    mock_processor = mock_spm_cls.return_value

    result = params.create_tokenizer('/fake/tokenizer.model')

    mock_path_cls.assert_called_once_with('/fake/tokenizer.model')
    mock_processor.LoadFromSerializedProto.assert_called_once_with(fake_bytes)
    self.assertIs(result, mock_processor)


if __name__ == '__main__':
  absltest.main()
