# [MODULE DOC]
# Tests for tunix.models.safetensors_loader.load_and_create_model using a toy transformer
# (test_common.ToyTransformer with 4 layers, 4 kv heads, head dim 16). The test saves the toy model's
# weights as a safetensors file and checks that loading reproduces identical state, for absolute,
# relative and "./"-relative paths, for the optimized and original loader modes, and for a gs:// path
# (with the GCS download mocked). Side effects: writes temp files and changes the working directory
# (restored via cleanup).
"""Tests for safetensors_loader."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from safetensors import numpy as stnp
from tunix.models import safetensors_loader
from tunix.tests import test_common
from tunix.utils import env_utils


def key_mapping(config):
  # [FUNCTION DOC]
  # HELPER: key mapping function passed to the loader. Ignores its config argument and returns a dict
  # from regex patterns on checkpoint tensor names to a pair of (target model parameter path template
  # with backreferences, transform None). Covers the embedding, per-layer attention query/key/value/out
  # kernels, MLP w1/w2 kernels and biases, and lm_head kernel/bias; all names map to themselves.
  del config
  return {
      r'^emb\.embedding$': ('emb.embedding', None),
      r'^layers\.(\d+)\.attn\.query\.kernel$': (
          r'layers.\1.attn.query.kernel',
          None,
      ),
      r'^layers\.(\d+)\.attn\.key\.kernel$': (
          r'layers.\1.attn.key.kernel',
          None,
      ),
      r'^layers\.(\d+)\.attn\.value\.kernel$': (
          r'layers.\1.attn.value.kernel',
          None,
      ),
      r'^layers\.(\d+)\.attn\.out\.kernel$': (
          r'layers.\1.attn.out.kernel',
          None,
      ),
      r'^layers\.(\d+)\.w1\.kernel$': (r'layers.\1.w1.kernel', None),
      r'^layers\.(\d+)\.w1\.bias$': (r'layers.\1.w1.bias', None),
      r'^layers\.(\d+)\.w2\.kernel$': (r'layers.\1.w2.kernel', None),
      r'^layers\.(\d+)\.w2\.bias$': (r'layers.\1.w2.bias', None),
      r'^lm_head\.kernel$': ('lm_head.kernel', None),
      r'^lm_head\.bias$': ('lm_head.bias', None),
  }


class SafetensorsLoaderTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests safetensors loading round-trips for the toy transformer.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # HELPER: class fixture. Builds the toy model with seed 0, captures its nnx state, and builds a
    # dict of name to numpy array for every parameter covered by the key mapping. The w2 bias of each
    # layer is stored wrapped in an nnx.Param to exercise handling of Param-wrapped values.
    super().setUpClass()
    config = test_common.ModelConfig(num_layers=4, num_kv_heads=4, head_dim=16)
    cls.model = test_common.ToyTransformer(config=config, rngs=nnx.Rngs(0))

    cls.state = nnx.state(cls.model)
    cls.tensors = {
        'emb.embedding': np.array(cls.state['emb']['embedding'].value),
        'lm_head.kernel': np.array(cls.state['lm_head']['kernel'].value),
        'lm_head.bias': np.array(cls.state['lm_head']['bias'].value),
    }
    for i in range(cls.model.config.num_layers):
      layer_state = cls.state['layers'][i]
      cls.tensors[f'layers.{i}.attn.query.kernel'] = np.array(
          layer_state['attn']['query']['kernel'].value
      )
      cls.tensors[f'layers.{i}.attn.key.kernel'] = np.array(
          layer_state['attn']['key']['kernel'].value
      )
      cls.tensors[f'layers.{i}.attn.value.kernel'] = np.array(
          layer_state['attn']['value']['kernel'].value
      )
      cls.tensors[f'layers.{i}.attn.out.kernel'] = np.array(
          layer_state['attn']['out']['kernel'].value
      )
      cls.tensors[f'layers.{i}.w1.kernel'] = np.array(
          layer_state['w1']['kernel'].value
      )
      cls.tensors[f'layers.{i}.w1.bias'] = np.array(
          layer_state['w1']['bias'].value
      )
      cls.tensors[f'layers.{i}.w2.kernel'] = np.array(
          layer_state['w2']['kernel'].value
      )
      # Test that nnx.Param are correctly handled.
      cls.tensors[f'layers.{i}.w2.bias'] = nnx.Param(
          np.array(layer_state['w2']['bias'].value),
      )

  @parameterized.named_parameters(
      *(([dict(testcase_name='opt_loader_enabled', mode='optimized')] 
         if not env_utils.is_internal_env() else []) + [
          dict(testcase_name='absolute_path', path_type='abs'),
          dict(testcase_name='relative_path', path_type='rel'),
          dict(testcase_name='relative_dot_path', path_type='rel_dot'),
          dict(testcase_name='opt_loader_disabled', mode='original'),
      ])
  )
  def test_load_and_create_model(
      self, path_type='abs', mode='auto'
  ):
    # [FUNCTION DOC]
    # Verifies load_and_create_model reproduces the original state. Inputs: path_type ("abs", "rel" or
    # "rel_dot"; default "abs") and mode ("auto", "optimized" or "original"; default "auto"). The
    # "optimized" case is only generated outside the internal environment. Creates a temp dir (falling back
    # to a tempfile directory), optionally chdirs to its parent to use a relative path, writes the tensors
    # to model.safetensors, loads with float32 dtype and asserts every leaf of the loaded state equals the
    # original. Raises ValueError for an unknown path_type.
    # KNOWN ISSUE: the fallback uses the name of an unreferenced TemporaryDirectory object, which may be
    # deleted when the object is garbage-collected; the directory is then recreated with makedirs and
    # never cleaned up.
    try:
      st_dir_abs = self.create_tempdir().full_path
    except Exception:  # pylint: disable=broad-except
      st_dir_abs = tempfile.TemporaryDirectory().name
      os.makedirs(st_dir_abs, exist_ok=True)

    origin_dir = os.getcwd()
    self.addCleanup(os.chdir, origin_dir)
    if path_type == 'abs':
      load_dir = st_dir_abs
    elif path_type == 'rel':
      os.chdir(os.path.dirname(st_dir_abs))
      load_dir = os.path.basename(st_dir_abs)
    elif path_type == 'rel_dot':
      os.chdir(os.path.dirname(st_dir_abs))
      load_dir = f'./{os.path.basename(st_dir_abs)}'
    else:
      raise ValueError(f'Unknown path_type: {path_type}')

    filename = os.path.join(st_dir_abs, 'model.safetensors')
    stnp.save_file(self.tensors, filename)

    loaded_model = safetensors_loader.load_and_create_model(
        load_dir,
        test_common.ToyTransformer,
        self.model.config,
        key_mapping,
        dtype=jnp.float32,
        mode=mode,
    )
    loaded_state = nnx.state(loaded_model)
    jax.tree.map(
        np.testing.assert_array_equal,
        self.state,
        loaded_state,
    )

  def test_load_and_create_model_from_gcs(self):
    # [FUNCTION DOC]
    # Verifies loading from a "gs://bucket/model" path calls load_file_from_gcs exactly once with that
    # path (mocked to return a local dir containing the saved tensors) and reproduces the original state.
    # Skipped in the internal environment.
    if env_utils.is_internal_env():
      self.skipTest('GCS is not supported in GOOGLE_INTERNAL_PACKAGE_PATH')
    try:
      st_dir_abs = self.create_tempdir().full_path
    except Exception:  # pylint: disable=broad-except
      st_dir_abs = tempfile.TemporaryDirectory().name
      os.makedirs(st_dir_abs, exist_ok=True)

    filename = os.path.join(st_dir_abs, 'model.safetensors')
    stnp.save_file(self.tensors, filename)

    with mock.patch.object(
        safetensors_loader, 'load_file_from_gcs'
    ) as mock_load:
      mock_load.return_value = st_dir_abs
      loaded_model = safetensors_loader.load_and_create_model(
          'gs://bucket/model',
          test_common.ToyTransformer,
          self.model.config,
          key_mapping,
          dtype=jnp.float32,
      )
      mock_load.assert_called_once_with('gs://bucket/model')

    loaded_state = nnx.state(loaded_model)
    jax.tree.map(
        np.testing.assert_array_equal,
        self.state,
        loaded_state,
    )


if __name__ == '__main__':
  absltest.main()
