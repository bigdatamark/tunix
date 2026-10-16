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
# Generic loader that builds a tunix NNX model and fills its parameters from
# Hugging Face style safetensors files. Model packages supply a key-mapping
# function (config -> dict of regex on checkpoint key -> (tunix key template,
# transform)), resolved per key by torch_utils.torch_key_to_jax_key; transform
# is None or (permutation or None, reshape-shape or None) applied in that
# order. Two strategies: 'original' (safetensors library, threaded per-key
# reads, per-file processing, works with GCS/internal/pathways) and
# 'optimized' (manual header parse + mmap of the raw data block, zero-copy
# numpy views, one pass over all files). load_and_create_model selects one.
# Aliases: load_file_from_gcs (tunix.oss.utils; downloads a gs:// directory
# and returns a local path) and torch_key_to_jax_key.
"""Common utilities for loading OSS model weights from safetensors files.

Not compatible with GOOGLE_INTERNAL_PACKAGE_PATH.
"""

import concurrent.futures
import contextlib
import json
import mmap
import os
import struct
import threading

from absl import logging
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import safetensors.flax as safetensors
from tunix.oss import utils
from tunix.utils import compat
from tunix.utils import env_utils
from tunix.utils import torch_utils

load_file_from_gcs = utils.load_file_from_gcs

torch_key_to_jax_key = torch_utils.torch_key_to_jax_key


def stoi(s):
  # [FUNCTION DOC]
  # HELPER: Returns int(s) if s parses as an int, else s unchanged.
  """Convert string to int if possible, otherwise return as is."""
  try:
    return int(s)
  except ValueError:
    return s


def path_to_key(path):
  # [FUNCTION DOC]
  # HELPER: Converts a JAX tree path (sequence of key entries) to a dot-joined
  # string, using each entry's .key attribute when present (else the entry
  # itself), normalizing numeric strings to ints before str(). E.g. a path to
  # layers -> 0 -> attn -> q_proj -> w becomes 'layers.0.attn.q_proj.w'.
  """Convert path to string key."""
  return '.'.join(
      str(stoi(key.key if hasattr(key, 'key') else key)) for key in path
  )


def to_np_dtype(dtype):
  # [FUNCTION DOC]
  # HELPER: Maps a safetensors dtype string ('BF16','F16','F32','F64') or the
  # equivalent jnp dtype to the numpy/ml_dtypes type. Returns None for anything
  # else (e.g. integer or FP8 dtypes).
  # KNOWN ISSUE: unsupported dtypes silently yield None, which later makes numpy
  # use float64 or fail in confusing ways.
  if dtype == 'BF16' or dtype == jnp.bfloat16:
    return ml_dtypes.bfloat16
  elif dtype == 'F16' or dtype == jnp.float16:
    return np.float16
  elif dtype == 'F32' or dtype == jnp.float32:
    return np.float32
  elif dtype == 'F64' or dtype == jnp.float64:
    return np.float64


def load_safetensors_with_offsets(filepath):
  # [FUNCTION DOC]
  # HELPER: Parses a safetensors file without the library. Reads the 8-byte
  # little-endian header length and the JSON header, then for every tensor
  # (skipping '__metadata__') records name, element offset and count (byte
  # offsets divided by the item size), shape and dtype. The item size is taken
  # from the FIRST tensor's dtype and assumed common to all tensors. Opens the
  # file again, memory-maps it read-only and creates one numpy view over the
  # whole data block with that common dtype.
  # Returns (contiguous_array, tensor_metadata list, mmap object, open file
  # handle); the caller must close the handle (and mmap).
  # Raises: struct/JSON/IO errors on malformed files.
  # KNOWN ISSUE: files mixing dtypes (or with byte offsets not divisible by the
  # common item size) are decoded incorrectly; the docstring describes metadata
  # correctly but does not mention that dtype is assumed uniform.
  """Loads safetensors file and returns tensor metadata with offsets.

  Args:
    filepath: The path to the safetensors file.

  Returns:
    A tuple containing:
      - contiguous_array: A numpy array containing the concatenated tensor data.
      - tensor_metadata: A list of dictionaries, each containing metadata
        (name, offset_elements, size_elements, shape, dtype) for a tensor.
      - mm: The mmap object used to read the file.
      - f: The file handle.
  """
  with open(filepath, 'rb') as f:
    header_size_bytes = f.read(8)
    header_size = struct.unpack('<Q', header_size_bytes)[0]
    header_bytes = f.read(header_size)
    header = json.loads(header_bytes.decode('utf-8'))

  data_block_start_offset_bytes = 8 + header_size

  tensor_metadata = []

  itemsize = 2  # Default to bfloat16
  common_dtype = None
  for tensor_name, metadata in header.items():
    if tensor_name == '__metadata__':
      continue

    dtype = metadata['dtype']
    if common_dtype is None:
      common_dtype = dtype
      np_type = to_np_dtype(dtype)
      itemsize = np.dtype(np_type).itemsize

    start_byte, end_byte = metadata['data_offsets']
    shape = tuple(metadata['shape'])

    size_bytes = end_byte - start_byte
    size_elements = size_bytes // itemsize
    offset_elements = start_byte // itemsize

    tensor_metadata.append({
        'name': tensor_name,
        'offset_elements': offset_elements,
        'size_elements': size_elements,
        'shape': shape,
        'dtype': dtype,
    })

  file_size = os.path.getsize(filepath)
  data_size_bytes = file_size - data_block_start_offset_bytes
  total_elements = data_size_bytes // itemsize

  f = open(filepath, 'rb')

  mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
  contiguous_array = np.frombuffer(
      mm,
      dtype=to_np_dtype(common_dtype),
      count=total_elements,
      offset=data_block_start_offset_bytes,
  )

  return contiguous_array, tensor_metadata, mm, f


def load_and_create_model_orig(
    file_dir: str,
    model_class,
    config,
    key_mapping,
    mesh=None,
    preprocess_fn=None,
    dtype: jnp.dtype | None = None,
):
  # [FUNCTION DOC]
  # 'Original' loading strategy. Inputs as load_and_create_model (minus mode).
  # Steps: if file_dir is gs://, download it locally; glob '*.safetensors'
  # (ValueError if none). Under compat.set_mesh(mesh) (or a null context),
  # shape-evaluate the model to get an abstract model; split into graph def and
  # abstract state; pure-dict state; optional named shardings from the mesh;
  # build key_map = key_mapping(config). For each file: open with
  # safetensors.safe_open (numpy) and, in a thread pool of cpu_count workers,
  # process every key (see process_key) into a per-file dict of tunix key ->
  # jnp array (cast to dtype if given); re-raise the first worker exception;
  # apply preprocess_fn to the per-file dict; log a warning listing unmapped
  # keys; then tree-map over the state (with shardings) replacing each leaf whose
  # path key (or key + '.value') is in the per-file dict with the loaded array
  # placed via device_put (onto its sharding, after pulling to host, or onto
  # the first device). Leaves not loaded stay abstract.
  # Returns nnx.merge(graph_def, state) i.e. the model.
  # Raises: ValueError (no files, duplicate key within a file, shape mismatch),
  # RuntimeError wrapping per-key failures.
  # Side effects: file/GCS I/O, device allocation, logging.
  # KNOWN ISSUE: preprocess_fn is applied per file (breaks expert stacking when
  # a layer's experts span files); skipped_keys accumulates across files so the
  # warning repeats earlier keys; params never found remain ShapeDtypeStructs
  # without any error; the jnp.array conversion places every tensor on the
  # default device first, which is then pulled back to host.
  """Generic function to load model from safetensors files.

  Args:
      file_dir: Directory containing safetensors files
      model_class: Model class to instantiate
      config: Model configuration
      key_mapping: Function that returns key mapping dictionary
      mesh: Optional JAX mesh for sharding
      preprocess_fn: Optional function to preprocess loaded parameters
      dtype: Optional dtype to cast loaded parameters to

  Returns:
      Model instance with loaded weights
  """

  if file_dir.startswith('gs://'):
    file_dir = load_file_from_gcs(file_dir)

  files = list(epath.Path(file_dir).expanduser().glob('*.safetensors'))

  if not files:
    raise ValueError(f'No safetensors found in {file_dir}')

  # Create model structure
  context_manager = (
      compat.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()
  )

  with context_manager:
    model = nnx.eval_shape(lambda: model_class(config, rngs=nnx.Rngs(params=0)))

  graph_def, abs_state = nnx.split(model)
  state_dict = abs_state.to_pure_dict()

  if mesh is not None:
    sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
  else:
    sharding_dict = None

  key_map = key_mapping(config)

  file_lock = threading.Lock()

  # Load tensors from all files
  skipped_keys = []
  for f in files:
    file_loaded_tensors = {}
    with safetensors.safe_open(f, framework='numpy') as sf:
      keys = sf.keys()

      def process_key(k_name, f, sf_file, file_loaded_tensors):
        # [FUNCTION DOC]
        # HELPER: Worker for one checkpoint key: reads the tensor under a lock (the
        # safetensors handle is not thread-safe), maps the key (ValueError from the
        # mapper means unmapped: appended to skipped_keys and returned), applies the
        # optional permutation then reshape, converts to a jnp array cast to dtype if
        # needed, and stores it in file_loaded_tensors. Raises ValueError for a
        # duplicate mapped key, and wraps any exception in RuntimeError naming the
        # tensor and file. Side effects: mutates the shared dict and list.
        try:
          with file_lock:
            v = sf_file.get_tensor(k_name)  # get_tensor is not thread-safe
          try:
            jax_key_mapped, transform = torch_utils.torch_key_to_jax_key(
                key_map, k_name
            )
          except ValueError:
            skipped_keys.append(k_name)
            return

          if transform is not None:
            permute, reshape = transform
            if permute:
              v = v.transpose(permute)
            if reshape:
              v = v.reshape(reshape)

          current_arr = jnp.array(v)
          if dtype and current_arr.dtype != dtype:
            current_arr = current_arr.astype(dtype)

          if jax_key_mapped in file_loaded_tensors:
            raise ValueError(
                f'Duplicate key {jax_key_mapped} found within file {f.name}.'
            )
          file_loaded_tensors[jax_key_mapped] = current_arr

        except Exception as e:
          raise RuntimeError(
              f'Failed to load tensor {k_name} from file {f.name}: {e}'
          ) from e

      with concurrent.futures.ThreadPoolExecutor(
          max_workers=os.cpu_count()
      ) as executor:
        futures = [
            executor.submit(process_key, key, f, sf, file_loaded_tensors)
            for key in keys
        ]

      for future in concurrent.futures.as_completed(futures):
        if future.exception():
          raise future.exception()

    # Apply preprocessing if provided (e.g., for MoE expert stacking)
    if preprocess_fn is not None:
      file_loaded_tensors = preprocess_fn(file_loaded_tensors)

    if skipped_keys:
      logging.warning(
          'Skipped loading %d keys because they could not be mapped to model '
          'weights. This may be expected, for example when loading only text '
          'weights from a multimodal checkpoint. Missing keys: [%s]',
          len(skipped_keys),
          ', '.join(skipped_keys),
      )

    def make_update_tensor_fn(current_file_tensors):
      # [FUNCTION DOC]
      # HELPER: Returns an update_tensor closure bound to one file's loaded tensors.
      def update_tensor(path, param, shard=None):
        # [FUNCTION DOC]
        # HELPER: Tree-map callback for one state leaf. Computes the path key and
        # checks it and key + '.value' against the loaded tensors; on a hit, raises
        # ValueError on shape mismatch, else returns the array device_put onto the
        # given sharding (after device_get to host) or onto the first device. Returns
        # the original leaf when not found.
        current_path_key = path_to_key(path)

        # nnx.Param adds a .value suffix to the key
        possible_keys = [current_path_key, f'{current_path_key}.value']

        for k in possible_keys:
          if k in current_file_tensors:
            loaded_arr = current_file_tensors[k]
            if loaded_arr.shape != param.shape:
              raise ValueError(
                  f'Shape mismatch for {k}: got'
                  f' {loaded_arr.shape}, expected {param.shape}'
              )
            if shard is not None:
              # Ensure loaded_arr is a numpy array on host. device_put on a
              # device array will compile an expensive device-to-device
              # broadcast program instead of a simple host-to-device copy.
              loaded_arr = jax.device_get(loaded_arr)
              return jax.device_put(loaded_arr, shard)
            else:
              return jax.device_put(loaded_arr, jax.devices()[0])

        return param

      return update_tensor

    current_file_update_tensor = make_update_tensor_fn(file_loaded_tensors)

    if sharding_dict is not None:
      state_dict = jax.tree.map_with_path(
          current_file_update_tensor, state_dict, sharding_dict
      )
    else:
      state_dict = jax.tree.map_with_path(
          current_file_update_tensor, state_dict
      )

  return nnx.merge(graph_def, state_dict)


def load_and_create_model_opt(
    file_dir: str,
    model_class,
    config,
    key_mapping,
    mesh=None,
    preprocess_fn=None,
    dtype: jnp.dtype | None = None,
):
  # [FUNCTION DOC]
  # 'Optimized' loading strategy (Linux, local or downloaded files). Same
  # inputs as the original strategy. Steps: resolve gs:// paths; glob files
  # (ValueError if none); shape-evaluate the model under the mesh; split; pure
  # dict; shardings from the mesh, or every leaf mapped to the first device when
  # no mesh. Memory-map every file via load_safetensors_with_offsets, then for
  # every tensor map its name (unmapped names collected and warned about),
  # slice the flat view by element offset/count, reshape to its shape, apply the
  # optional permutation and reshape (still zero-copy where possible), store in
  # one global dict. Apply preprocess_fn once to the whole dict. Tree-map over
  # the sharding dict: look up path key or key + '.value', optionally cast to
  # dtype (via to_np_dtype) and device_put onto the leaf's sharding. Assert
  # every resulting leaf's shape equals the abstract shape. Close file handles
  # and merge into the model.
  # Returns the model. Raises: ValueError (no files), AssertionError (shape
  # mismatch).
  # KNOWN ISSUE: a parameter missing from the checkpoint becomes
  # device_put(None) and then fails in the shape assertion with an
  # AttributeError instead of a clear error; mmap objects are never explicitly
  # closed; the skipped-keys warning is emitted once per file with the cumulative
  # list; dtype values unsupported by to_np_dtype cast to None (float64).
  """Loads safetensors files and creates an NNX model.

  This version is optimized for linux file systems.

  Args:
    file_dir: Directory containing the safetensors files.
    model_class: The NNX model class to instantiate.
    config: The configuration object for the model.
    key_mapping: A function that takes the config and returns a mapping from
      torch keys to jax keys and optional transformations.
    mesh: An optional JAX device mesh for sharding.
    preprocess_fn: An optional function to preprocess the loaded state dict
      before sharding.
    dtype: Optional dtype to cast the loaded tensors to.

  Returns:
    An NNX model instance with weights loaded from the safetensors files.

  Raises:
    ValueError: If no safetensors files are found in the specified directory.
  """
  if file_dir.startswith('gs://'):
    file_dir = load_file_from_gcs(file_dir)

  files = list(epath.Path(file_dir).expanduser().glob('*.safetensors'))

  if not files:
    raise ValueError(f'No safetensors found in {file_dir}')

  # Create model structure
  context_manager = (
      compat.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()
  )

  with context_manager:
    model = nnx.eval_shape(lambda: model_class(config, rngs=nnx.Rngs(params=0)))

  graph_def, abs_state = nnx.split(model)
  nnx_state_dict = abs_state.to_pure_dict()

  if mesh is not None:
    sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
  else:
    device = jax.devices()[0]
    sharding_dict = jax.tree.map(lambda _: device, nnx_state_dict)

  key_map = key_mapping(config)

  arrays = []
  mmaps = []
  file_handles = []
  for f in files:
    contiguous_array, tensor_metadata, mm, fh = load_safetensors_with_offsets(f)
    arrays.append((contiguous_array, tensor_metadata))
    mmaps.append(mm)
    file_handles.append(fh)

  state_dict = {}
  skipped_keys = []
  for array, metadata_list in arrays:
    for metadata in metadata_list:
      try:
        jax_key_mapped, transform = torch_key_to_jax_key(
            key_map, metadata['name']
        )
      except ValueError:
        skipped_keys.append(metadata['name'])
        continue
      parameter = array[
          metadata['offset_elements'] : metadata['offset_elements']
          + metadata['size_elements']
      ].reshape(metadata['shape'])
      if transform is not None:
        permute, reshape = transform
        if permute:
          parameter = parameter.transpose(permute)
        if reshape:
          parameter = parameter.reshape(reshape)
      state_dict[jax_key_mapped] = parameter

    if skipped_keys:
      logging.warning(
          'Skipped loading %d keys because they could not be mapped to model '
          'weights. This may be expected, for example when loading only text '
          'weights from a multimodal checkpoint. Missing keys: [%s]',
          len(skipped_keys),
          ', '.join(skipped_keys),
      )

  if preprocess_fn is not None:
    state_dict = preprocess_fn(state_dict)

  def shard_state(state_dict):
    # [FUNCTION DOC]
    # HELPER: Returns a tree-map callback closed over the loaded state dict.
    def _shard_state(path, sharding):
      # [FUNCTION DOC]
      # HELPER: For a sharding leaf at path, finds the tensor by path key or key +
      # '.value', casts it to dtype if requested, and returns it device_put onto the
      # sharding (None if not found).
      key = path_to_key(path)
      possible_keys = [
          key,
          f'{key}.value',
      ]  # for nnx.Param where a value suffix will be there
      tensor = None
      for k in possible_keys:
        if k in state_dict:
          tensor = state_dict[k]
          break
      if tensor is not None and dtype is not None:
        np_dtype = to_np_dtype(dtype)
        tensor = tensor.astype(np_dtype)
      return jax.device_put(tensor, sharding)

    return _shard_state

  shard_function = shard_state(state_dict)
  state_dict = jax.tree.map_with_path(shard_function, sharding_dict)

  def _assert_shapes_match(path, x, y):
    # [FUNCTION DOC]
    # HELPER: Asserts that a loaded leaf x has the abstract leaf y's shape
    # (AssertionError with path otherwise); returns x.
    assert (
        x.shape == y.shape
    ), f'Shape mismatch for {path}: expected {y.shape}, got {x.shape}'
    return x

  jax.tree.map_with_path(_assert_shapes_match, state_dict, nnx_state_dict)

  for fh in file_handles:
    fh.close()

  return nnx.merge(graph_def, state_dict)


def load_and_create_model(
    file_dir: str,
    model_class,
    config,
    key_mapping,
    mesh=None,
    preprocess_fn=None,
    dtype: jnp.dtype | None = None,
    mode: str = 'auto',
):
  # [FUNCTION DOC]
  # Public entry point: loads safetensors from file_dir into a new model.
  # Inputs: file_dir (local or gs:// directory), model_class (called as
  # model_class(config, rngs=...)), config, key_mapping (config -> regex map),
  # mesh (optional), preprocess_fn (optional dict -> dict transform of loaded
  # tensors keyed by tunix keys), dtype (optional cast), mode ('auto',
  # 'optimized', 'original').
  # 'auto' picks 'original' in Google-internal environments or when Pathways is
  # initialized, else 'optimized'. Explicit 'optimized' in those environments
  # raises ValueError. 'original' outside them logs a recommendation to use
  # optimized. Any mode other than 'optimized' (including unknown strings) runs
  # the original strategy. Returns the loaded model.
  """Loads safetensors files and creates an NNX model.

  Args:
    file_dir: Directory containing the safetensors files.
    model_class: The NNX model class to instantiate.
    config: The configuration object for the model.
    key_mapping: A function that takes the config and returns a mapping from
      torch keys to jax keys and optional transformations.
    mesh: An optional JAX device mesh for sharding.
    preprocess_fn: An optional function to preprocess the loaded state dict
      before sharding.
    dtype: Optional dtype to cast the loaded tensors to.
    mode: The mode to use for loading the model. Options are ('auto',
      'optimized', 'original').

  Returns:
    An NNX model instance with weights loaded from the safetensors files.
  """
  if mode == 'auto':
    if env_utils.is_internal_env() or env_utils.is_pathways_initialized():
      mode = 'original'
    else:
      mode = 'optimized'

  # TODO(tunix-dev): Fix optimized mode when pathways is initialized.
  if mode == 'optimized':
    if env_utils.is_internal_env() or env_utils.is_pathways_initialized():
      raise ValueError(
          'Optimized mode is not supported in GOOGLE_INTERNAL_PACKAGE_PATH or pathways initialized.'
      )

  if mode == 'original':
    if (
        not env_utils.is_internal_env()
        and not env_utils.is_pathways_initialized()
    ):
      logging.warning('Optimized mode is faster and recommended if possible.')

  if mode == 'optimized':
    return load_and_create_model_opt(
        file_dir, model_class, config, key_mapping, mesh, preprocess_fn, dtype
    )
  else:
    return load_and_create_model_orig(
        file_dir, model_class, config, key_mapping, mesh, preprocess_fn, dtype
    )
