# Copyright 2025 Google LLC
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
# HELPER: model/tokenizer construction utilities for the Tunix CLI. Loads models through the
# AutoModel factory from a configured source (e.g. GCS, Kaggle, HuggingFace), optionally wraps
# them with LoRA/QLoRA via qwix, resolves the tokenizer path for some model families, and builds
# the tokenizer adapter. Module constant: the default tokenizer path from the base config
# ("meta-llama/Llama-3.1-8B"), used to detect "not overridden".

"""Utilities for creating and managing models in Tunix CLI."""

import os
from typing import Any, Tuple

from absl import logging
from flax import nnx
import jax
import qwix
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models import automodel
from tunix.models import naming
from tunix.rl import reshard

_DEFAULT_TOKENIZER_PATH = 'meta-llama/Llama-3.1-8B'


def apply_lora_to_model(base_model, mesh, lora_config):
  # [FUNCTION DOC]
  # Wraps a base model with LoRA (or QLoRA) adapters.
  # Inputs: base_model - an nnx model that provides a method returning example model inputs;
  # mesh - JAX mesh or None; lora_config - mapping with required module_path, rank, alpha and
  # optional tile_size and weight_qtype.
  # Logic: logs the config; builds LoraProvider kwargs from the required keys plus tile_size and
  # weight_qtype when present (logging "Qlora is applied" when weight_qtype exists, else "Lora is
  # applied"). Creates the qwix LoraProvider (on TypeError logs the error and re-raises). Applies
  # LoRA using the model's example inputs as keyword arguments. If mesh is given, reshards the
  # LoRA model onto the mesh.
  # Output: the LoRA-wrapped model. Exceptions: KeyError for missing required keys, TypeError from
  # the provider, qwix errors. Side effects: logging, device memory allocation.
  """Apply Lora to the base model if given lora config."""
  logging.info('lora_config %r', lora_config)
  # Basic keyword arguments for LoraProvider
  lora_kwargs = {
      'module_path': lora_config['module_path'],
      'rank': lora_config['rank'],
      'alpha': lora_config['alpha'],
  }
  has_tile_size = 'tile_size' in lora_config
  has_weight_qtype = 'weight_qtype' in lora_config
  if has_tile_size:
    lora_kwargs['tile_size'] = lora_config['tile_size']
  if has_weight_qtype:
    lora_kwargs['weight_qtype'] = lora_config['weight_qtype']
    logging.info('Qlora is applied')
  else:
    logging.info('Lora is applied')

  try:
    lora_provider = qwix.LoraProvider(**lora_kwargs)
  except TypeError as e:
    logging.error(
        'Error initializing qwix.LoraProvider: %s. Kwargs: %s', e, lora_kwargs
    )
    # Depending on desired behavior, you might re-raise or return base_model
    raise

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )
  if mesh is not None:
    lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)
  return lora_model


def create_tokenizer(tokenizer_config, tokenizer_path: str | None):
  # [FUNCTION DOC]
  # Builds the tokenizer adapter. Inputs: tokenizer_config mapping (tokenizer_path,
  # tokenizer_type, add_bos, add_eos) and an optional tokenizer_path override (used when truthy,
  # otherwise the config's path). Output: Tokenizer constructed positionally with type, path,
  # add_bos, add_eos and the HF_TOKEN environment variable (or None). Exceptions: KeyError for
  # missing config keys, tokenizer loading errors. Side effects: may download tokenizer files.
  if not tokenizer_path:
    tokenizer_path = tokenizer_config['tokenizer_path']
  tokenizer_type, add_bos, add_eos = (
      tokenizer_config['tokenizer_type'],
      tokenizer_config['add_bos'],
      tokenizer_config['add_eos'],
  )

  return tokenizer_lib.Tokenizer(
      tokenizer_type,
      tokenizer_path,
      add_bos,
      add_eos,
      os.environ.get('HF_TOKEN'),
  )


def create_model(
    model_config: dict[str, Any],
    tokenizer_config: dict[str, Any],
    mesh: jax.sharding.Mesh,
) -> Tuple[nnx.Module, str]:
  # [FUNCTION DOC]
  # Creates a model from config and resolves the tokenizer path.
  # Inputs: model_config (model_source, model_id, model_name, model_display required; optional
  # model_download_path, intermediate_ckpt_dir, rng_seed default 0, model_path,
  # use_flash_attention default False, flash_attention_block_size default 1024, remat_config
  # default 1, lora_config), tokenizer_config (tokenizer_path), mesh.
  # Logic: converts model_source to the ModelSource enum, raising ValueError listing valid sources
  # if unknown. Loads model and its local path via AutoModel from_pretrained with the above options.
  # Determines the model family from model_name: for gemma3 from "gcs", if the tokenizer path is
  # empty or the default, uses the public Gemma3 tokenizer GCS path; for gemma/gemma1p1/gemma2 from
  # "kaggle", under the same condition uses "tokenizer.model" inside the model path. If lora_config
  # is truthy applies LoRA, otherwise logs full-weight training. If model_display is truthy, prints
  # the model with nnx display.
  # Output: tuple of (model, tokenizer_path). Exceptions: ValueError, KeyError, loading errors.
  # Side effects: downloads, device allocation, logging/printing.
  """Creates a model and determines the tokenizer path based on the model config.

  This function handles model loading from various sources (GCS, Kaggle, HF)
  and applies LoRA if specified in the config.

  Args:
      model_config: A dictionary containing model configuration, including
        'model_name', 'model_source', 'model_id', 'model_download_path',
        'intermediate_ckpt_dir', and optionally 'lora_config'.
      tokenizer_config: A dictionary containing tokenizer configuration,
        including 'tokenizer_path'.
      mesh: The JAX sharding Mesh object.

  Returns:
      A tuple containing:
          - model: The loaded and potentially LoRA-applied nnx.Module.
          - tokenizer_path: The determined path to the tokenizer model.
  """
  tokenizer_path: str = tokenizer_config['tokenizer_path']
  model_source_str = model_config['model_source']

  # Create Model
  try:
    model_source = automodel.ModelSource(model_source_str)
  except ValueError as exc:
    raise ValueError(
        f'Unsupported model source: {model_source_str}. '
        f'Available sources: {[s.value for s in automodel.ModelSource]}'
    ) from exc

  model, model_path = automodel.AutoModel.from_pretrained(
      model_id=model_config['model_id'],
      mesh=mesh,
      model_source=model_source,
      model_download_path=model_config.get('model_download_path'),
      intermediate_ckpt_dir=model_config.get('intermediate_ckpt_dir'),
      rng_seed=model_config.get('rng_seed', 0),
      model_path=model_config.get('model_path'),
      use_flash_attention=model_config.get('use_flash_attention', False),
      flash_attention_block_size=model_config.get(
          'flash_attention_block_size', 1024
      ),
      remat_config=model_config.get('remat_config', 1),
  )

  # Handle Tokenizer Path overrides
  model_family = naming.ModelNaming(
      model_name=model_config['model_name']
  ).model_family
  if model_family == 'gemma3' and model_source_str == 'gcs':
    # Use the provided tokenizer path, unless it is the default from base_config
    if not tokenizer_path or tokenizer_path == _DEFAULT_TOKENIZER_PATH:
      tokenizer_path = 'gs://gemma-data/tokenizers/tokenizer_gemma3.model'
  elif (
      model_family in ('gemma', 'gemma1p1', 'gemma2')
      and model_source_str == 'kaggle'
  ):
    # Use the provided tokenizer path, unless it is the default from base_config
    if not tokenizer_path or tokenizer_path == _DEFAULT_TOKENIZER_PATH:
      tokenizer_path = os.path.join(model_path, 'tokenizer.model')

  if model_config.get('lora_config'):
    # Apply Lora to model if given lora config
    model = apply_lora_to_model(model, mesh, model_config['lora_config'])
  else:
    logging.info('Training with Full Weight')

  if model_config['model_display']:
    nnx.display(model)

  return model, tokenizer_path
