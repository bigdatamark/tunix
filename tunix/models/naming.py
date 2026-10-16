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
# Model naming utilities. Parses a model identifier, which may be a Hugging Face
# model id (e.g. "meta-llama/Llama-3.1-8B", recognized by containing '-' or '.')
# or a tunix ConfigId (e.g. "llama3p1_8b", recognized by containing '_' and no
# '-' or '.'), into a standardized family, version, config category (the
# subpackage under tunix/models holding the ModelConfig) and config id
# ("<family>_<version>"). Two prefix tables (one for HF names such as 'llama-3.1-',
# one for config ids such as 'llama3p1_') map a raw family prefix to a
# _ModelFamilyInfo (standardized family + config category); the longest matching
# prefix wins. Defines NewType aliases HFModelId and ConfigId (both str).

"""Model naming utilities.

This module provides utility functions to parse and handle model names and
convert them to internal model naming structures.
"""


import dataclasses
from typing import NewType
import immutabledict


HFModelId = NewType('HFModelId', str)
ConfigId = NewType('ConfigId', str)


def _is_hf_model_id_type(model_id_or_name: str) -> bool:
  # [FUNCTION DOC]
  # HELPER: Returns True if the string contains '-' or '.', i.e. looks like a
  # Hugging Face model id/name. No exceptions or side effects.
  return '-' in model_id_or_name or '.' in model_id_or_name


def _is_config_id_type(model_id_or_name: str) -> bool:
  # [FUNCTION DOC]
  # HELPER: Returns True if the string is not HF-style (no '-' or '.') and contains
  # '_', i.e. looks like a ConfigId. No exceptions or side effects.
  return not _is_hf_model_id_type(model_id_or_name) and '_' in model_id_or_name


@dataclasses.dataclass(frozen=True)
class ModelNaming:
  # [CLASS DOC]
  # Frozen dataclass holding all naming facets of one model. Fields (all optional
  # on construction): model_id, model_name, model_family, model_version,
  # model_config_category, model_config_id. After construction every field except
  # possibly model_id is filled in by derivation from the model name; any
  # explicitly supplied field must equal the derived value.
  """Model naming information.

  Attributes:
    model_id: A unique identifier for the model, which can be either a
      Huggingface model ID (e.g., "meta-llama/Llama-3.1-8B") or a standardized
      ConfigId (e.g., "llama3p1_8b").
    model_name: The unique full name identifier of the model. This should be the
      full name and should match exactly with the model name used in Hugging
      Face. e.g., "gemma-2b","llama-3.1-8b". The model name is all lowercase and
      typically formatted as <model-family>-<model-version>.
    model_family: The standardized model family, e.g., "gemma", "gemma2", or
      "qwen2p5". The model is standerdardized by removing unnecessary '-', e.g.,
      "gemma-2" --> "gemma2", replacing '-' with '_',  and replacing '.' with
      'p',  e.g., "qwen2.5" would be standardized to "qwen2p5".
    model_version: The standardized version of this model family. This would be
      the second portion of the model name. It includes additional information
      such as size, whether it is instruction tuned ("it'), etc. The model
      version is standardized by lowercasing, replacing '-' with '_', and
      replacing '.' with 'p'. e.g., "2b-it" would be standardized to "2b_it".
    model_config_category: The model config category is the python class name of
      the ModelConfig class. e.g., both gemma and gemma2 models have the
      category "gemma" with the ModelConfig class being defined under
      gemma/model.py.
    model_config_id: The standardized config id composed of the model family and
      model version, used in the ModelConfig class. e.g., "gemma_2b_it" or
      "qwen2p5_0p5b".
  """
  # TODO(b/451662153): use HFModelId and ConfigId throughout, add validation,
  # and then remove str support.
  model_id: HFModelId | ConfigId | str | None = None
  model_name: str | None = None
  model_family: str | None = None
  model_version: str | None = None
  model_config_category: str | None = None
  model_config_id: str | None = None

  def __post_init__(self):
    # [FUNCTION DOC]
    # Validates and fills derived fields (uses object.__setattr__ because the
    # dataclass is frozen). If model_id is set, derives model_name from it via
    # get_model_name_from_model_id and raises ValueError if a provided model_name
    # differs; otherwise uses model_name. Raises ValueError if neither yields a
    # non-empty name. Then derives family and version (get_model_family_and_version),
    # config category (get_model_config_category) and config id
    # (get_model_config_id); for each, raises ValueError on mismatch with a
    # non-None provided value, else stores the derived value.
    # Side effects: mutates the instance fields. Also propagates ValueError from the
    # parsing helpers for unknown families or invalid formats.
    if self.model_id:
      # We infer model_name from model_id.
      model_name = get_model_name_from_model_id(self.model_id)
      if self.model_name and self.model_name != model_name:
        raise ValueError(
            'model_name set in ModelNaming and one inferred from model_id do'
            f' not match. model_name: {self.model_name} and model_id:'
            f' {self.model_id}, model_name inferred from model_id:'
            f' {model_name}.'
        )
    else:
      # If no model_id is provided, we use the model_name.
      model_name = self.model_name
    if not model_name:
      raise ValueError('Either model_name or model_id must be provided.')
    object.__setattr__(self, 'model_name', model_name)

    family, version = get_model_family_and_version(model_name)
    if self.model_family is not None and self.model_family != family:
      raise ValueError(
          f'model_family mismatch: {self.model_family} != {family}'
      )
    object.__setattr__(self, 'model_family', family)

    if self.model_version is not None and self.model_version != version:
      raise ValueError(
          f'model_version mismatch: {self.model_version} != {version}'
      )
    object.__setattr__(self, 'model_version', version)

    category = get_model_config_category(model_name)
    if (
        self.model_config_category is not None
        and self.model_config_category != category
    ):
      raise ValueError(
          f'model_config_category mismatch: {self.model_config_category} !='
          f' {category}'
      )
    object.__setattr__(self, 'model_config_category', category)

    config_id = get_model_config_id(model_name)
    if self.model_config_id is not None and self.model_config_id != config_id:
      raise ValueError(
          f'model_config_id mismatch: {self.model_config_id} != {config_id}'
      )
    object.__setattr__(self, 'model_config_id', config_id)


@dataclasses.dataclass(frozen=True)
class _ModelFamilyInfo:
  # [CLASS DOC]
  # HELPER: Frozen dataclass with two str fields: family (standardized family name
  # used in config ids) and config_category (tunix model subpackage name).
  """Configuration for handling model family mappings."""

  family: str  # standardized model family, as used in id in ModelConfig
  config_category: str  # category in the path to the ModelConfig class


# HF model family info mapping.
_HF_MODEL_FAMILY_INFO_MAPPING = immutabledict.immutabledict({
    'gemma-': _ModelFamilyInfo(family='gemma', config_category='gemma'),
    'gemma-1.1-': _ModelFamilyInfo(family='gemma1p1', config_category='gemma'),
    'gemma-2-': _ModelFamilyInfo(family='gemma2', config_category='gemma'),
    'gemma-3-': _ModelFamilyInfo(family='gemma3', config_category='gemma3'),
    'gemma-4-': _ModelFamilyInfo(family='gemma4', config_category='gemma4'),
    'llama-3-': _ModelFamilyInfo(family='llama3', config_category='llama3'),
    'llama-3.1-': _ModelFamilyInfo(family='llama3p1', config_category='llama3'),
    'llama-3.2-': _ModelFamilyInfo(family='llama3p2', config_category='llama3'),
    'qwen2.5-': _ModelFamilyInfo(family='qwen2p5', config_category='qwen2'),
    'qwen3-': _ModelFamilyInfo(family='qwen3', config_category='qwen3'),
    'deepseek-r1-distill-qwen-': _ModelFamilyInfo(
        family='deepseek_r1_distill_qwen', config_category='qwen2'
    ),
})

# Config id model family info mapping.
_CONFIG_ID_MODEL_FAMILY_INFO_MAPPING = immutabledict.immutabledict({
    'gemma_': _ModelFamilyInfo(family='gemma', config_category='gemma'),
    'gemma1p1_': _ModelFamilyInfo(family='gemma1p1', config_category='gemma'),
    'gemma2_': _ModelFamilyInfo(family='gemma2', config_category='gemma'),
    'gemma3_': _ModelFamilyInfo(family='gemma3', config_category='gemma3'),
    'gemma4_': _ModelFamilyInfo(family='gemma4', config_category='gemma4'),
    'llama3_': _ModelFamilyInfo(family='llama3', config_category='llama3'),
    'llama3p1_': _ModelFamilyInfo(family='llama3p1', config_category='llama3'),
    'llama3p2_': _ModelFamilyInfo(family='llama3p2', config_category='llama3'),
    'qwen2p5_': _ModelFamilyInfo(family='qwen2p5', config_category='qwen2'),
    'qwen3_': _ModelFamilyInfo(family='qwen3', config_category='qwen3'),
    'deepseek_r1_distill_qwen_': _ModelFamilyInfo(
        family='deepseek_r1_distill_qwen', config_category='qwen2'
    ),
})


def _get_model_family_mapping(
    model_name: str,
) -> immutabledict.immutabledict[str, _ModelFamilyInfo]:
  # [FUNCTION DOC]
  # HELPER: Chooses the prefix table for a name: HF table when the name is HF-style,
  # config-id table when it is config-id-style. Raises ValueError otherwise (e.g.
  # a name with neither '-', '.', nor '_'). Returns the immutable mapping.
  """Returns the model family mapping based on the model name format."""
  if _is_hf_model_id_type(model_name):
    return _HF_MODEL_FAMILY_INFO_MAPPING
  elif _is_config_id_type(model_name):
    return _CONFIG_ID_MODEL_FAMILY_INFO_MAPPING
  else:
    raise ValueError(
        f'Invalid model ID format: {model_name!r}. Expected a Huggingface'
        ' model ID or a ConfigId.'
    )


def split(model_name: str) -> tuple[str, str]:
  # [FUNCTION DOC]
  # Splits a model name into (raw family prefix, raw version). Lowercases the
  # name, selects the prefix table, and finds the longest table key that is a
  # prefix of the name. Returns that key (still including its trailing '-' or '_')
  # and the remainder with leading '-' stripped. Raises ValueError if no prefix
  # matches or if the name format is invalid. No side effects.
  # KNOWN ISSUE: the docstring calls the first element the model family, but it is
  # the raw table key including the trailing separator (e.g. 'llama-3.1-').
  # For config ids the remainder is not stripped of a leading '_' (prefix keys end
  # with '_' so this is normally harmless).
  """Splits model name into model family and model version.

  Find the longest matching prefix of the model name in the
  model family info mapping. Returns the remaining string as the model version,
  stripping leading hyphens.

  Args:
    model_name: The model name, e.g., llama-3.1-8b.

  Returns:
    A tuple containing the un-standardized model_family and model_version.
  """
  model_name = model_name.lower()
  mapping = _get_model_family_mapping(model_name)
  matched_family = ''
  for family in mapping:
    if model_name.startswith(family) and len(family) > len(matched_family):
      matched_family = family
  if matched_family:
    return matched_family, model_name[len(matched_family) :].lstrip('-')
  else:
    raise ValueError(
        f'Could not determine model family for: {model_name}. Not one of the'
        ' known families:'
        f' {list(mapping.keys())}'
    )


def _standardize_model_version(raw_model_version: str) -> str:
  # [FUNCTION DOC]
  # HELPER: Standardizes a raw version string: returns '' for empty input;
  # otherwise lowercases, replaces '-' with '_' and '.' with 'p'. Raises
  # ValueError if the result is longer than one character and its first character
  # is not alphanumeric. Returns the standardized version.
  # KNOWN ISSUE: the docstring says dots are replaced by underscores, but the code
  # replaces them with 'p'. Single-character non-alphanumeric versions are not
  # rejected.
  """Standardizes model version name.

  Operations include:
  - Lowercase
  - Replace hyphens with underscores
  - Replace dots with underscores
  - Validate the model version starts with an alphanumeric character.

  Args:
    raw_model_version: The raw model version string.

  Returns:
    The standardized model version name.
  """
  if not raw_model_version:
    return ''
  model_version = raw_model_version.lower().replace('-', '_').replace('.', 'p')

  # Validate the model version starts with an alphanumeric character.
  if len(model_version) > 1 and not model_version[0].isalnum():
    raise ValueError(
        'Invalid model version format. Expected alphanumeric starting'
        f' character, found: {model_version}'
    )
  return model_version


def get_model_family_and_version(model_name: str) -> tuple[str, str]:
  # [FUNCTION DOC]
  # Returns (standardized family, standardized version) for a model name: splits
  # it, looks up the raw prefix in the matching table to get the family, and
  # standardizes the version. Raises ValueError from split/standardization.
  """Splits model name into internal, standardized model family and model version."""
  raw_model_family, raw_model_version = split(model_name)
  mapping = _get_model_family_mapping(model_name)
  model_family = mapping[raw_model_family].family
  model_version = _standardize_model_version(raw_model_version)
  return model_family, model_version


def get_model_config_category(model_name: str) -> str:
  # [FUNCTION DOC]
  # Returns the config category (tunix model subpackage, e.g. 'qwen2') for a
  # model name by splitting it and looking up the raw prefix. Raises ValueError
  # from split.
  """Returns the model config category from the model family."""
  raw_model_family, _ = split(model_name)
  mapping = _get_model_family_mapping(model_name)
  return mapping[raw_model_family].config_category


def get_model_config_id(model_name: str) -> str:
  # [FUNCTION DOC]
  # Returns the config id: "<family>_<version>" with any remaining '.' replaced by
  # 'p' and '-' by '_'. Raises ValueError from parsing.
  # KNOWN ISSUE: if the version is empty the result ends in a bare '_'.
  """Returns the model config ID from the model name."""
  model_family, model_version = get_model_family_and_version(model_name)
  config_id = f'{model_family}_{model_version}'
  config_id = config_id.replace('.', 'p').replace('-', '_')
  return config_id


def get_model_name_from_model_id(model_id: HFModelId | ConfigId | str) -> str:
  # [FUNCTION DOC]
  # Derives a model name from a model id. If the id is HF-style or contains '/',
  # takes the last '/'-separated component, lowercases it, raises ValueError if it
  # is empty, and rewrites a leading 'meta-llama-' to 'llama-'. If config-id-style,
  # returns it lowercased. Otherwise returns it lowercased as-is (assumed to
  # already be a name). No side effects.
  """Extracts model name from model ID by taking the last part of path.

  Args:
    model_id: The full model name identifier, as it appears on huggingface,
      including the parent directory. E.g., meta-llama/Llama-3.1-8B. Can also be
      the model_config_id directly, e.g., llama3p1_8b.

  Returns:
    The model_name string.
  """
  if _is_hf_model_id_type(model_id) or '/' in model_id:
    model_name = model_id.split('/')[-1].lower()
    if not model_name:
      raise ValueError(
          f'Invalid model ID format: {model_id!r}. Model name cannot be empty.'
      )
    if model_name.startswith('meta-llama-'):
      model_name = model_name.replace('meta-llama-', 'llama-', 1)
    return model_name
  elif _is_config_id_type(model_id):
    return model_id.lower()
  else:
    # If the model_id is not a HFModelId or ConfigId, we assume it is already
    # a model_name and convert it to lowercase to be consistent.
    return model_id.lower()
