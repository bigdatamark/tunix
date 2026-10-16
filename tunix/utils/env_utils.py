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
# HELPER: Small utility module with environment detection and setup helpers.
# Exports three functions (sharding env setup, internal-env detection, Pathways
# detection) and one constant, SGLANG_JAX_TP_AXIS_NAME, read once at import
# time from the environment variable of the same name, defaulting to the string
# "tensor"; it names the mesh axis used for tensor parallelism by SGLang-JAX.
"""Environment utils."""

import os

import flax


def setup_sharding_environment():
  # [FUNCTION DOC]
  # Disables Flax's "always shard variable" behavior when the installed flax
  # config exposes the flax_always_shard_variable option (it is set to False);
  # otherwise does nothing.
  # Inputs: none. Output: None. Exceptions: none expected.
  # Side effects: mutates the global flax configuration.
  """Sets up the sharding environment."""
  if hasattr(flax.config, 'flax_always_shard_variable'):
    flax.config.update('flax_always_shard_variable', False)


def is_internal_env():
  # [FUNCTION DOC]
  # Returns True if an internal-only package (the gfile module from the internal
  # pyglib package path placeholder) can be imported, else False when the import
  # raises ImportError. Inputs: none. Output: bool.
  # Side effects: attempts an import (may load that module).
  # KNOWN ISSUE: the import path is a literal placeholder
  # (GOOGLE_INTERNAL_PACKAGE_PATH) that is presumably rewritten when exported;
  # in the open-source build it always fails, so this always returns False.
  """Checks if the code is running within the internal environment."""
  try:
    from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile  # noqa: F401

    return True
  except ImportError:
    return False


def is_pathways_initialized():
  # [FUNCTION DOC]
  # Returns whether the Pathways backend is in use: tries to import the
  # pathwaysutils package and returns the result of its
  # is_pathways_backend_used() query; returns False if the import fails.
  # Inputs: none. Output: bool. Exceptions: any non-ImportError exception from
  # the query propagates. Side effects: may import pathwaysutils.
  """Checks if Pathways is initialized."""
  try:
    import pathwaysutils  # noqa: F401
    return pathwaysutils.is_pathways_backend_used()
  except ImportError:
    return False

SGLANG_JAX_TP_AXIS_NAME = os.getenv('SGLANG_JAX_TP_AXIS_NAME', 'tensor')
