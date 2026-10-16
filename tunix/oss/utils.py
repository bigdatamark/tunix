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
# HELPER: utilities for the open-source build: detecting Pathways availability, copying files
# from Google Cloud Storage to a local directory, and downloading model checkpoints from Kaggle
# or the HuggingFace Hub. No side effects at import time beyond importing fsspec and huggingface_hub.

"""Utils for OSS code."""

import os
from typing import Any

from absl import logging
import fsspec
import huggingface_hub as hf


def pathways_available() -> bool:
  # [FUNCTION DOC]
  # Returns whether Pathways (proxy JAX backend) is usable. Returns False if the JAX_PLATFORMS
  # environment variable does not contain the substring "proxy"; otherwise tries to import the
  # pathwaysutils package and returns True on success, False on ImportError.
  # Inputs: none. Output: boolean. Side effects: may import pathwaysutils.
  if 'proxy' not in os.getenv('JAX_PLATFORMS', ''):
    return False
  try:
    import pathwaysutils  # pylint: disable=g-import-not-at-top, unused-import # pytype: disable=import-error

    return True
  except ImportError:
    return False


def load_file_from_gcs(file_dir: str, target_dir: str | None = None) -> str:
  # [FUNCTION DOC]
  # Copies a GCS file or directory to the local filesystem.
  # Inputs: file_dir - a "gs://bucket/prefix" path; target_dir - local base directory (defaults to
  # the system temp directory).
  # Logic: raises ValueError if the path does not start with "gs://". Splits the remainder at the
  # first slash into bucket and prefix (a path with no slash after the bucket raises ValueError from
  # unpacking). Local destination is target_dir joined with the prefix. Uses the fsspec "gs"
  # filesystem to recursively download to that destination.
  # Output: the local destination path string. Exceptions: ValueError as above; ImportError is
  # re-raised with a message asking to install google-cloud-storage; fsspec errors propagate.
  # Side effects: network download and local file writes.
  """Load file from GCS."""
  if not file_dir.startswith('gs://'):
    raise ValueError(f'Invalid GCS path: {file_dir}')

  _, prefix = file_dir[5:].split('/', 1)
  try:
    import tempfile  # pylint: disable=g-import-not-at-top

    if target_dir is None:
      target_dir = tempfile.gettempdir()
    local_dir = os.path.join(target_dir, prefix)

    fsspec_fs = fsspec.filesystem('gs')
    fsspec_fs.get(file_dir, local_dir, recursive=True)

    return local_dir
  except ImportError as e:
    raise ImportError(
        'Please install google-cloud-storage to load model from GCS.'
    ) from e


def kaggle_pipeline(model_id: str, model_download_path: str):
  # [FUNCTION DOC]
  # Downloads a model from Kaggle via kagglehub.
  # Inputs: model_id - Kaggle model handle; model_download_path - cache directory.
  # Logic: best-effort compatibility shim: if kagglesdk's kaggle_env module lacks get_web_endpoint
  # but has get_endpoint, aliases the former to the latter (any exception is swallowed). Imports
  # kagglehub; if KAGGLE_USERNAME or KAGGLE_KEY is missing from the environment, triggers an
  # interactive kagglehub login. Sets the KAGGLEHUB_CACHE environment variable to
  # model_download_path and downloads the model.
  # Output: the local path returned by kagglehub. Exceptions: ImportError if kagglehub is absent;
  # download/auth errors propagate. Side effects: monkey-patches kagglesdk, mutates os.environ,
  # network download, possible interactive login.
  """Download model from Kaggle."""
  try:
    import kagglesdk.kaggle_env  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
    if not hasattr(kagglesdk.kaggle_env, 'get_web_endpoint') and hasattr(kagglesdk.kaggle_env, 'get_endpoint'):
      kagglesdk.kaggle_env.get_web_endpoint = kagglesdk.kaggle_env.get_endpoint
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  import kagglehub  # pylint: disable=g-import-not-at-top

  if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
    kagglehub.login()
  os.environ['KAGGLEHUB_CACHE'] = model_download_path
  return kagglehub.model_download(model_id)


def hf_pipeline(model_id: str, model_download_path: str):
  # [FUNCTION DOC]
  # Downloads all files of a HuggingFace Hub repo except those under "original/".
  # Inputs: model_id - repo id; model_download_path - local directory.
  # Logic: if HF_TOKEN is not in the environment, calls the interactive HF login. Lists all repo
  # files, filters out paths beginning with "original/", downloads each to model_download_path one
  # by one, then logs the file list and destination at INFO.
  # Output: model_download_path. Exceptions: hub/network errors propagate. Side effects: network
  # download, file writes, possible interactive login, logging.
  """Download model from HuggingFace."""
  if 'HF_TOKEN' not in os.environ:
    hf.login()
  all_files = hf.list_repo_files(model_id)
  filtered_files = [f for f in all_files if not f.startswith('original/')]
  for filename in filtered_files:
    hf.hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=model_download_path,
    )
  logging.info(
      'Downloaded %s to: %s',
      filtered_files,
      model_download_path,
  )
  return model_download_path
