# [MODULE DOC]
# HELPER: Utilities for Tunix example/training scripts: dataset loading
# (GSM8K-style question/answer data) into Grain datasets, train/eval splitting,
# and a profiling + log capture context manager.
# Module state: DEBUG_LEVELS maps level names (DEBUG, INFO, WARNING, ERROR,
#   FATAL) to absl logging levels. At import, tries to import the internal xprof
#   session API and internal gfile; on success ENV is "g3", otherwise those names
#   are None and ENV is "oss", in which case tensorflow_datasets and fsspec are
#   imported (ImportError if missing).
"""Utilities for tunix scripts."""

from collections.abc import Callable
import json
import logging
import os
from absl import logging as absl_logging
import grain


DEBUG_LEVELS = {
    'DEBUG': absl_logging.DEBUG,
    'INFO': absl_logging.INFO,
    'WARNING': absl_logging.WARNING,
    'ERROR': absl_logging.ERROR,
    'FATAL': absl_logging.FATAL,
}

try:
  # This is a g3-only import.
  from GOOGLE_INTERNAL_PACKAGE_PATH.perftools.accelerators.xprof.api.python import xprof_session  # pytype: disable=import-error
  from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile  # pytype: disable=import-error

  ENV = 'g3'
except ImportError:
  xprof_session = None
  gfile = None
  ENV = 'oss'

if ENV == 'oss':
  import tensorflow_datasets as tfds
  import fsspec


def get_dataset(
    path: str,
    split: str,
    seed: int,
    system_prompt: str,
    *,
    answer_extractor: Callable[[str], str | None],
    dataset_name: str = 'gsm8k',
) -> grain.MapDataset:
  # [FUNCTION DOC]
  # Loads a question/answer dataset and maps it into prompt records.
  # Inputs: path - file path (g3: JSON file; OSS: "gs://" JSON file or a local
  #   directory used as TFDS data_dir); split - TFDS split name; seed - shuffle
  #   seed; system_prompt - string prepended to each question; answer_extractor
  #   (keyword-only) - function turning the raw answer text into the target
  #   answer (or None); dataset_name - TFDS dataset name (default "gsm8k").
  # Behavior: g3 reads and JSON-parses the file via gfile. OSS: "gs://" paths are
  #   read as JSON with fsspec; otherwise prints a download message, creates the
  #   directory if missing, and obtains a TFDS data source (array-record format,
  #   download enabled). Builds a Grain map dataset from the source, shuffles it
  #   with seed, and maps each record (having "question" and "answer" fields,
  #   bytes decoded as UTF-8 when not already str) to a dict with "prompts" and
  #   "question" both equal to system_prompt + question, and "answer" equal to
  #   answer_extractor(answer text).
  # Output: grain.MapDataset of those dicts (lazy).
  # Exceptions: I/O, JSON, TFDS errors propagate.
  # Side effects: possible network download, directory creation, stdout print.
  """Loads the dataset, from CNS in g3 or downloading in OSS."""
  if ENV == 'g3':
    with gfile.Open(path, 'rb') as f:
      data = json.loads(f.read())
  else:  # oss
    if path.startswith('gs://'):
      with fsspec.open(path, 'r') as f:
        data = json.load(f)
    else:
      print(
          f"Downloading {dataset_name.upper()} dataset ('{split}' split) to"
          f" '{path}'..."
      )
      if not os.path.exists(path):
        os.makedirs(path)
      # Using TFDS to download dataset.
      data = tfds.data_source(
          dataset_name,
          split=split,
          data_dir=path,
          builder_kwargs={'file_format': tfds.core.FileFormat.ARRAY_RECORD},
          download=True,
      )

  def _as_text(v):
    # [FUNCTION DOC]
    # HELPER: Returns v unchanged if it is a str, else decodes it as UTF-8.
    return v if isinstance(v, str) else v.decode('utf-8')

  dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=seed)
      .map(
          lambda x: {
              # passed to model forward pass
              'prompts': system_prompt + _as_text(x['question']),
              # passed to reward functions
              'question': system_prompt + _as_text(x['question']),
              # passed to reward functions
              'answer': answer_extractor(_as_text(x['answer'])),
          }
      )
  )
  return dataset


def get_train_and_eval_datasets(
    data_path: str,
    split: str,
    seed: int,
    system_prompt: str,
    batch_size: int,
    num_batches: int | None,
    train_fraction: float,
    num_epochs: int | None,
    *,
    answer_extractor: Callable[[str], str | None],
    dataset_name: str = 'gsm8k',
) -> tuple[grain.MapDataset, grain.MapDataset | None]:
  # [FUNCTION DOC]
  # Builds batched train and optional validation datasets.
  # Inputs: same loading arguments as get_dataset (data_path, split, seed,
  #   system_prompt, answer_extractor, dataset_name) plus batch_size,
  #   num_batches (optional cap on number of batches; falsy means no cap),
  #   train_fraction (fraction of batches used for training), num_epochs
  #   (optional repeat count; falsy means no repeat).
  # Behavior: load via get_dataset, batch by batch_size, truncate to
  #   num_batches. If train_fraction is exactly 1.0, everything is training data
  #   (repeated num_epochs times if set) and validation is None. Otherwise the
  #   first floor(len * train_fraction) batches are training and the rest
  #   validation, both repeated num_epochs times if set.
  # Output: tuple (train dataset, validation dataset or None).
  # Exceptions: propagated from loading. Side effects: those of get_dataset.
  """Loads and splits the dataset for training and evaluation."""
  dataset = get_dataset(
      data_path,
      split=split,
      seed=seed,
      system_prompt=system_prompt,
      answer_extractor=answer_extractor,
      dataset_name=dataset_name,
  ).batch(batch_size)

  if num_batches:
    dataset = dataset[:num_batches]

  if train_fraction == 1.0:
    train_dataset = dataset
    if num_epochs:
      train_dataset = train_dataset.repeat(num_epochs)
    val_dataset = None
  else:
    train_size = int(len(dataset) * train_fraction)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    if num_epochs:
      train_dataset = train_dataset.repeat(num_epochs)
      val_dataset = val_dataset.repeat(num_epochs)

  return train_dataset, val_dataset


class profile_and_capture_log:
  # [CLASS DOC]
  # Context manager that, in the g3 environment, optionally starts an xprof
  # profiling session and attaches a stream handler to the root Python logger
  # for the duration of the block; on exit it removes the handler and, if
  # profiling ran, ends the session and prints the xprof URL (retained one
  # year). In OSS it does nothing. Lower-case class name kept for use like a
  # function. Does not suppress exceptions.
  """A context manager for profiling and capturing logs.

  This is a no-op in OSS.

  Args:
    tag: A tag for the xprof session.
    enable_profile: Whether to enable profiling.
    device_name: The device name for xprof.
    host_trace_level: The host trace level for xprof.
  """

  def __init__(
      self,
      tag: str = '',
      enable_profile: bool = True,
      device_name: str = 'viperfish',
      host_trace_level: int = 2,
  ):
    # [FUNCTION DOC]
    # Stores configuration: tag (xprof session tag), enable_profile flag,
    # device_name (default "viperfish"), host_trace_level (default 2). Also
    # initializes the profiler session and log handler references to None.
    # No side effects beyond attribute assignment.
    self._tag = tag
    self._enable_profile = enable_profile
    self._device_name = device_name
    self._host_trace_level = host_trace_level
    self._xprof = None
    self._log_handler = None

  def __enter__(self):
    # [FUNCTION DOC]
    # Enter: only when ENV is "g3" and the xprof API is available. If profiling
    # is enabled, create an xprof session and start it with the configured
    # device name, python tracer enabled and the configured host trace level.
    # Then always (in g3) create a stream handler and add it to the root logger.
    # Output: None (the "as" target is None). Side effects: starts profiling,
    # modifies global logging handlers.
    if ENV == 'g3' and xprof_session is not None:
      if self._enable_profile:
        self._xprof = xprof_session.XprofSession()
        self._xprof.start_session(
            device_name=self._device_name,
            enable_python_tracer=True,
            host_trace_level=self._host_trace_level,
        )
      self._log_handler = logging.StreamHandler()
      logging.root.addHandler(self._log_handler)

  def __exit__(self, exc_type, exc_value, traceback):
    # [FUNCTION DOC]
    # Exit: inputs are the standard exception triple (ignored). Removes the log
    # handler if one was added; if a profiling session exists, ends it with the
    # stored tag and a TTL of 365 days and prints the returned URL.
    # Output: None (exceptions are not suppressed).
    if self._log_handler:
      logging.root.removeHandler(self._log_handler)
    if self._xprof:
      xprof_url = self._xprof.end_session_and_get_url(
          tag=self._tag, ttl_seconds=60 * 60 * 24 * 365
      )
      print(xprof_url)
