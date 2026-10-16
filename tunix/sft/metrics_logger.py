# [MODULE DOC]
# Metrics logging for Tunix trainers built on metrax logging backends.
# Module aliases: LoggingBackend, TensorboardBackend, WandbBackend,
#   CluBackend (None if metrax lacks it), BackendFactory (zero-arg callable
#   returning a LoggingBackend).
# Defines MetricsLoggerOptions (backend config/factory), Mode (train/eval
#   enum), _calculate_geometric_mean, and MetricsLogger which keeps an
#   in-memory history keyed by prefix -> mode -> metric name and forwards
#   every value to jax.monitoring.record_scalar, to which backends are
#   subscribed as scalar listeners. No license header. No side effects at
#   import.
"""Metric logger with a unified, protocol-based backend system."""

import collections
import dataclasses
import enum
from typing import Any, Callable

from absl import logging
import jax
from metrax import logging as metrax_logging
import numpy as np
from tunix.utils import env_utils

LoggingBackend = metrax_logging.LoggingBackend
TensorboardBackend = metrax_logging.TensorboardBackend
WandbBackend = metrax_logging.WandbBackend
CluBackend = getattr(metrax_logging, "CluBackend", None)

# User backends MUST be factories (callables) to keep Options pure and copyable.
BackendFactory = Callable[[], LoggingBackend]


@dataclasses.dataclass
class MetricsLoggerOptions:
  # [CLASS DOC]
  # Mutable dataclass. Fields: log_dir, project_name ("tunix"), run_name
  # (""), flush_every_n_steps (100), backend_kwargs (dict keyed by backend
  # name "clu"/"tensorboard"/"wandb" -> constructor kwargs, or
  # "custom_backend" -> list of BackendFactory). Backends are created
  # lazily via create_backends so the options object stays copyable.
  """Metrics Logger options."""

  log_dir: str
  project_name: str = "tunix"
  run_name: str = ""
  flush_every_n_steps: int = 100
  # Keyword arguments for backend initialization. The key is the backend name
  # (e.g., 'wandb', 'clu', 'tensorboard' or 'custom_backend' which uses custom
  # LoggingBackend factories) and the value is a dictionary of
  # keyword arguments to be passed to the backend's constructor.
  # For example:
  # backend_kwargs={
  #   'wandb': {
  #     'resume': 'must',
  #     'id': '12345',
  #     'project': 'my-project',
  #     'name': 'my-run',
  #   },
  #   'tensorboard': {
  #      'log_dir': '/path/to/log',
  #      'flush_every_n_steps': 100,
  #   }
  # }
  backend_kwargs: dict[str, dict[str, Any] | list[BackendFactory]] = (
      dataclasses.field(default_factory=dict)
  )

  def create_backends(self) -> list[LoggingBackend]:
    # [FUNCTION DOC]
    # Creates live backends. Output: list of LoggingBackend.
    # Steps: non-zero process index -> empty list. If backend_kwargs has a
    #   non-empty "custom_backend" list, return the result of calling each
    #   factory (defaults ignored). Otherwise: in internal env
    #   (env_utils.is_internal_env) build CluBackend(log_dir, **clu kwargs),
    #   raising ImportError if CluBackend is unavailable; else build
    #   TensorboardBackend(log_dir, flush_every_n_steps, **tensorboard
    #   kwargs) and try WandbBackend(project=project_name, name=run_name,
    #   **wandb kwargs), skipping it with an info log on ImportError.
    # Exceptions: ImportError (internal without CLU), backend constructor
    #   errors; passing log_dir/flush_every_n_steps in tensorboard kwargs
    #   causes a duplicate keyword TypeError. Side effects: backends may
    #   open files or network sessions.
    """Factory method to create a fresh set of live backends."""
    # Only create live backends on the main process.
    if jax.process_index() != 0:
      return []

    # Case 1: Override. Use user-provided factories.
    if (
        "custom_backend" in self.backend_kwargs
        and self.backend_kwargs["custom_backend"]
    ):
      return [factory() for factory in self.backend_kwargs["custom_backend"]]

    # Case 2: Defaults.
    active_backends = []
    kwargs_dict = self.backend_kwargs or {}

    if env_utils.is_internal_env():
      if CluBackend is None:
        raise ImportError(
            "Internal environment detected, but CluBackend not available."
        )
      clu_kwargs = kwargs_dict.get("clu", {})
      active_backends.append(CluBackend(log_dir=self.log_dir, **clu_kwargs))
    else:
      tb_kwargs = kwargs_dict.get("tensorboard", {})
      active_backends.append(
          TensorboardBackend(
              log_dir=self.log_dir,
              flush_every_n_steps=self.flush_every_n_steps,
              **tb_kwargs,
          )
      )
      try:
        wandb_kwargs = kwargs_dict.get("wandb", {})
        active_backends.append(
            WandbBackend(
                project=self.project_name,
                name=self.run_name,
                **wandb_kwargs,
            )
        )
      except ImportError:
        logging.info("WandbBackend skipped: 'wandb' library not installed.")
    return active_backends


class Mode(str, enum.Enum):
  # [CLASS DOC]
  # String enum of logging modes: TRAIN = "train", EVAL = "eval".
  TRAIN = "train"
  EVAL = "eval"

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the raw value ("train"/"eval") so f-strings and str() yield
    # it instead of "Mode.TRAIN".
    return self.value


def _calculate_geometric_mean(x: np.ndarray) -> np.ndarray:
  """Calculates geometric mean of a batch of values."""
  # [FUNCTION DOC]
  # HELPER: returns exp(mean(log(x))) over all elements of x (geometric
  #   mean). Non-positive values give nan/-inf per numpy semantics.
  return np.exp(np.mean(np.log(x)))


class MetricsLogger:
  """Simple Metrics logger.
  # [CLASS DOC]
  # Metrics logger. State: _metrics nested dict prefix -> mode ->
  # defaultdict(list) of metric name -> list of logged values; _backends.
  # Aggregation: get_metric returns the mean of all values ever logged,
  # except "perplexity" which uses the geometric mean.
  # KNOWN ISSUE: docstring says it logs to default backends when none are
  #   specified, but with no options no backends are created at all.

  Log metrics to multiple backends. If no backends are specified, it will log to
  the default backends.
  """

  def __init__(
      self,
      metrics_logger_options: MetricsLoggerOptions | None = None,
  ):
    # [FUNCTION DOC]
    # Input: metrics_logger_options (optional). Initializes empty history;
    #   backends = options.create_backends() if options given else []. On
    #   process 0 with options, registers each backend.log_scalar as a
    #   global jax.monitoring scalar listener.
    # Side effects: global listener registration; backend creation.
    self._metrics = {}
    self._backends = (
        metrics_logger_options.create_backends()
        if metrics_logger_options
        else []
    )
    if metrics_logger_options and jax.process_index() == 0:
      for backend in self._backends:
        jax.monitoring.register_scalar_listener(backend.log_scalar)

  def log(
      self,
      metrics_prefix: str,
      metric_name: str,
      scalar_value: float | np.ndarray,
      mode: Mode | str,
      step: int,
  ):
    # [FUNCTION DOC]
    # Inputs: metrics_prefix, metric_name, scalar_value, mode (Mode or str),
    #   step. Appends the value to history[prefix][mode][name] (creating
    #   containers) and calls jax.monitoring.record_scalar with name
    #   "<prefix>/<mode>/<name>" and step, which fans out to listeners.
    # Note: Mode.TRAIN and "train" hash equal (str enum), so they share
    #   history. Side effects: history growth, backend writes.
    """Logs the scalar metric value to local history and via jax.monitoring."""
    prefix_metrics = self._metrics.setdefault(metrics_prefix, {})
    mode_metrics = prefix_metrics.setdefault(
        mode, collections.defaultdict(list)
    )
    mode_metrics[metric_name].append(scalar_value)

    jax.monitoring.record_scalar(
        f"{metrics_prefix}/{mode}/{metric_name}", scalar_value, step=step
    )

  def metric_exists(
      self, metrics_prefix, metric_name: str, mode: Mode | str
  ) -> bool:
    # [FUNCTION DOC]
    # Inputs: prefix, name, mode. Returns True iff history contains
    #   prefix, then mode under it, then name under that. No side effects.
    """Checks if the metric exists for the given metric name and mode."""
    if metrics_prefix not in self._metrics:
      return False
    if mode not in self._metrics[metrics_prefix]:
      return False
    return metric_name in self._metrics[metrics_prefix][mode]

  def get_metric(self, metrics_prefix, metric_name: str, mode: Mode | str):
    # [FUNCTION DOC]
    # Inputs: prefix, name, mode. Raises ValueError if metric missing.
    #   Stacks all logged values with numpy and returns their geometric
    #   mean if name == "perplexity", else the arithmetic mean (numpy
    #   scalar/array).
    """Returns the mean metric value for the given metric name and mode."""
    if not self.metric_exists(metrics_prefix, metric_name, mode):
      raise ValueError(
          f"Metric '{metrics_prefix}/{mode}/{metric_name}' not found."
      )
    values = np.stack(self._metrics[metrics_prefix][mode][metric_name])
    if metric_name == "perplexity":
      return _calculate_geometric_mean(values)
    return np.mean(values)

  def get_metric_history(
      self, metrics_prefix, metric_name: str, mode: Mode | str
  ):
    # [FUNCTION DOC]
    # Inputs: prefix, name, mode. Returns numpy stack of all logged values.
    #   Raises ValueError listing available metric names if missing.
    # KNOWN ISSUE: building that error message indexes
    #   _metrics[prefix][mode], which raises KeyError instead when the
    #   prefix or mode itself is missing.
    """Returns all past metric values for the given metric name and mode."""
    if not self.metric_exists(metrics_prefix, metric_name, mode):
      raise ValueError(
          f" Metric '{metrics_prefix}/{mode}/{metric_name}' not found."
          f" Available metrics for mode '{mode}':"
          f" {list(self._metrics[metrics_prefix][mode].keys())}"
      )
    return np.stack(self._metrics[metrics_prefix][mode][metric_name])

  def close(self):
    # [FUNCTION DOC]
    # Closes every backend, then calls jax.monitoring.clear_event_listeners
    #   ignoring any exception. Side effects: clears ALL global
    #   jax.monitoring listeners in the process, not only this logger's.
    """Closes all registered logging backends."""
    for backend in self._backends:
      backend.close()
    try:
      jax.monitoring.clear_event_listeners()
    except Exception:  # pylint: disable=broad-exception-caught
      # We didn't register the scalar listener, so this is expected.
      pass
