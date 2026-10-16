# [MODULE DOC]
# tqdm-based training progress bar that displays selected metrics read
#   from a MetricsLogger as postfix values.
# Defines _is_in_ipython, module alias tqdm = tqdm.auto.tqdm, and
#   ProgressBar. No license header. No side effects at import.
"""Progress bar."""

import sys

from absl import logging
from tqdm import auto
from tunix.sft import metrics_logger as ml


def _is_in_ipython() -> bool:
  """Returns True if running in an IPython environment."""
  # [FUNCTION DOC]
  # HELPER: returns True iff the IPython module is already imported, has
  #   get_ipython, and get_ipython() returns non-None (i.e. running inside
  #   an IPython kernel/shell). Does not import IPython. No exceptions.
  ipython = sys.modules.get("IPython")
  if ipython is not None and hasattr(ipython, "get_ipython"):
    return ipython.get_ipython() is not None
  return False


tqdm = auto.tqdm


class ProgressBar:
  # [CLASS DOC]
  # Progress bar over training steps. Attributes: tqdm_bar, metrics (dict of
  # display key -> rounded value), initial_steps, max_steps, metrics_prefix,
  # metrics_logger, description, disable_warning_for_metrics (set of metric
  # names that do not warn when missing; {"learning_rate"}).
  # Display key format: "<prefix>_<mode>_<metric name>".
  """Progress bar."""

  def __init__(
      self,
      metrics_prefix: str,
      metrics_logger: ml.MetricsLogger,
      initial_steps: int,
      max_steps: int,
      description: str | None = None,
  ):
    # [FUNCTION DOC]
    # Inputs: metrics_prefix (str), metrics_logger (MetricsLogger),
    #   initial_steps (start count), max_steps (total), description
    #   (optional label).
    # Creates a tqdm bar (total=max_steps, initial=initial_steps,
    #   unit "step", dynamic width, leave=True, custom bar_format with
    #   elapsed/remaining/rate/postfix), disabled unless in IPython or
    #   stderr is a TTY; stores all arguments and empty metrics.
    # Side effects: may write to stderr.
    # Initialise progress bar.
    self.tqdm_bar = tqdm(
        total=max_steps,
        initial=initial_steps,
        desc=description,
        unit="step",
        dynamic_ncols=True,
        leave=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
            "{rate_fmt}{postfix}]"
        ),
        # Disable progress bar if not in IPython or TTY.
        disable=not (_is_in_ipython() or sys.stderr.isatty()),
    )

    # Also, initialise a dictionary for metrics.
    self.metrics = {}
    self.initial_steps = initial_steps
    self.max_steps = max_steps
    self.metrics_prefix = metrics_prefix
    self.metrics_logger = metrics_logger
    self.description = description
    self.disable_warning_for_metrics = {"learning_rate"}

  def _update_metric(self, metric_name: str, mode: ml.Mode, ndigits: int = 3):
    # [FUNCTION DOC]
    # HELPER: Inputs: metric_name, mode (ml.Mode), ndigits (default 3).
    #   If metrics_logger has the metric for (prefix, name, mode), store
    #   round(get_metric(...).item(), ndigits) in metrics under
    #   "<prefix>_<str(mode)>_<name>"; else, unless the name is in
    #   disable_warning_for_metrics, log a warning. No output.
    """Update metric corresponding to `metric_name`."""

    mode_str = str(mode)
    if self.metrics_logger.metric_exists(
        self.metrics_prefix, metric_name, mode
    ):
      self.metrics[f"{self.metrics_prefix}_{mode_str}_{metric_name}"] = round(
          self.metrics_logger.get_metric(
              self.metrics_prefix, metric_name, mode
          ).item(),
          ndigits,
      )
    elif metric_name not in self.disable_warning_for_metrics:
      logging.warning(
          "Metric %s not found for mode %s. Not logging metric.",
          metric_name,
          mode_str,
      )

  def update_metrics(
      self, metric_names: list[str], mode: ml.Mode, ndigits: int = 3
  ):
    # [FUNCTION DOC]
    # Inputs: metric_names (list), mode, ndigits. Calls _update_metric for
    #   each name. No output.
    """Update metrics corresponding to `metric_names`."""

    for metric_name in metric_names:
      self._update_metric(metric_name, mode, ndigits)

  def _sort_metrics(self):
    # [FUNCTION DOC]
    # HELPER: rebuilds self.metrics so all "train" keys come first then
    #   "eval" keys (each in insertion order), where mode is the second
    #   underscore-separated token of the key. Keys of any other mode are
    #   dropped.
    # KNOWN ISSUE: if metrics_prefix contains "_" the parsed mode is wrong
    #   and those metrics are silently discarded.
    """Sort metrics by mode."""
    sorted_metrics = {}

    for prefix_mode_metric_name, metric_value in self.metrics.items():
      mode = prefix_mode_metric_name.split("_")[1]
      if mode == "train":
        sorted_metrics[prefix_mode_metric_name] = metric_value

    for prefix_mode_metric_name, metric_value in self.metrics.items():
      mode = prefix_mode_metric_name.split("_")[1]
      if mode == "eval":
        sorted_metrics[prefix_mode_metric_name] = metric_value

    self.metrics = sorted_metrics

  def update(self, n: int = 1):
    # [FUNCTION DOC]
    # Input: n (steps to advance, default 1). If any metrics exist, sort
    #   them and set them as the tqdm postfix; then advance the bar by n.
    # Side effects: terminal output.
    """Update progress bar."""
    if self.metrics:
      self._sort_metrics()
      self.tqdm_bar.set_postfix(self.metrics)
    self.tqdm_bar.update(n)

  def close(self):
    # [FUNCTION DOC]
    # Deletes the metrics attribute and closes the tqdm bar.
    # Side effect: calling update/_update_metric afterwards raises
    #   AttributeError.
    del self.metrics
    self.tqdm_bar.close()
