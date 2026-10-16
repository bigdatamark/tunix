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
# RL-specific trainer built on the PEFT trainer (tunix.sft.peft_trainer). Adds
# per-step collection of auxiliary RL metrics returned by the loss function,
# configurable extra progress-bar metrics, and checkpoint metadata supplied by
# a callback (used to persist the RL global step). No import-time side
# effects; MetricsLogger is imported for typing.
"""RL Trainer."""

from typing import Any, Callable, Optional

from flax import nnx
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import optax
from tunix.sft import peft_trainer
from typing_extensions import override
from tunix.perf import trace as perf_trace
from tunix.perf.experimental import tracer as perf_tracer_lib
from tunix.sft.metrics_logger import MetricsLogger  # pylint: disable=unused-import


class Trainer(peft_trainer.PeftTrainer):
  # [CLASS DOC]
  # PeftTrainer subclass. Class attribute supports_sequence_packing is True.
  # Instance state: rl_metrics_to_log (dict metric name to aggregation
  # function; the name is also the key read from the loss aux dict),
  # tqdm_metrics_to_display (list of metric names or zero-arg callables returning
  # a name or a falsy value), custom_checkpoint_metadata_fn (zero-arg callable
  # returning a dict saved with checkpoints).
  """Handles additional RL metrics logging and display."""

  supports_sequence_packing = True

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: peft_trainer.TrainingConfig,
      custom_checkpoint_metadata_fn: Callable[[], dict[str, Any]],
      metrics_logger: Optional[MetricsLogger] = None,
      perf_tracer: Optional[perf_trace.Tracer] = None,
      perf_tracer_v2: Optional[perf_tracer_lib.Tracer] = None,
  ):
    # [FUNCTION DOC]
    # Inputs: model, optax optimizer, TrainingConfig, checkpoint-metadata
    # callback, optional metrics logger, optional perf tracers (v1 and v2).
    # Forwards model, optimizer, config, logger and tracers positionally to
    # PeftTrainer.__init__, then initializes rl_metrics_to_log to an empty dict,
    # tqdm_metrics_to_display to an empty list and stores the metadata callback.
    super().__init__(
        model,
        optimizer,
        training_config,
        metrics_logger,
        perf_tracer,
        perf_tracer_v2,
    )
    self.rl_metrics_to_log = {}  # Metric name -> key in aux.
    self.tqdm_metrics_to_display = []
    self.custom_checkpoint_metadata_fn = custom_checkpoint_metadata_fn

  def with_rl_metrics_to_log(
      self,
      rl_metrics_to_log: dict[str, Callable[[ArrayLike], ArrayLike]],
  ) -> None:
    # [FUNCTION DOC]
    # Replaces rl_metrics_to_log with the given mapping of metric name to
    # aggregation function. Returns None.
    self.rl_metrics_to_log = rl_metrics_to_log

  def with_tqdm_metrics_to_display(
      self, tqdm_metrics_to_display: list[str | Callable[[], str]]
  ) -> None:
    # [FUNCTION DOC]
    # Replaces tqdm_metrics_to_display with the given list of names or callables.
    # Returns None.
    self.tqdm_metrics_to_display = tqdm_metrics_to_display

  @override
  def custom_checkpoint_metadata(self) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Override: returns the dict produced by calling the stored metadata callback;
    # saved alongside checkpoints by the base trainer.
    return self.custom_checkpoint_metadata_fn()

  def restored_global_step(self) -> int:
    # [FUNCTION DOC]
    # Returns the "global_step" entry of the custom metadata restored from a
    # checkpoint by the base trainer, or 0 if absent.
    return self._restored_custom_metadata.get("global_step", 0)

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    # [FUNCTION DOC]
    # Override called after each train step with the loss aux dict. Asserts the
    # buffered train metrics object exists (AssertionError otherwise). For every
    # configured metric name: if not yet buffered, stores a pair of a one-element
    # list holding aux value and the aggregation function in additional_metrics;
    # otherwise appends the aux value to the existing list. Raises KeyError if
    # aux lacks a configured metric. Returns None.
    assert self._buffered_train_metrics is not None
    for metric_name, op in self.rl_metrics_to_log.items():
      if metric_name not in self._buffered_train_metrics.additional_metrics:
        self._buffered_train_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_train_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )

  @override
  def _post_process_eval_step(self, aux: Any) -> None:
    # [FUNCTION DOC]
    # Same as _post_process_train_step but for evaluation steps, writing into the
    # buffered eval metrics.
    assert self._buffered_eval_metrics is not None
    for metric_name, op in self.rl_metrics_to_log.items():
      if metric_name not in self._buffered_eval_metrics.additional_metrics:
        self._buffered_eval_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_eval_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )

  def _get_additional_tqdm_metrics(self) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Builds the set of extra progress-bar metric names: string entries
    # are added as is; callables are invoked and their result added if truthy.
    # Returns the set as a list (order not guaranteed, duplicates removed).
    metrics = set()
    for key_or_fn in self.tqdm_metrics_to_display:
      if isinstance(key_or_fn, str):
        metrics.add(key_or_fn)
      elif val := key_or_fn():
        metrics.add(val)
    return list(metrics)

  @property
  def _tqdm_train_metrics(self) -> list[str]:
    # [FUNCTION DOC]
    # Property: the base trainer's train progress-bar metric list extended with
    # _get_additional_tqdm_metrics. Returns the list.
    # KNOWN ISSUE: it extends the list returned by the parent in place, which may
    # mutate a shared list if the parent returns a stored attribute.
    metrics = super()._tqdm_train_metrics
    metrics.extend(self._get_additional_tqdm_metrics())
    return metrics
