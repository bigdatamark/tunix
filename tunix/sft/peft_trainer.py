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
# Core supervised fine-tuning trainer (PeftTrainer) used directly for SFT /
#   LoRA and as the base class for DPO and distillation trainers.
# Defines TrainingConfig, TrainingInput, MetricsBuffer,
#   _calculate_global_batch_size, PeftTrainer and _default_loss_fn. Module
#   aliases: _ModelInputT (dict of str -> array), P (ParamSpec),
#   MetricsLogger / MetricsLoggerOptions (from tunix.sft.metrics_logger).
# Training model: optimizer updates only nnx.LoRAParam when the model has
#   LoRA params, otherwise all nnx.Param; optional gradient accumulation via
#   optax.MultiSteps; jitted train/eval steps; checkpoint restore on init and
#   periodic save; metrics buffered per update step and written one step
#   late to overlap host I/O; optional eval every N steps, profiler window,
#   progress bar, hooks, inflight throttling and perf tracing spans.
# No side effects at import.
"""PEFT trainer."""

from collections.abc import Iterable
import contextlib
import dataclasses
import functools
import time
from typing import Any, Callable, Concatenate, Dict, List, ParamSpec, Tuple

from absl import logging
import flax
from flax import nnx
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import jax.sharding as shd
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import numpy as np
import optax
import orbax.checkpoint as ocp
from tunix.perf import metrics as perf_metrics
from tunix.perf import trace as perf_trace
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_lib
from tunix.sft import checkpoint_manager
from tunix.sft import hooks
from tunix.sft import inflight_throttler
from tunix.sft import metrics_logger as sft_metrics_logger
from tunix.sft import profiler
from tunix.sft import progress_bar
from tunix.sft import sharding_utils
from tunix.sft import utils

_ModelInputT = Dict[str, ArrayLike]
P = ParamSpec("P")
MetricsLogger = sft_metrics_logger.MetricsLogger
MetricsLoggerOptions = sft_metrics_logger.MetricsLoggerOptions


@dataclasses.dataclass(slots=True, kw_only=True)
class TrainingConfig:
  # [CLASS DOC]
  # Keyword-only slotted trainer config. Fields: eval_every_n_steps
  # (required; eval cadence in update steps), max_steps (update-step cap or
  # None), gradient_accumulation_steps (None = 1), checkpoint_root_directory
  # (None disables checkpointing), checkpointing_options (Orbax v0
  # CheckpointManagerOptions), metrics_logging_options, profiler_options,
  # perf_metrics_options (not used by PeftTrainer itself),
  # data_sharding_axis (("fsdp",)), max_inflight_computations (2; <= 0 means
  # unlimited), metrics_prefix (""), pbar_description ("Training"),
  # max_seq_token_per_tpu (sequence packing; must be None unless the trainer
  # class supports packing).
  # KNOWN ISSUE: the comment says checkpoints contain the train data
  #   iterator state, but only model/optimizer state and custom metadata
  #   are saved; resumption skips already-consumed examples instead.
  """Configuration for the trainer."""

  eval_every_n_steps: int
  max_steps: int | None = None
  gradient_accumulation_steps: int | None = None

  # If set, the checkpoints will be saved to this path. Checkpoints
  # contains the model params and the train data iterator state.
  checkpoint_root_directory: str | None = None
  # Checkpoint configurations. If None, the default options will be used.
  checkpointing_options: ocp.CheckpointManagerOptions | None = None

  # Configs for the metrics logger.
  metrics_logging_options: MetricsLoggerOptions | None = None

  # Configs for the profiler.
  profiler_options: profiler.ProfilerOptions | None = None

  # Configs for performance metrics.
  perf_metrics_options: perf_metrics.PerfMetricsOptions | None = None

  data_sharding_axis: Tuple[str, ...] = ("fsdp",)

  # Controls how many train_steps can be scheduled ahead of time.
  max_inflight_computations: int = 2

  # Prefix for metric names for logging. Not sticking it in
  # `metrics_logging_options` because the latter is optional.
  metrics_prefix: str = ""

  # Progress bar description.
  pbar_description: str | None = "Training"

  # Sequence packing configuration.
  max_seq_token_per_tpu: int | None = None

  def get_with_default(self, key: str, default: Any) -> Any:
    # [FUNCTION DOC]
    # HELPER: Inputs: key (attribute name), default. Returns the attribute
    #   value, or default when it is None. AttributeError for unknown keys.
    val = getattr(self, key)
    if val is None:
      return default
    return val


@flax.struct.dataclass(frozen=True)
class TrainingInput:
  # [CLASS DOC]
  # Frozen flax struct (pytree) batch: input_tokens [B, L], input_mask
  # [B, L] (valid-token mask), optional images.
  # Input tokens provided to the model.
  input_tokens: jax.Array | np.ndarray

  # A mask that determines which input tokens are valid.
  input_mask: jax.Array | np.ndarray

  # Optional images for vision models.
  images: jax.Array | np.ndarray | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class MetricsBuffer:
  # [CLASS DOC]
  # Per-update-step metric accumulator (keyword-only, slotted dataclass).
  # Fields: step, losses (list of per-micro-step losses), additional_metrics
  # (name -> (list of values, aggregation fn)).
  """Metrics collected for a specific step.

  Attributes:
    step: The training step number.
    losses: A list of loss values recorded within this step (e.g., across
      gradient accumulation steps).
    additional_metrics: Dictionary for storing additional metrics. The key is
      the metric name, and the value is a tuple containing a list of metric
      values and a callable to aggregate them.
  """

  step: int
  losses: List[ArrayLike]
  additional_metrics: Dict[
      str, Tuple[List[ArrayLike], Callable[[ArrayLike], ArrayLike]]
  ] = dataclasses.field(default_factory=dict)

  @property
  def loss(self):
    # [FUNCTION DOC]
    # Returns the arithmetic mean of all recorded losses after converting
    # each to numpy (forces device-to-host transfer).
    """Returns the mean of the recorded losses for the step."""
    return np.mean(np.array([np.array(x) for x in self.losses]))


def _calculate_global_batch_size(train_example: Any) -> int:
  """Calculates the global batch size from a training example.

  Args:
    train_example: A training example, which can be a dataclass, a dict, or an
      object with attributes.

  Returns:
    The global batch size.

  Raises:
    TypeError: If the batch size cannot be determined from the training example.
  """
  # [FUNCTION DOC]
  # HELPER: Input: train_example (dataclass, dict, or object with __dict__).
  #   Collects its fields (dataclasses.asdict / the dict / vars()), and
  #   returns shape[0] of the first value that is a jax or numpy array.
  # Exceptions: TypeError if no array field exists (vars() raises TypeError
  #   for objects without __dict__). Not used within this module.
  if dataclasses.is_dataclass(train_example):
    attributes = dataclasses.asdict(train_example)
  elif isinstance(train_example, dict):
    attributes = train_example
  else:
    attributes = vars(train_example)

  for field_value in attributes.values():
    if isinstance(field_value, (jax.Array, np.ndarray)):
      # Assume the first array we find has the batch dimension.
      return field_value.shape[0]

  raise TypeError(
      "Could not automatically determine batch size. No JAX or NumPy "
      "array found in the training example."
  )


class PeftTrainer:
  """PEFT trainer for LoRA. Only LoRA parameters are updated.
  # [CLASS DOC]
  # Generic trainer. Class attribute supports_sequence_packing = False.
  # Key state: model, config, optimizer (nnx.Optimizer, wrt LoRAParam or
  # Param), loss_fn / eval_loss_fn (called as fn(model, **model_inputs)),
  # gen_model_input_fn (batch -> kwargs dict for the loss; identity by
  # default), checkpoint_manager, metrics_logger, metrics_prefix,
  # is_managed_externally (if True: no max_steps stop, no skipping of
  # already-trained examples, no auto close), training_hooks, data_hooks,
  # _train_steps (optimizer updates), _iter_steps (micro steps), _mode,
  # _has_aux, _pbar, _prof, metric buffers, jitted step caches.
  # Subclasses customize via _prepare_inputs, _post_process_train_step,
  # _post_process_eval_step and custom_checkpoint_metadata.
  # KNOWN ISSUE: docstring says only LoRA params are updated; without LoRA
  #   params all nnx.Param are trained.

  Attributes:
    model: The model to train.
    config: The training config.
    optimizer: The optimizer to use. To monitor the learning rate at each step,
      use `optax.schedules.inject_hyperparams` to inject learning rate as a
      hyperparameter. For example: ``optimizer =
      optax.schedules.inject_hyperparams(optax.sgd)(learning_rate=learning_rate_schedule)``
    loss_fn: The loss function to use.
    eval_loss_fn: The loss function to use for evaluation.
    gen_model_input_fn: The function to generate model input from training
      input.
    checkpoint_manager: The checkpoint manager to use.
    metrics_logger: The metrics logger to use.
    metrics_prefix: The prefix for metric names for logging.
    is_managed_externally: Whether the trainer is managed externally.
    training_hooks: The training hooks to use.
    data_hooks: The data hooks to use.
  """

  supports_sequence_packing = False

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: TrainingConfig,
      metrics_logger: MetricsLogger | None = None,
      perf_tracer: perf_trace.Tracer | None = None,
      perf_tracer_v2: perf_tracer_lib.Tracer | None = None,
  ):
    # [FUNCTION DOC]
    # Inputs: model, optimizer (optax transform), training_config,
    #   metrics_logger (optional; created from config if None), perf_tracer
    #   and perf_tracer_v2 (optional; no-op tracers by default).
    # Steps: ValueError if max_seq_token_per_tpu is set and the class does
    #   not support packing. Detect LoRA; wrap optimizer in optax.MultiSteps
    #   if gradient_accumulation_steps set; build nnx.Optimizer wrt LoRAParam
    #   or Param. Default loss fns = _default_loss_fn; identity input fn.
    #   Build CheckpointManager from config; metrics logger; throttler;
    #   mode TRAIN; _has_aux False. Restore from the latest checkpoint
    #   (model + optimizer, LoRA-only if LoRA) giving _train_steps and
    #   restored custom metadata; _iter_steps = _train_steps * accumulation.
    #   Build Profiler with initial step _iter_steps and max step
    #   max_steps * accumulation. Init buffers/hooks/jit cache to empty.
    # Exceptions: ValueError above; checkpoint/profiler errors (profiler
    #   ValueError if its window is empty). Side effects: restores model and
    #   optimizer state in place; creates logging backends.
    # TODO(noghabi): Implement sequence packing for SFT and remove this check.
    if (
        training_config.max_seq_token_per_tpu is not None
        and not self.supports_sequence_packing
    ):
      raise ValueError(
          "Sequence packing is not supported in SFT PeftTrainer yet."
      )

    self.model = model
    self.config = training_config
    self._lora_enabled = utils.is_lora_enabled(self.model)
    if training_config.gradient_accumulation_steps is not None:
      optimizer = optax.MultiSteps(
          optimizer, training_config.gradient_accumulation_steps
      )
    if self._lora_enabled:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.LoRAParam)
    else:
      self.optimizer = nnx.Optimizer(self.model, optimizer, wrt=nnx.Param)

    self.loss_fn = _default_loss_fn
    self.eval_loss_fn = _default_loss_fn
    self.gen_model_input_fn = lambda x: x
    self.checkpoint_manager = checkpoint_manager.CheckpointManager(
        root_directory=self.config.checkpoint_root_directory,
        options=self.config.checkpointing_options,
    )
    self.metrics_logger = metrics_logger
    self.metrics_prefix = self.config.metrics_prefix
    if self.metrics_logger is None:
      self.metrics_logger = MetricsLogger(
          self.config.metrics_logging_options,
      )
    self.is_managed_externally = False
    self._perf_tracer = (
        perf_tracer if perf_tracer is not None else perf_trace.NoopTracer()
    )
    self._perf_tracer_v2 = (
        perf_tracer_v2
        if perf_tracer_v2 is not None
        else perf_tracer_lib.NoopTracer()
    )

    self._train_steps = 0  # represent # of times model has been updated
    self._iter_steps = 0  # represent # of times trainer has looped
    self._throttler = inflight_throttler.InflightThrottler(
        max_inflight=training_config.max_inflight_computations
    )
    self._mode: sft_metrics_logger.Mode = sft_metrics_logger.Mode.TRAIN
    self._has_aux = False
    self._pbar = None

    self._train_steps, self._restored_custom_metadata = (
        self.checkpoint_manager.maybe_restore(
            self.model,
            self.optimizer,
            restore_only_lora_params=self._lora_enabled,
        )
    )
    self._iter_steps = self._train_steps * self.config.get_with_default(
        "gradient_accumulation_steps", 1
    )

    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None
    max_step = None
    if self.config.max_steps is not None:
      max_step = self.config.max_steps * self.config.get_with_default(
          "gradient_accumulation_steps", 1
      )
    self._prof = profiler.Profiler(
        initial_step=self._iter_steps,
        max_step=max_step,
        profiler_options=self.config.profiler_options,
    )
    self._buffered_train_metrics: MetricsBuffer | None = None
    self._prev_buffered_train_metrics: MetricsBuffer | None = None
    self._buffered_eval_metrics: MetricsBuffer | None = None
    self.training_hooks = None
    self.data_hooks = None
    self._jit_cache = set()
    self._mini_batch_size = None

  def with_training_hooks(self, training_hooks: hooks.TrainingHooks):
    # [FUNCTION DOC]
    # Sets self.training_hooks. Returns None (not chainable).
    self.training_hooks = training_hooks

  def with_data_hooks(self, data_hooks: hooks.DataHooks):
    # [FUNCTION DOC]
    # Sets self.data_hooks (batch loading callbacks). Returns None.
    self.data_hooks = data_hooks

  def clear_jit_cache(self):
    # [FUNCTION DOC]
    # Resets cached jitted train/eval step fns to None so the next
    # jit_train_and_eval_step rebuilds them. No output.
    """Clears the JIT cache of the train and eval step functions.

    This function should be called when the trainer is being reused after
    overriding the training related states, for example, the loss function.
    """
    self._jitted_train_step_fn = None
    self._jitted_eval_step_fn = None

  def with_loss_fn(
      self,
      loss_fn: Callable[
          Concatenate[nnx.Module, P], ArrayLike | Tuple[ArrayLike, Any]
      ],
      has_aux: bool = False,
  ):
    # [FUNCTION DOC]
    # Inputs: loss_fn (model first, then kwargs from gen_model_input_fn;
    #   returns loss or (loss, aux)), has_aux. Clears jit cache, sets both
    #   loss_fn and eval_loss_fn, stores has_aux. Returns self.
    self.clear_jit_cache()
    self.loss_fn = loss_fn
    self.eval_loss_fn = loss_fn
    self._has_aux = has_aux
    return self

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], _ModelInputT]
  ):
    # [FUNCTION DOC]
    # Input: gen_model_input_fn (batch -> kwargs dict for the loss fn).
    #   Clears jit cache, stores it, returns self.
    """Generates model input from training input.

    NB: output of this function will be passed to the loss function, so the args
    should match what loss function expects.

    Args:
      gen_model_input_fn: A function that generates model input from training
        input.

    Returns:
      PeftTrainer.
    """
    self.clear_jit_cache()
    self.gen_model_input_fn = gen_model_input_fn
    return self

  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any
  ) -> Tuple[ArrayLike, Any | None, ArrayLike]:
    # [FUNCTION DOC]
    # Train step body (jitted later). Inputs: model, optimizer, inputs
    #   (prepared, sharded batch). Steps: kwargs = gen_model_input_fn(inputs);
    #   value_and_grad of loss_fn w.r.t. the model (restricted to LoRAParam
    #   when LoRA enabled) with has_aux; compute optax.global_norm of grads;
    #   optimizer.update(model, grads).
    # Output: (loss, aux or None, grad_norm). Side effects: updates model
    #   params and optimizer state in place.
    """Main body for one train step.

    Args:
      model: The model to train.
      optimizer: The optimizer to use.
      inputs: The training input.

    Returns:
      A tuple containing the loss, auxiliary data (or None if has_aux is False),
      and the gradient norm.
    """
    inputs = self.gen_model_input_fn(inputs)

    grad_fn = nnx.value_and_grad(
        self.loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=self._has_aux,
    )
    out, grads = grad_fn(model, **inputs)
    grad_norm = optax.global_norm(grads)
    optimizer.update(model, grads)
    if self._has_aux:
      loss, aux = out
      return loss, aux, grad_norm
    else:
      return out, None, grad_norm

  def _eval_step(
      self, model: nnx.Module, inputs: Any
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    # [FUNCTION DOC]
    # Eval step body. Inputs: model, inputs. kwargs = gen_model_input_fn;
    #   out = eval_loss_fn(model, **kwargs). Output: (loss, aux) if has_aux
    #   else (out, None). No side effects.
    # KNOWN ISSUE: the return annotation suggests a bare loss is possible,
    #   but a 2-tuple is always returned.
    inputs = self.gen_model_input_fn(inputs)
    out = self.eval_loss_fn(model, **inputs)
    if self._has_aux:
      loss, aux = out
      return loss, aux
    else:
      return out, None

  def create_train_step_fn(
      self,
  ) -> Callable[..., Tuple[ArrayLike, Any | None, ArrayLike]]:
    # [FUNCTION DOC]
    # Returns the unjitted train step callable (self._train_step). Override
    # point for subclasses.
    """Creates the train step function."""
    return self._train_step

  def create_eval_step_fn(self) -> Callable[..., ArrayLike]:
    # [FUNCTION DOC]
    # Returns the unjitted eval step callable (self._eval_step). Override
    # point for subclasses.
    """Creates the eval step function."""
    return self._eval_step

  def _shard_optimizer(self, mesh: shd.Mesh) -> None:
    """Optimizer states should be sharded before calling the jit function.

    If not, the _train_step will be compiled 2 times.

    Args:
      mesh: The mesh used for sharding.
    """
    # [FUNCTION DOC]
    # HELPER: Input: mesh. No-op if the mesh is empty. Otherwise reads the
    #   optimizer OptState, gets its partition specs from nnx metadata,
    #   applies jax.lax.with_sharding_constraint with those specs and writes
    #   the result back so the jitted step is not compiled twice.
    # Side effects: mutates optimizer state placement.
    if mesh.empty:
      return
    optimizer_state = nnx.state(self.optimizer, nnx.optimizer.OptState)
    optimizer_pspecs = nnx.get_partition_spec(optimizer_state)

    optimizer_sharded_state = jax.lax.with_sharding_constraint(
        optimizer_state, optimizer_pspecs
    )
    nnx.update(self.optimizer, optimizer_sharded_state)

  def jit_train_and_eval_step(
      self, skip_jit: bool = False, cache_nnx_graph: bool = False
  ):
    """Creates and returns the train and eval step functions.

    This function will return the cached ones if available.

    Args:
      skip_jit: If True, the train and eval step functions will not be JITed.
      cache_nnx_graph: If True, the nnx graph will be cached.

    Returns:
      A tuple of train and eval step functions.
    """
    # [FUNCTION DOC]
    # Inputs: skip_jit, cache_nnx_graph. If skip_jit, return the raw
    #   (train_step, eval_step) fns (which expect model/optimizer args).
    #   Otherwise, if not cached: shard optimizer on the active mesh; jit the
    #   train step (donating the optimizer argument) and eval step with
    #   nnx.jit; bind model and optimizer (train) / model (eval) as leading
    #   arguments via functools.partial, using nnx.cached_partial first when
    #   cache_nnx_graph. Cache and return (train_fn(batch), eval_fn(batch)).
    # KNOWN ISSUE: with skip_jit the returned fns are not bound to
    #   model/optimizer, yet train() calls them with only the batch.
    train_step = self.create_train_step_fn()
    eval_step = self.create_eval_step_fn()
    if skip_jit:
      return train_step, eval_step

    if self._jitted_train_step_fn is None:
      self._shard_optimizer(pxla.thread_resources.env.physical_mesh)
      self._jitted_train_step_fn = nnx.jit(
          train_step, donate_argnames=("optimizer",)
      )
      self._jitted_eval_step_fn = nnx.jit(eval_step)

      def maybe_cache_and_partial(f, *args):
      # [FUNCTION DOC]
      # HELPER: nested. Inputs: f, positional args to bind. Returns
      #   functools.partial(nnx.cached_partial(f, *args)) when
      #   cache_nnx_graph (graph traversal cached), else
      #   functools.partial(f, *args); the outer partial gives a uniform
      #   .func attribute.
        if cache_nnx_graph:
          # wrap with partial so we can access jitted_fn in a consistent way.
          return functools.partial(nnx.cached_partial(f, *args))
        else:
          return functools.partial(f, *args)

      self._jitted_train_step_fn = maybe_cache_and_partial(
          self._jitted_train_step_fn, self.model, self.optimizer
      )
      self._jitted_eval_step_fn = maybe_cache_and_partial(
          self._jitted_eval_step_fn, self.model
      )
    return self._jitted_train_step_fn, self._jitted_eval_step_fn

  def _prepare_inputs(self, input_data: Any) -> Any:
    # [FUNCTION DOC]
    # Override hook: transform a raw batch before sharding. Default returns
    # it unchanged.
    """Override this function for additional input preparation."""
    return input_data

  def _post_process_train_step(self, aux: Any) -> None:
    # [FUNCTION DOC]
    # Override hook receiving aux from each train step. Default no-op.
    """Override this function for post processing aux data from train step."""
    pass

  def _post_process_eval_step(self, aux: Any) -> None:
    # [FUNCTION DOC]
    # Override hook receiving aux from each eval step. Default no-op.
    """Override this function for post processing aux data from eval step."""
    pass

  def _try_get_learning_rate(self) -> float | None:
    # [FUNCTION DOC]
    # HELPER: returns the current learning rate if exposed via
    #   optax.inject_hyperparams. Tries opt_state.hyperparams
    #   ["learning_rate"].value; on AttributeError iterates opt_state
    #   elements (chain parts), stops at the first optax.EmptyState, and
    #   returns the first part's hyperparams learning rate if any; else None.
    # KNOWN ISSUE: a KeyError (hyperparams without "learning_rate") or a
    #   non-iterable opt_state is not caught.
    """Returns the learning rate from the optimizer state if available."""
    try:
      return self.optimizer.opt_state.hyperparams["learning_rate"].value
    except AttributeError:
      for chainpart in self.optimizer.opt_state:
        if isinstance(chainpart, optax.EmptyState):
          break
        if hasattr(chainpart, "hyperparams"):
          return chainpart.hyperparams["learning_rate"].value
      return None

  def _log_metrics(
      self,
      loss: ArrayLike,
      step: int | None = None,
      additional_metrics: dict[str, ArrayLike] | None = None,
  ):
    # [FUNCTION DOC]
    # HELPER: Inputs: loss, step, additional_metrics (name -> value).
    #   Logs "loss", "perplexity" (= exp(loss) on host) and, if available,
    #   "learning_rate" under the current mode; in TRAIN mode also writes an
    #   info log line; then logs each additional metric. Side effects:
    #   metrics logger writes.
    """Logs the metrics to the metrics logger and console."""
    perplexity = np.exp(jax.device_get(loss))
    self.metrics_logger.log(self.metrics_prefix, "loss", loss, self._mode, step)
    self.metrics_logger.log(
        self.metrics_prefix, "perplexity", perplexity, self._mode, step
    )
    learning_rate = self._try_get_learning_rate()
    if learning_rate is not None:
      self.metrics_logger.log(
          self.metrics_prefix,
          "learning_rate",
          jax.device_get(learning_rate),
          self._mode,
          step,
      )

    if self._mode == sft_metrics_logger.Mode.TRAIN:
      logging.info(
          "Train step %d training loss: %f  - training perplexity: %f",
          step,
          loss,
          perplexity,
      )
    for k, v in (additional_metrics or {}).items():
      self.metrics_logger.log(self.metrics_prefix, k, v, self._mode, step)

  def _buffer_metrics(
      self,
      metrics_buffer: MetricsBuffer | None,
      loss: ArrayLike,
      step: int,
      additional_metrics: (
          dict[str, Tuple[ArrayLike, Callable[[ArrayLike], ArrayLike]]] | None
      ) = None,
  ) -> MetricsBuffer:
    # [FUNCTION DOC]
    # HELPER: Inputs: metrics_buffer (or None), loss, step,
    #   additional_metrics (name -> (value, aggregation fn)). Creates a
    #   buffer for step with [loss] if None; else asserts the buffer step
    #   matches (AssertionError otherwise) and appends loss. Appends each
    #   additional value to its list (creating (list, fn) entries).
    #   Output: the buffer.
    """Buffers metrics for the current step."""
    if metrics_buffer is None:
      metrics_buffer = MetricsBuffer(
          step=step,
          losses=[loss],
      )
    else:
      assert metrics_buffer.step == step
      metrics_buffer.losses.append(loss)
    if additional_metrics is not None:
      for k, (v, op) in additional_metrics.items():
        if k not in metrics_buffer.additional_metrics:
          metrics_buffer.additional_metrics[k] = ([v], op)
        else:
          metrics_buffer.additional_metrics[k][0].append(v)
    return metrics_buffer

  def _write_train_metrics(self):
    # [FUNCTION DOC]
    # HELPER: delayed train metric writing. If there is no previous buffer,
    #   move the current buffer to "previous" and return (first update step
    #   is written later to overlap I/O). Otherwise increment the previous
    #   buffer's step by 1 (log 1-based step), write it, update the progress
    #   bar / step-end hook with its step and mean loss, then move the
    #   current buffer to previous and clear current.
    """Writes previous buffered train metrics."""
    if self._prev_buffered_train_metrics is None:
      # skip the first step so we can overlap I/O with next step.
      self._prev_buffered_train_metrics = self._buffered_train_metrics
      self._buffered_train_metrics = None
      return
    # increment the step by one for logging purpose, because train_step is not
    # incremented until the next model update.
    self._prev_buffered_train_metrics.step += 1
    self._write_metrics(self._prev_buffered_train_metrics)
    self._may_update_pbar(
        self._tqdm_train_metrics,
        step=self._prev_buffered_train_metrics.step,
        loss=self._prev_buffered_train_metrics.loss,
    )
    self._prev_buffered_train_metrics = self._buffered_train_metrics
    self._buffered_train_metrics = None

  def _write_metrics(self, metrics_buffer: MetricsBuffer):
    # [FUNCTION DOC]
    # HELPER: Input: metrics_buffer. Logs its mean loss at its step and each
    #   additional metric aggregated by its op applied to the value list
    #   (jax arrays converted to float32 numpy, recursively through lists).
    def _to_np_array(v):
      # [FUNCTION DOC]
      # HELPER: nested. Converts a jax.Array to a float32 numpy array, maps
      #   lists recursively, and returns other values unchanged.
      if isinstance(v, jax.Array):
        return np.asarray(v, dtype=np.float32)
      elif isinstance(v, list):
        return [_to_np_array(x) for x in v]
      return v

    self._log_metrics(
        loss=metrics_buffer.loss,
        step=metrics_buffer.step,
        additional_metrics={
            k: op(_to_np_array(v))
            for k, (
                v,
                op,
            ) in metrics_buffer.additional_metrics.items()
        },
    )

  @contextlib.contextmanager
  def _switch_mode(self, mode: sft_metrics_logger.Mode):
    # [FUNCTION DOC]
    # HELPER: context manager temporarily setting self._mode to mode and
    #   restoring the original on exit (even on exceptions).
    original_mode = self._mode
    self._mode = mode
    try:
      yield
    finally:
      self._mode = original_mode

  @property
  def _tqdm_train_metrics(self) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: property listing metric names shown in the progress bar:
    #   ["loss", "perplexity", "learning_rate"].
    return ["loss", "perplexity", "learning_rate"]

  def _may_update_pbar(
      self,
      metrics: list[str],
      step: int | None = None,
      loss: ArrayLike | None = None,
  ):
    # [FUNCTION DOC]
    # HELPER: Inputs: metrics (names), step, loss. If a progress bar exists,
    #   update its metrics for the current mode and advance it by one. If
    #   training hooks exist and mode is TRAIN, call
    #   on_train_step_end(self, step, loss).
    """Updates the progress bar with the given metrics if available."""
    if self._pbar is not None:
      self._pbar.update_metrics(metrics, self._mode, ndigits=3)
      self._pbar.update()

    if self.training_hooks and self._mode == sft_metrics_logger.Mode.TRAIN:
      self.training_hooks.on_train_step_end(self, step, loss)

  def train(
      self,
      train_ds: Iterable[Any],
      eval_ds: Iterable[Any] | None = None,
      skip_jit: bool = False,
      *,
      cache_nnx_graph: bool = True,
  ) -> None:
    # [FUNCTION DOC]
    # Main training loop. Inputs: train_ds (iterable of batches), eval_ds
    #   (optional iterable, re-iterated for each eval), skip_jit,
    #   cache_nnx_graph (keyword-only, default True).
    # Steps: get step fns; if jitted, log compile cache size when new. If
    #   eval_ds is truthy run an initial eval. Create a progress bar if
    #   max_steps set. Call on_train_start. Loop: activate profiler at
    #   _iter_steps; fetch a batch from data_hooks or the iterator (on
    #   resume, unless externally managed, skip the first _iter_steps
    #   examples); stop when the batch is None/exhausted or (not externally
    #   managed) _train_steps >= max_steps. Prepare inputs, shard, throttle,
    #   on_train_step_start, compute perf tags (global step and mini batch
    #   derived from custom_checkpoint_metadata "global_step"/"role", micro
    #   batch = _iter_steps mod accumulation), run the train step inside
    #   perf spans, register the loss with the throttler, buffer loss and
    #   grad_norm, call _post_process_train_step(aux), increment
    #   _iter_steps. At each accumulation boundary: increment _train_steps,
    #   write delayed metrics, ask the checkpoint manager to save (policy
    #   decides), and run eval every eval_every_n_steps. Deactivate profiler.
    #   After the loop: wait for inflight work, log duration, on_train_end,
    #   and close() unless externally managed.
    # Exceptions: from steps, data, hooks, checkpointing.
    # Side effects: trains the model, writes metrics/checkpoints/traces.
    # KNOWN ISSUE: when eval_ds is a one-shot iterator it is exhausted after
    #   the first eval, so later evals see no data; the max_steps check runs
    #   after a batch was already pulled from the iterator.
    """Training loop."""
    logging.log_first_n(
        logging.INFO,
        f"Training with mesh: {pxla.thread_resources.env.physical_mesh}",
        1,
    )
    train_step, eval_step = self.jit_train_and_eval_step(
        skip_jit, cache_nnx_graph
    )
    if not skip_jit:
      cache_size = train_step.func.jitted_fn._cache_size()  # pytype: disable=attribute-error
      logging.log_if(
          logging.INFO,
          f"Compiled train_step cache size: {cache_size}",
          condition=cache_size not in self._jit_cache,
      )
      self._jit_cache.add(cache_size)

    if eval_ds:
      self._run_eval(eval_ds, eval_step)

    if self.config.max_steps is not None and self._pbar is None:
      self._pbar = progress_bar.ProgressBar(
          metrics_prefix=self.metrics_prefix,
          metrics_logger=self.metrics_logger,
          initial_steps=self._train_steps,
          max_steps=self.config.max_steps,
          description=self.config.pbar_description,
      )

    if self.training_hooks:
      self.training_hooks.on_train_start(self)

    train_iterator = iter(train_ds)
    index = 0
    last_step_completion_time = time.perf_counter()
    while True:
      self._prof.maybe_activate(self._iter_steps)
      with jax.profiler.StepTraceAnnotation("train", step_num=self._iter_steps):
        train_example = None
        if self.data_hooks:
          train_example = self.data_hooks.load_next_train_batch(self)
        else:
          try:
            train_example = next(train_iterator)
            if not self.is_managed_externally:
              # TODO(mridulsahu): Add support to restore the iterator state
              # instead of skipping the already trained examples.
              if index < self._iter_steps:
                # Skip the examples that are already trained.
                index += 1
                continue
            index += 1
          except StopIteration:
            pass

        if train_example is None:
          break

        # Stop training if max_steps is reached.
        if (
            not self.is_managed_externally
            and self.config.max_steps is not None
            and self._train_steps >= self.config.max_steps
        ):
          break

        train_example = self._prepare_inputs(train_example)
        train_example = sharding_utils.shard_input(
            train_example, self.config.data_sharding_axis
        )

        self._throttler.wait_for_next()
        if self.training_hooks:
          self.training_hooks.on_train_step_start(self)

        # Collect tags for the span
        metadata = self.custom_checkpoint_metadata()
        global_step = metadata.get("global_step")

        if global_step is not None:
          # Offset by 1 since global_step is incremented for checkpointing.
          global_step -= 1
          if global_step > 0:
            if self._mini_batch_size is None:
              self._mini_batch_size = max(1, self._train_steps // global_step)
            mini_batch = self._train_steps % self._mini_batch_size
          else:
            mini_batch = self._train_steps
        else:
          mini_batch = None
          global_step = None
        micro_batch = self._iter_steps % self.config.get_with_default(
            "gradient_accumulation_steps", 1
        )
        tags = {
            perf_constants.STEP: global_step,
            perf_constants.ROLE: metadata.get("role"),
            perf_constants.MICRO_BATCH: micro_batch,
            perf_constants.MINI_BATCH: mini_batch,
        }

        with self._perf_tracer.span(
            "peft_train_step",
            pxla.thread_resources.env.physical_mesh.devices,
        ) as span, self._perf_tracer_v2.span(
            perf_constants.PEFT_TRAIN,
            pxla.thread_resources.env.physical_mesh.devices,
            tags=tags,
        ) as span_v2:
          train_loss, aux, grad_norm = train_step(train_example)
          span.device_end([train_loss])
          span_v2.async_end([train_loss])

        self._throttler.add_computation(train_loss)
        self._buffered_train_metrics = self._buffer_metrics(
            self._buffered_train_metrics,
            loss=train_loss,
            step=self._train_steps,
            additional_metrics={"grad_norm": (grad_norm, np.mean)},
        )
        # NB: put this after self._buffer_metrics is important
        self._post_process_train_step(aux)
        self._iter_steps += 1

        if (
            self._iter_steps
            % self.config.get_with_default("gradient_accumulation_steps", 1)
            == 0
        ):
          self._train_steps += 1
          self._write_train_metrics()

          # Checkpoint frequency is configured by checkpointing_options.
          self.checkpoint_manager.save(
              self._train_steps,
              self.model,
              self.optimizer,
              save_only_lora_params=self._lora_enabled,
              custom_metadata=self.custom_checkpoint_metadata(),
          )

          if (
              eval_ds
              and self._train_steps % self.config.eval_every_n_steps == 0
          ):
            self._run_eval(eval_ds, eval_step)

      self._prof.maybe_deactivate(self._iter_steps)

    self._throttler.wait_for_all()
    logging.info(
        "Train loop finished in: %.4f seconds",
        time.perf_counter() - last_step_completion_time,
    )
    if self.training_hooks:
      self.training_hooks.on_train_end(self)
    if not self.is_managed_externally:
      self.close()

  def _save_last_checkpoint(self):
    # [FUNCTION DOC]
    # HELPER: forces a checkpoint save at _train_steps (with LoRA-only
    #   params if LoRA) when no checkpoint exists or the latest saved step is
    #   older. No custom metadata is attached to this final save.
    last_saved_step = self.checkpoint_manager.latest_step()
    if last_saved_step is None or last_saved_step < self._train_steps:
      self.checkpoint_manager.save(
          self._train_steps,
          self.model,
          self.optimizer,
          save_only_lora_params=self._lora_enabled,
          force=True,
      )

  @property
  def train_steps(self) -> int:
    """Returns the number of train steps taken."""
    # [FUNCTION DOC]
    # Property: number of optimizer updates performed (_train_steps).
    return self._train_steps

  @property
  def iter_steps(self) -> int:
    """Returns the number of iterator steps taken."""
    # [FUNCTION DOC]
    # Property: number of micro steps (batches) processed (_iter_steps).
    return self._iter_steps

  def custom_checkpoint_metadata(self) -> dict[str, Any]:
    """Override this function to return the custom metadata for the checkpoint manager."""
    # [FUNCTION DOC]
    # Override hook returning a dict saved as checkpoint custom metadata and
    # used for perf tags ("global_step", "role"). Default: empty dict.
    return {}

  def close(self):
    """Closes the trainer and its associated resources.

    This includes writing any buffered metrics, saving the last checkpoint,
    and closing the checkpoint manager and metrics logger.
    """
    # [FUNCTION DOC]
    # Finalizes: writes pending delayed train metrics, saves a last
    #   checkpoint if needed, closes checkpoint manager (waits for async
    #   saves) and metrics logger (closes backends, clears jax.monitoring
    #   listeners), and closes the progress bar.
    # KNOWN ISSUE: metrics of a partially accumulated final step move to
    #   the "previous" buffer and are never written.
    self._write_train_metrics()
    self._save_last_checkpoint()
    self.checkpoint_manager.close()
    self.metrics_logger.close()
    if self._pbar is not None:
      self._pbar.close()
      self._pbar = None

  def _run_eval(
      self,
      eval_ds: Iterable[Any],
      eval_step_fn: Callable[..., Any],
  ) -> None:
    # [FUNCTION DOC]
    # HELPER: evaluation loop. Inputs: eval_ds, eval_step_fn (batch ->
    #   (loss, aux)). In EVAL mode: iterate batches (data_hooks or
    #   iterator) until None/exhausted; prepare, shard, on_eval_step_start,
    #   run step, stop_gradient the loss, buffer it at the current train
    #   step, _post_process_eval_step(aux), accumulate sum of losses. If no
    #   batches, warn and return. Else write eval metrics (mean loss,
    #   perplexity of mean loss, aux metrics), log the running eval loss /
    #   perplexity averages from the logger, clear the buffer and call
    #   on_eval_step_end(self, summed eval loss).
    # KNOWN ISSUE: the logged "eval loss" is the mean over all evals so far
    #   (MetricsLogger history), not just this eval; the hook receives the
    #   sum, not the mean, of batch losses.
    """Runs evaluation loop."""
    logging.info("Running evaluation on train step %d.", self._train_steps)
    eval_iterator = iter(eval_ds)
    with self._switch_mode(sft_metrics_logger.Mode.EVAL):
      eval_loss, eval_steps = 0, 0
      while True:
        if self.data_hooks:
          eval_example = self.data_hooks.load_next_eval_batch(self)
        else:
          try:
            eval_example = next(eval_iterator)
          except StopIteration:
            eval_example = None
        if eval_example is None:
          break
        eval_example = self._prepare_inputs(eval_example)
        eval_example = sharding_utils.shard_input(
            eval_example, self.config.data_sharding_axis
        )
        if self.training_hooks:
          self.training_hooks.on_eval_step_start(self)
        loss, aux = eval_step_fn(eval_example)
        loss = jax.lax.stop_gradient(loss)
        self._buffered_eval_metrics = self._buffer_metrics(
            self._buffered_eval_metrics,
            loss=loss,
            step=self._train_steps,
        )
        self._post_process_eval_step(aux)
        eval_loss += loss
        eval_steps += 1

      if eval_steps == 0:
        logging.warning(
            "No eval examples found. Skipping eval metrics logging."
        )
        return

      self._write_metrics(self._buffered_eval_metrics)
      logging.info(
          "Train step %d eval loss: %f - eval perplexity: %f",
          self._train_steps,
          self.metrics_logger.get_metric(self.metrics_prefix, "loss", "eval"),
          self.metrics_logger.get_metric(
              self.metrics_prefix, "perplexity", "eval"
          ),
      )
      self._buffered_eval_metrics = None
      if self.training_hooks:
        self.training_hooks.on_eval_step_end(self, eval_loss)


def _default_loss_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
    images: jax.Array | None = None,
) -> ArrayLike:
  # [FUNCTION DOC]
  # HELPER: default next-token NLL loss. Inputs: model, input_tokens [B, L],
  #   input_mask [B, L], positions, attention_mask, images (optional; only
  #   passed when not None).
  # Steps: logits, _ = model(input_tokens, positions, None (cache),
  #   attention_mask[, images=...]); drop the last logit position; targets
  #   = tokens shifted left by one with mask shifted likewise; one-hot the
  #   targets, zero rows where the target mask is 0; loss = -sum(log_softmax
  #   (logits) * one_hot) / (number of valid targets + 1e-8).
  # Output: scalar mean token NLL. No side effects.
  # KNOWN ISSUE: the default gen_model_input_fn is the identity, but this
  #   loss needs positions and attention_mask that TrainingInput lacks, so
  #   callers must supply a gen_model_input_fn that builds them.
  """Default loss function for PEFT training."""
  # Weird kwargs workaround because not all models support `images` right now.
  kwargs = {} if images is None else {"images": images}
  logits, _ = model(input_tokens, positions, None, attention_mask, **kwargs)

  # Exclude the last step as it does not appear in the targets.
  logits = logits[:, :-1, :]
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  # Convert the target labels to one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]

  # Define the normalization factor.
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

  # Return the negative log likelihood (NLL) loss.
  # Equivalent to: optax.softmax_cross_entropy(logits, one_hot).mean()
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor
