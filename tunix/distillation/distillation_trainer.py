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
# Distillation trainer built on tunix.sft.peft_trainer.PeftTrainer.
# Behavior: trains a student model to mimic a teacher. On every training
#   batch the teacher's outputs are computed by the strategy (in
#   _prepare_inputs), attached to the batch as teacher_output, and the loss
#   is delegated to strategy.get_train_loss; in eval mode no teacher outputs
#   are computed and strategy.get_eval_loss is used. Losses are always
#   normalized to a (loss, metrics-dict) pair and the parent trainer is told
#   the loss has auxiliary output.
# Defines: TrainingConfig (unchanged PeftTrainer config), TrainingInput
#   (PeftTrainer input + teacher_output), DistillationTrainer.
# Side effects: none at import.
"""Distillation trainer."""
import dataclasses
from typing import Any, Callable

import flax
from flax import nnx
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import optax
from tunix.distillation import strategies
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from typing_extensions import override


@dataclasses.dataclass(slots=True, kw_only=True)
class TrainingConfig(peft_trainer.TrainingConfig):
  # [CLASS DOC]
  # Distillation training config. A keyword-only, slotted dataclass that
  # adds no fields to peft_trainer.TrainingConfig; it exists as a distinct
  # type for the distillation API. See peft_trainer.TrainingConfig for fields.
  """Distillation training config."""


@flax.struct.dataclass(frozen=True)
class TrainingInput(peft_trainer.TrainingInput):
  # [CLASS DOC]
  # Immutable flax struct dataclass extending peft_trainer.TrainingInput
  # (input_tokens, input_mask) with one extra pytree field teacher_output
  # (default None) holding whatever the strategy's get_teacher_outputs
  # returned (e.g. teacher logits or captured features). Being a pytree, it
  # can be passed through jitted train steps.
  """Distillation training input."""

  teacher_output: Any = None


class DistillationTrainer(peft_trainer.PeftTrainer):
  # [CLASS DOC]
  # Trainer that distills a teacher into a student using a BaseStrategy.
  # State: strategy, teacher_model (frozen; never optimized), model (the
  #   student, managed by PeftTrainer), loss_fn/eval_loss_fn bound to
  #   get_train_loss/get_eval_loss, _has_aux forced True, and
  #   gen_model_input_fn which maps a batch to a dict with keys "inputs"
  #   (model kwargs) and "teacher_output".
  # Lifecycle: constructor lets the strategy pre-process models; close()
  #   lets the strategy post-process (e.g. unwrap) them afterwards.
  # Custom loss functions are not allowed (with_loss_fn raises).
  """Distillation trainer."""

  def __init__(
      self,
      student_model: nnx.Module,
      teacher_model: nnx.Module,
      strategy: strategies.BaseStrategy,
      optimizer: optax.GradientTransformation,
      training_config: TrainingConfig,
  ):
    # [FUNCTION DOC]
    # Builds the trainer.
    # Inputs: student_model, teacher_model (nnx modules), strategy
    #   (BaseStrategy), optimizer (optax transform), training_config.
    # Steps: (1) call strategy.pre_process_models(student, teacher) and use
    #   the returned pair (may be wrapped/augmented models); (2) initialize
    #   PeftTrainer with the processed student, optimizer and config;
    #   (3) store strategy and teacher; (4) set loss_fn = get_train_loss and
    #   eval_loss_fn = get_eval_loss; (5) set _has_aux True since losses
    #   return (loss, metrics); (6) default gen_model_input_fn returns
    #   {"inputs": {"input_tokens", "input_mask"} from the batch,
    #    "teacher_output": batch.teacher_output if the attribute exists,
    #   else None}.
    # Outputs: none. Exceptions: whatever pre_process_models or the parent
    #   constructor raise. Side effects: may mutate/wrap the given models.
    """Initializes the DistillationTrainer.

    Args:
        student_model: The student model to train.
        teacher_model: The teacher model to use for distillation.
        strategy: The distillation strategy to use.
        optimizer: The optimizer to use for training.
        training_config: The training config.
    """
    student_model, teacher_model = strategy.pre_process_models(
        student_model, teacher_model
    )
    super().__init__(student_model, optimizer, training_config)
    self.strategy = strategy
    self.teacher_model = teacher_model
    self.loss_fn = self.get_train_loss
    self.eval_loss_fn = self.get_eval_loss

    # Since distillation strategies return (loss, metrics),
    # we explicitly enable auxiliary metric handling.
    self._has_aux = True

    self.gen_model_input_fn = lambda x: {
        "inputs": {"input_tokens": x.input_tokens, "input_mask": x.input_mask},
        "teacher_output": (
            x.teacher_output if hasattr(x, "teacher_output") else None
        ),
    }

  @override
  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, ArrayLike]]
  ) -> "DistillationTrainer":
    # [FUNCTION DOC]
    # Overrides the batch-to-model-input mapping. Inputs: gen_model_input_fn,
    #   a callable taking a batch and returning the model kwargs dict.
    # Behavior: stores a wrapper that returns {"inputs": user_fn(batch),
    #   "teacher_output": batch.teacher_output if present else None}, so the
    #   teacher output is always forwarded to the loss alongside inputs.
    # Output: self (for chaining). Exceptions: none. Side effect: replaces
    #   self.gen_model_input_fn.
    # KNOWN ISSUE: _prepare_inputs rebuilds TrainingInput with only
    #   input_tokens, input_mask and teacher_output, so any extra batch
    #   fields read by a custom gen_model_input_fn are dropped before the
    #   loss step (the fn is applied again to the rebuilt batch).
    self.gen_model_input_fn = lambda x: {
        "inputs": gen_model_input_fn(x),
        "teacher_output": (
            x.teacher_output if hasattr(x, "teacher_output") else None
        ),
    }
    return self

  @override
  def with_loss_fn(
      self,
      loss_fn: Callable[..., ArrayLike | tuple[ArrayLike, Any]],
      has_aux: bool = False,
  ) -> "DistillationTrainer":
    # [FUNCTION DOC]
    # Disallowed override. Inputs: loss_fn, has_aux (ignored).
    # Always raises NotImplementedError telling the user to define the loss
    #   via the strategy. No output, no side effects.
    raise NotImplementedError(
        "with_loss_fn is not supported for distillation. Use the strategy to"
        " define the loss."
    )

  @override
  def _prepare_inputs(self, input_data: TrainingInput) -> TrainingInput:
    # [FUNCTION DOC]
    # Per-batch preprocessing hook called by PeftTrainer before the step.
    # Input: input_data (a TrainingInput-like batch).
    # Behavior: compute model inputs as gen_model_input_fn(batch)["inputs"].
    #   If the trainer's current mode is EVAL, teacher_output is None;
    #   otherwise teacher_output = strategy.get_teacher_outputs(teacher_model,
    #   inputs) (runs the teacher forward pass).
    # Output: a new TrainingInput with the batch's input_tokens, input_mask
    #   and the computed teacher_output.
    # Exceptions: propagates teacher/strategy errors; batch must have
    #   input_tokens and input_mask attributes.
    # Side effects: teacher forward pass (may sow intermediates on teacher).
    inputs = self.gen_model_input_fn(input_data)["inputs"]
    if self._mode == metrics_logger.Mode.EVAL:
      teacher_output = None
    else:
      teacher_output = self.strategy.get_teacher_outputs(
          self.teacher_model, inputs
      )

    return TrainingInput(
        input_tokens=input_data.input_tokens,
        input_mask=input_data.input_mask,
        teacher_output=teacher_output,
    )

  def _standardize_loss_output(
      self, output: Any
  ) -> tuple[ArrayLike, dict[str, Any]]:
    # HELPER:
    # [FUNCTION DOC]
    # Normalizes a strategy loss result. Input: output, either a tuple or a
    #   bare scalar loss. Output: the tuple unchanged if output is a tuple
    #   (assumed (loss, metrics)), otherwise (output, empty dict).
    # No exceptions, no side effects. Tuples of other lengths are passed
    #   through unvalidated.
    """Adapts strategy output to always be (loss, metrics).

    This helper function ensures that the return value from a strategy's
    loss computation is consistent, regardless of whether the strategy
    returns just a scalar loss or a tuple of (loss, metrics).

    Args:
        output: The raw output from a strategy's `compute_loss` or
          `compute_eval_loss` method. Can be a single scalar (loss) or a tuple.

    Returns:
        A tuple containing:
          - loss: The scalar loss value (ArrayLike).
          - metrics: A dictionary of auxiliary metrics. If the input was just
            a scalar, this will be an empty dictionary.
    """
    if isinstance(output, tuple):
      return output
    # If strategy returned just a scalar loss, wrap it with empty metrics
    return output, {}

  def get_train_loss(
      self,
      model: nnx.Module,
      teacher_output: Any,
      inputs: dict[str, ArrayLike],
  ) -> tuple[ArrayLike, dict[str, Any]]:
    # [FUNCTION DOC]
    # Training loss used as loss_fn. Inputs: model (student), teacher_output
    #   (from the batch), inputs (model kwargs dict). These are produced by
    #   gen_model_input_fn's "inputs"/"teacher_output" keys unpacked by the
    #   parent trainer.
    # Behavior: calls strategy.get_train_loss(model, teacher_output, inputs)
    #   and normalizes it to (loss, metrics).
    # Output: (scalar loss, metrics dict). Exceptions: from the strategy.
    # Side effects: none beyond the strategy's forward pass.
    output = self.strategy.get_train_loss(model, teacher_output, inputs)
    return self._standardize_loss_output(output)

  def get_eval_loss(
      self,
      model: nnx.Module,
      teacher_output: Any,
      inputs: dict[str, ArrayLike],
  ) -> tuple[ArrayLike, dict[str, Any]]:
    # [FUNCTION DOC]
    # Evaluation loss used as eval_loss_fn. Inputs: model, teacher_output
    #   (ignored; it is None in eval), inputs. Calls
    #   strategy.get_eval_loss(model, inputs) and normalizes to
    #   (loss, metrics). Exceptions: from the strategy. No side effects.
    del teacher_output  # Not computed in eval.
    output = self.strategy.get_eval_loss(model, inputs)
    return self._standardize_loss_output(output)

  def close(self):
    # [FUNCTION DOC]
    # Finalizes training. Calls PeftTrainer.close() (flushes/closes
    #   checkpointing, metrics, etc.), then replaces self.model and
    #   self.teacher_model with strategy.post_process_models(model, teacher)
    #   (e.g. unwrapping sowed modules or stripping projection layers).
    # Inputs: none. Output: none. Exceptions: from parent close or strategy.
    # Side effects: closes resources and mutates model attributes.
    super().close()
    self.model, self.teacher_model = self.strategy.post_process_models(
        self.model, self.teacher_model
    )
