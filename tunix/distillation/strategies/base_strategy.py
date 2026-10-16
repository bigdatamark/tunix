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
# Base contract for distillation strategies.
# Defines ModelForwardCallable (protocol: callable taking an nnx model plus
#   arbitrary args/kwargs and returning some output R) and BaseStrategy, the
#   abstract class every strategy extends. ABC/abstractmethod are local
#   aliases of abc.ABC/abc.abstractmethod; R is a TypeVar.
# Flow used by DistillationTrainer: get_teacher_outputs (no gradient) on
#   each training batch, then get_train_loss (student forward + labels +
#   compute_loss) or get_eval_loss (student forward + labels +
#   compute_eval_loss). pre/post_process_models let subclasses wrap/unwrap
#   models. No side effects at import.
"""Implements Distillation Strategies."""

import abc
from typing import Any, Callable, Protocol, TypeVar

from flax import nnx
import jax

ABC = abc.ABC
abstractmethod = abc.abstractmethod
R = TypeVar("R")


class ModelForwardCallable(Protocol[R]):
  # [CLASS DOC]
  # Typing protocol for model forward functions: called as
  # fn(model, *args, **kwargs) and returns a value of generic type R (e.g.
  # logits array). Used purely for static typing; no runtime behavior.

  def __call__(self, model: nnx.Module, *args: Any, **kwargs: Any) -> R:
    # [FUNCTION DOC]
    # Protocol signature only (body is an ellipsis). Inputs: model (nnx
    # module), any positional/keyword args. Output: R. No behavior.
    ...


class BaseStrategy(ABC):
  # [CLASS DOC]
  # Abstract distillation strategy. Holds three jitted callables:
  #   _student_forward_fn and _teacher_forward_fn (model, **inputs) -> output
  #   and _labels_fn (**inputs) -> labels, all wrapped with nnx.jit.
  # Subclasses must implement compute_loss and compute_eval_loss (both
  #   return a scalar loss or (loss, metrics dict)); they may override
  #   pre_process_models/post_process_models and the get_* methods.
  # KNOWN ISSUE: the docstring says only compute_loss must be implemented,
  #   but compute_eval_loss is also abstract, so both are required.
  """Abstract Base Class for all distillation strategies.

  Defines the common interface for computing the distillation loss. Concrete
  strategies must implement the `compute_loss` method.
  """

  def __init__(
      self,
      student_forward_fn: ModelForwardCallable[Any],
      teacher_forward_fn: ModelForwardCallable[Any],
      labels_fn: Callable[..., jax.Array],
  ):
    # [FUNCTION DOC]
    # Inputs: student_forward_fn, teacher_forward_fn (ModelForwardCallable),
    #   labels_fn (callable from model-input kwargs to a labels array).
    # Behavior: stores each wrapped in nnx.jit as _student_forward_fn,
    #   _teacher_forward_fn and _labels_fn. Output: none. Exceptions: none at
    #   construction (jit errors surface on first call). Side effects: none.
    """Initializes the BaseStrategy.

    Args:
      student_forward_fn: Function to compute student model outputs.
      teacher_forward_fn: Function to compute teacher model outputs.
      labels_fn: Function to compute labels from model inputs.
    """
    self._student_forward_fn = nnx.jit(student_forward_fn)
    self._teacher_forward_fn = nnx.jit(teacher_forward_fn)
    self._labels_fn = nnx.jit(labels_fn)

  @abstractmethod
  def compute_loss(
      self, student_output: Any, teacher_output: Any, labels: jax.Array
  ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
    # [FUNCTION DOC]
    # Abstract. Inputs: student_output, teacher_output (as returned by the
    #   forward fns), labels array. Output: scalar training loss, or tuple of
    #   (loss, dict of metric name -> array). Must be implemented by subclass.
    """Computes the distillation loss based on model outputs and labels."""

  @abstractmethod
  def compute_eval_loss(
      self, student_output: Any, labels: jax.Array
  ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
    # [FUNCTION DOC]
    # Abstract. Inputs: student_output, labels. Output: scalar evaluation
    #   loss or (loss, metrics dict); no teacher output is available in eval.
    # KNOWN ISSUE: docstring repeats "distillation loss" but eval has no
    #   teacher, so this is normally a pure task loss.
    """Computes the distillation loss based on model outputs and labels."""

  def pre_process_models(
      self,
      student_model: nnx.Module,
      teacher_model: nnx.Module,
  ) -> tuple[nnx.Module, nnx.Module]:
    # [FUNCTION DOC]
    # Hook called once by the trainer before training. Inputs: student and
    #   teacher nnx modules. Default: returns them unchanged as
    #   (student, teacher). Subclasses may wrap or augment models. No
    #   exceptions or side effects by default.
    """Pre-processes the models to prepare for distillation.

    This method is called before the distillation process. It can be used to
    modify the models to prepare them for distillation.

    Args:
      student_model: Original student model.
      teacher_model: Original teacher model.

    Returns:
      Student model processed for distillation.
      Teacher model processed for distillation.
    """
    return student_model, teacher_model

  def post_process_models(
      self,
      student_model: nnx.Module,
      teacher_model: nnx.Module,
  ) -> tuple[nnx.Module, nnx.Module]:
    # [FUNCTION DOC]
    # Hook called on trainer close to undo pre_process_models. Inputs:
    #   processed student and teacher. Default: returns them unchanged as
    #   (student, teacher). No exceptions or side effects by default.
    """Post-processes the models after distillation.

    This method usually reverts the changes made in `pre_process_models`.

    Args:
      student_model: Pre processed student model.
      teacher_model: Pre processed teacher model.

    Returns:
      Original student model.
      Original processed teacher model.
    """
    return student_model, teacher_model

  def get_teacher_outputs(
      self,
      teacher_model: nnx.Module,
      inputs: dict[str, jax.Array],
  ) -> Any:
    # [FUNCTION DOC]
    # Runs the jitted teacher forward fn as fn(teacher_model, **inputs) and
    #   applies jax.lax.stop_gradient to the (pytree) result.
    # Inputs: teacher_model, inputs dict of model kwargs. Output: teacher
    #   output with gradients stopped. Exceptions: from the forward fn.
    """Computes the teacher model outputs."""
    output = self._teacher_forward_fn(teacher_model, **inputs)
    output = jax.lax.stop_gradient(output)
    return output

  def get_student_outputs(
      self,
      student_model: nnx.Module,
      inputs: dict[str, jax.Array],
  ) -> Any:
    # [FUNCTION DOC]
    # Runs the jitted student forward fn as fn(student_model, **inputs) and
    #   returns its output (gradients flow). Inputs: student model, inputs
    #   dict. Exceptions: from the forward fn.
    """Computes the student model outputs."""
    return self._student_forward_fn(student_model, **inputs)

  def get_train_loss(
      self,
      student_model: nnx.Module,
      teacher_output: Any,
      inputs: dict[str, jax.Array],
  ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
    # [FUNCTION DOC]
    # Training loss. Inputs: student_model, teacher_output (precomputed),
    #   inputs dict. Steps: student_output = get_student_outputs; labels =
    #   _labels_fn(**inputs); return compute_loss(student_output=...,
    #   teacher_output=..., labels=...) by keyword.
    # Output: loss or (loss, metrics). Exceptions: from callees.
    """Computes the distillation loss."""
    student_output = self.get_student_outputs(student_model, inputs)
    labels = self._labels_fn(**inputs)
    return self.compute_loss(
        student_output=student_output,
        teacher_output=teacher_output,
        labels=labels,
    )

  def get_eval_loss(
      self,
      student_model: nnx.Module,
      inputs: dict[str, jax.Array],
  ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
    # [FUNCTION DOC]
    # Eval loss. Inputs: student_model, inputs dict. Steps: student forward,
    #   stop_gradient on its output, labels = _labels_fn(**inputs), return
    #   compute_eval_loss(student_output=..., labels=...).
    # Output: loss or (loss, metrics). Exceptions: from callees.
    """Computes the task loss based on student model forward pass and labels."""
    student_output = self.get_student_outputs(student_model, inputs)
    student_output = jax.lax.stop_gradient(student_output)
    labels = self._labels_fn(**inputs)
    return self.compute_eval_loss(
        student_output=student_output,
        labels=labels,
    )
