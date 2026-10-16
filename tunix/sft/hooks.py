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
# Abstract hook interfaces that PeftTrainer calls during training.
# Defines TrainingHooks (lifecycle callbacks for train/eval steps) and
#   DataHooks (pluggable batch loading). ABC/abstractmethod are local
#   aliases; PeftTrainer is aliased to typing.Any to avoid a circular import
#   (annotations reference it only as strings). No side effects at import.
"""Hooks for training and data loading."""

import abc
from typing import Any

ABC = abc.ABC
abstractmethod = abc.abstractmethod
PeftTrainer = Any


class TrainingHooks(ABC):
  # [CLASS DOC]
  # Abstract callback set invoked by the trainer. Every method is abstract,
  # so subclasses must implement all six (bodies may be no-ops). Each
  # receives the trainer instance as train_ctx for inspection/mutation.
  """Hooks to be used for training."""

  @abstractmethod
  def on_train_start(self, train_ctx: "PeftTrainer.PeftTrainer"):
    # [FUNCTION DOC]
    # Abstract. Called once before the training loop. Input: train_ctx
    # (trainer). Output: none (base body does nothing).
    """Called at the beginning of training."""
    pass

  @abstractmethod
  def on_train_end(self, train_ctx: "PeftTrainer.PeftTrainer"):
    # [FUNCTION DOC]
    # Abstract. Called once after the training loop ends. Input: train_ctx.
    # Output: none.
    """Called at the end of training."""
    pass

  @abstractmethod
  def on_train_step_start(self, train_ctx: "PeftTrainer.PeftTrainer"):
    # [FUNCTION DOC]
    # Abstract. Called before each training step. Input: train_ctx.
    # Output: none.
    """Called at the beginning of a training step."""
    pass

  @abstractmethod
  def on_train_step_end(
      self,
      train_ctx: "PeftTrainer.PeftTrainer",
      train_step: int,
      train_loss: float,
  ):
    # [FUNCTION DOC]
    # Abstract. Called after each training step. Inputs: train_ctx,
    # train_step (int step counter), train_loss (float loss of the step).
    # Output: none.
    """Called at the end of a training step."""
    pass

  @abstractmethod
  def on_eval_step_start(self, train_ctx: "PeftTrainer.PeftTrainer"):
    # [FUNCTION DOC]
    # Abstract. Called before each evaluation step. Input: train_ctx.
    # Output: none.
    """Called at the beginning of an evaluation step."""
    pass

  @abstractmethod
  def on_eval_step_end(
      self, train_ctx: "PeftTrainer.PeftTrainer", eval_loss: float
  ):
    # [FUNCTION DOC]
    # Abstract. Called after evaluation. Inputs: train_ctx, eval_loss
    # (float). Output: none.
    """Called at the end of an evaluation step."""
    pass


class DataHooks(ABC):
  # [CLASS DOC]
  # Abstract data-loading hooks letting an external loader supply batches
  # to the trainer instead of an iterator. Both methods are abstract.
  """Hooks to wire in external data loader and processing logic."""

  @abstractmethod
  def load_next_train_batch(self, train_ctx: "PeftTrainer.PeftTrainer") -> Any:
    # [FUNCTION DOC]
    # Abstract. Input: train_ctx. Output: next training batch (any type the
    # trainer accepts). Base body raises NotImplementedError.
    """Loads the next batch of data for training."""
    raise NotImplementedError()

  @abstractmethod
  def load_next_eval_batch(self, train_ctx: "PeftTrainer.PeftTrainer") -> Any:
    # [FUNCTION DOC]
    # Abstract. Input: train_ctx. Output: next evaluation batch. Base body
    # raises NotImplementedError.
    """Loads the next batch of data for evaluation."""
    raise NotImplementedError()
