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
# Utilities for feature-projection distillation: wrap a student model so
#   its sown intermediate features are passed through a learned linear
#   projection to the teacher's feature shape, plus setup/teardown helpers
#   used by FeatureProjectionStrategy. Relies on sowed_module for capture.
# Side effects at import: none.
"""Utilities for feature projection for distillation."""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.distillation.feature_extraction import sowed_module


class ModelWithFeatureProjection(nnx.Module):
  # [CLASS DOC]
  # nnx.Module wrapping a model whose target sublayers were already wrapped
  # with SowedModule. Attributes: model (the wrapped model) and
  # projection_layer (nnx.LinearGeneral contracting all axes of the stacked
  # feature array into feature_target_shape). Calling it returns
  # (model output, projected features).
  """Wraps a model with a projection layer for extracted features.

  This module takes a base model, extracts its intermediate features using
  `sowed_module`, and then applies a linear projection to these features.
  """

  def __init__(
      self,
      model: nnx.Module,
      feature_shape: int | tuple[int, ...],
      feature_target_shape: int | tuple[int, ...],
      *,
      rngs: nnx.Rngs,
  ):
    # [FUNCTION DOC]
    # Inputs: model (sowed nnx module); feature_shape (int or tuple; full
    #   shape of the stacked feature array); feature_target_shape (int or
    #   tuple); rngs (keyword-only).
    # Behavior: normalize an int feature_shape to a 1-tuple; store model;
    #   create LinearGeneral(in_features=feature_shape,
    #   out_features=feature_target_shape, axis=every input axis 0..n-1).
    # Output: none. Exceptions: from LinearGeneral. Side effects: draws
    #   params from rngs.
    # KNOWN ISSUE: feature_shape is the whole stacked array shape (layers,
    #   batch, seq, ...), so the projection is tied to the batch size and
    #   sequence length of the dummy input, and its kernel size is the
    #   product of both full shapes (potentially huge).
    """Initializes a ModelWithFeatureProjection.

    Args:
      model: The base sowed model to wrap.
      feature_shape: The shape of the features that will be extracted.
      feature_target_shape: The target shape of the features to project.
      rngs: The random number generator.
    """
    if isinstance(feature_shape, int):
      feature_shape = (feature_shape,)
    self.model = model
    self.projection_layer = nnx.LinearGeneral(
        feature_shape,
        feature_target_shape,
        axis=np.arange(len(feature_shape)),
        rngs=rngs,
    )

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forward pass. Inputs: any args/kwargs for the wrapped model.
    # Steps: output = model(*args, **kwargs); pop the model's
    #   nnx.Intermediate state; stack its leaves into one array; apply
    #   projection_layer. Output: (output, projected_features).
    # Exceptions: stacking/shape errors. Side effect: clears sown
    #   intermediates on the wrapped model.
    output = self.model(*args, **kwargs)
    model_features = sowed_module.pop_sowed_intermediate_outputs(self.model)
    model_features = jnp.stack(jax.tree.leaves(model_features))
    projected_features = self.projection_layer(model_features)
    return output, projected_features


def setup_models_with_feature_projection(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
    student_layer_to_capture: type[nnx.Module],
    teacher_layer_to_capture: type[nnx.Module],
    dummy_student_input: dict[str, Any],
    dummy_teacher_input: dict[str, Any],
    *,
    rngs: nnx.Rngs,
) -> tuple[ModelWithFeatureProjection, nnx.Module]:
  # [FUNCTION DOC]
  # Prepares student/teacher for feature projection distillation.
  # Inputs: student_model, teacher_model, student_layer_to_capture and
  #   teacher_layer_to_capture (module classes), dummy_student_input and
  #   dummy_teacher_input (kwargs dicts), rngs (keyword-only).
  # Steps: wrap matching sublayers of each model in place with SowedModule;
  #   run each model once on its dummy input (outputs discarded); pop and
  #   stack each model's sown features to obtain their shapes; build
  #   ModelWithFeatureProjection(student, student stacked shape, teacher
  #   stacked shape, rngs).
  # Output: (projected student wrapper, teacher model).
  # Exceptions: from wrapping, the forward passes, or stacking.
  # Side effects: mutates both models in place, runs forward passes.
  """Builds a student model with feature projection.

  This function takes a student and a teacher model, wraps them with sowed
  modules to capture intermediate features, and then builds a
  `ModelWithFeatureProjection` to project the student's features to match the
  teacher's features.

  Args:
    student_model: The student model.
    teacher_model: The teacher model.
    student_layer_to_capture: The type of layer to capture in the student model.
    teacher_layer_to_capture: The type of layer to capture in the teacher model.
    dummy_student_input: A dummy input for the student model.
    dummy_teacher_input: A dummy input for the teacher model.
    rngs: The random number generator.

  Returns:
    A tuple containing:
      - The student model with feature projection.
      - The teacher model.
  """
  sowed_module.wrap_model_with_sowed_modules(
      student_model, [student_layer_to_capture]
  )
  sowed_module.wrap_model_with_sowed_modules(
      teacher_model, [teacher_layer_to_capture]
  )
  _ = student_model(**dummy_student_input)
  _ = teacher_model(**dummy_teacher_input)

  student_features = sowed_module.pop_sowed_intermediate_outputs(student_model)
  student_features = jnp.stack(jax.tree.leaves(student_features))

  teacher_features = sowed_module.pop_sowed_intermediate_outputs(teacher_model)
  teacher_features = jnp.stack(jax.tree.leaves(teacher_features))

  student_model_with_feature_projection = ModelWithFeatureProjection(
      student_model,
      student_features.shape,
      teacher_features.shape,
      rngs=rngs,
  )
  return student_model_with_feature_projection, teacher_model


def remove_feature_projection_from_models(
    student_model: nnx.Module, teacher_model: nnx.Module
) -> tuple[nnx.Module, nnx.Module]:
  # [FUNCTION DOC]
  # Reverses setup_models_with_feature_projection. Inputs: student (maybe a
  #   ModelWithFeatureProjection), teacher. If student is the wrapper,
  #   replace it with its inner model (projection discarded); then remove
  #   SowedModule wrappers in place from both. Output: (student, teacher).
  # Exceptions: from unwrap_sowed_modules. Side effects: mutates models.
  """Returns original models from models modified with feature projection setup.

  Args:
    student_model: The student model with feature projection setup.
    teacher_model: The teacher model with feature projection setup.

  Returns:
    A tuple containing:
      - The original student model.
      - The original teacher model.
  """
  if isinstance(student_model, ModelWithFeatureProjection):
    student_model = student_model.model
  sowed_module.unwrap_sowed_modules(student_model)
  sowed_module.unwrap_sowed_modules(teacher_model)
  return student_model, teacher_model
