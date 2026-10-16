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
# Unit tests for tunix.distillation.strategies.feature_projection
# .FeatureProjectionStrategy: a distillation strategy that sows features from
# layers of type feature_layer, learns a projection from student feature
# shape to teacher feature shape (built using dummy_input and rngs), and
# combines a feature regression loss weighted alpha with the task loss
# weighted (1 - alpha); alpha must be in [0, 1] else ValueError.
# Module setup disables jax_threefry_partitionable.
# KNOWN ISSUE: the module docstring says 'feature pooling strategy' but the
# tests target the feature projection strategy.
"""Tests for feature pooling strategy."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy.testing as npt
from tunix.distillation.strategies import feature_projection

jax.config.update("jax_threefry_partitionable", False)

FeatureProjectionStrategy = feature_projection.FeatureProjectionStrategy


class DummyModel(nnx.Module):
  # [CLASS DOC]
  # HELPER: toy nnx model with one Linear(in_features -> out_features) layer
  # seeded with Rngs(0).

  def __init__(self, in_features=3, out_features=10):
    # [FUNCTION DOC]
    # HELPER: creates the Linear layer.
    self.linear = nnx.Linear(
        in_features=in_features, out_features=out_features, rngs=nnx.Rngs(0)
    )

  def __call__(self, x):
    # [FUNCTION DOC]
    # HELPER: returns linear(x).
    x = self.linear(x)
    return x


class FeatureProjectionStrategyTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for FeatureProjectionStrategy.

  def model_forward_fn(self, model: nnx.Module, x: jax.Array):
    # [FUNCTION DOC]
    # HELPER: forward function; returns model(x).
    return model(x)

  def get_labels_fn(self, x: jax.Array):
    # [FUNCTION DOC]
    # HELPER: labels function; returns a length-10 vector filled with mean(x).
    return jnp.ones(10) * jnp.mean(x)

  @parameterized.named_parameters(
      ("alpha_half", 0.5),
      ("alpha_zero", 0.0),
      ("alpha_one", 1.0),
  )
  def test_init_valid(self, alpha):
    # [FUNCTION DOC]
    # Verifies construction stores alpha for 0, 0.5 and 1.
    strategy = FeatureProjectionStrategy(
        student_forward_fn=self.model_forward_fn,
        teacher_forward_fn=self.model_forward_fn,
        labels_fn=self.get_labels_fn,
        alpha=alpha,
        feature_layer=nnx.Linear,
        dummy_input={"x": jnp.ones(3)},
        rngs=nnx.Rngs(0),
    )
    self.assertIsInstance(strategy, FeatureProjectionStrategy)
    self.assertEqual(strategy.alpha, alpha)

  @parameterized.named_parameters(
      ("alpha_neg", -0.1),
      ("alpha_over", 1.1),
  )
  def test_init_invalid(self, alpha):
    # [FUNCTION DOC]
    # Verifies ValueError for alpha outside [0, 1].
    with self.assertRaises(ValueError):
      FeatureProjectionStrategy(
          student_forward_fn=self.model_forward_fn,
          teacher_forward_fn=self.model_forward_fn,
          labels_fn=self.get_labels_fn,
          alpha=alpha,
          feature_layer=nnx.Linear,
          dummy_input={"x": jnp.ones(3)},
          rngs=nnx.Rngs(0),
      )

  def test_get_train_loss(self):
    # [FUNCTION DOC]
    # Verifies get_train_loss with a 3->10 student and 3->20 teacher after
    # pre_process_models equals reference 574.2112 (rtol 1e-5).
    strategy = FeatureProjectionStrategy(
        student_forward_fn=self.model_forward_fn,
        teacher_forward_fn=self.model_forward_fn,
        labels_fn=self.get_labels_fn,
        feature_layer=nnx.Linear,
        dummy_input={"x": jnp.ones(3)},
        rngs=nnx.Rngs(0),
    )
    inputs = {
        "x": jnp.array([10.0, 11.0, 12.0]),
    }
    student_model = DummyModel()
    teacher_model = DummyModel(out_features=20)
    student_model, teacher_model = strategy.pre_process_models(
        student_model, teacher_model
    )
    expected_loss = 574.2112

    teacher_output = strategy.get_teacher_outputs(
        teacher_model=teacher_model, inputs=inputs
    )
    computed_loss = strategy.get_train_loss(
        student_model=student_model,
        teacher_output=teacher_output,
        inputs=inputs,
    )

    npt.assert_allclose(computed_loss, expected_loss, rtol=1e-5)

  def test_get_eval_loss(self):
    # [FUNCTION DOC]
    # Verifies get_eval_loss of the preprocessed student equals reference
    # 1806.01306152 (rtol 1e-5).
    strategy = FeatureProjectionStrategy(
        student_forward_fn=self.model_forward_fn,
        teacher_forward_fn=self.model_forward_fn,
        labels_fn=self.get_labels_fn,
        feature_layer=nnx.Linear,
        dummy_input={"x": jnp.ones(3)},
        rngs=nnx.Rngs(0),
    )
    inputs = {
        "x": jnp.array([10.0, 11.0, 12.0]),
    }
    student_model = DummyModel()
    teacher_model = DummyModel()
    student_model, _ = strategy.pre_process_models(student_model, teacher_model)
    expected_loss = 1806.01306152

    computed_loss = strategy.get_eval_loss(
        student_model=student_model, inputs=inputs
    )

    npt.assert_allclose(computed_loss, expected_loss, rtol=1e-5)


if __name__ == "__main__":
  absltest.main()
