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
# Unit tests for tunix.distillation.strategies.logit.LogitStrategy: classic
# logit distillation combining alpha times the temperature-softened KL
# divergence between teacher and student (optionally restricted to the
# teacher's top-k logits via distill_top_k; 0 means full vocab) with
# (1 - alpha) times the hard-label cross entropy; temperature must be > 0 and
# alpha in [0, 1] else ValueError. compute_loss returns (loss, metrics with
# distill/soft_loss, distill/hard_loss, distill/kl_div, distill/teacher_loss);
# compute_eval_loss returns (task loss, empty dict).
# Module setup disables jax_threefry_partitionable.
"""Tests for distillation strategies."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy.testing as npt
from tunix.distillation.strategies import logit

jax.config.update("jax_threefry_partitionable", False)

LogitStrategy = logit.LogitStrategy


class DummyModel(nnx.Module):
  # [CLASS DOC]
  # HELPER: identity nnx module (returns its input).

  def __call__(self, x):
    # [FUNCTION DOC]
    # HELPER: returns x unchanged.
    return x


class LogitStrategyTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for LogitStrategy.

  def _get_dummy_logits(self, key, batch_size, num_classes):
    # [FUNCTION DOC]
    # HELPER: standard normal logits (batch_size, num_classes) from key.
    return jax.random.normal(key, (batch_size, num_classes))

  def _get_dummy_labels(self, key, batch_size, num_classes):
    # [FUNCTION DOC]
    # HELPER: random one-hot labels (batch_size, num_classes) from key.
    labels_int = jax.random.randint(key, (batch_size,), 0, num_classes)
    return jax.nn.one_hot(labels_int, num_classes)

  def dummy_fn(self):
    # [FUNCTION DOC]
    # HELPER: placeholder forward/labels function that does nothing.
    pass

  @parameterized.named_parameters(
      ("valid", 2.0, 0.5, 0),
      ("alpha_zero", 3.0, 0.0, 0),
      ("alpha_one", 2.0, 1.0, 0),
      ("top_k", 2.0, 0.5, 50),
  )
  def test_init_valid(self, temp, alpha, top_k):
    # [FUNCTION DOC]
    # Verifies valid construction stores temperature (as float), alpha and
    # distill_top_k.
    strategy = LogitStrategy(
        student_forward_fn=self.dummy_fn,
        teacher_forward_fn=self.dummy_fn,
        labels_fn=self.dummy_fn,
        temperature=temp,
        alpha=alpha,
        distill_top_k=top_k,
    )
    self.assertEqual(strategy.temperature, float(temp))
    self.assertEqual(strategy.alpha, alpha)
    self.assertEqual(strategy.distill_top_k, top_k)

  @parameterized.named_parameters(
      ("temp_zero", 0, 0.5),
      ("temp_neg", -1.0, 0.5),
      ("alpha_neg", 2.0, -0.1),
      ("alpha_over", 2.0, 1.1),
  )
  def test_init_invalid(self, temp, alpha):
    # [FUNCTION DOC]
    # Verifies ValueError for non-positive temperature or alpha outside [0, 1].
    with self.assertRaises(ValueError):
      LogitStrategy(
          student_forward_fn=self.dummy_fn,
          teacher_forward_fn=self.dummy_fn,
          labels_fn=self.dummy_fn,
          temperature=temp,
          alpha=alpha,
      )

  @parameterized.named_parameters(
      ("alpha_one", 1.0, 0.818179),
      ("alpha_half", 0.5, 1.530153),
      ("alpha_zero", 0.0, 2.242127),
  )
  def test_logit_distillation_compute_loss(self, alpha, expected_loss):
    # [FUNCTION DOC]
    # Verifies compute_loss with temperature 3 on seeded random logits/labels
    # matches reference losses for alpha 1, 0.5, 0 (tolerance 1e-4) and that the
    # four distill/* metrics are returned.
    s_key, t_key, l_key = jax.random.split(jax.random.key(0), 3)
    batch_size, num_classes = 4, 10
    student_logits = self._get_dummy_logits(s_key, batch_size, num_classes)
    teacher_logits = self._get_dummy_logits(t_key, batch_size, num_classes)
    labels = self._get_dummy_labels(l_key, batch_size, num_classes)
    temp = 3.0
    strategy = LogitStrategy(
        student_forward_fn=self.dummy_fn,
        teacher_forward_fn=self.dummy_fn,
        labels_fn=self.dummy_fn,
        temperature=temp,
        alpha=alpha,
    )

    # Now returns tuple (loss, metrics)
    computed_loss, metrics = strategy.compute_loss(
        student_output=student_logits,
        teacher_output=teacher_logits,
        labels=labels,
    )

    npt.assert_allclose(computed_loss, expected_loss, rtol=1e-4, atol=1e-4)
    # Check if metrics are present
    self.assertIn("distill/soft_loss", metrics)
    self.assertIn("distill/hard_loss", metrics)
    self.assertIn("distill/kl_div", metrics)
    self.assertIn("distill/teacher_loss", metrics)

  def test_top_k_distillation_logic(self):
    # [FUNCTION DOC]
    # Verifies distill_top_k=2 yields a different loss than full-vocabulary
    # distillation on 20 classes.
    """Verifies that setting distill_top_k changes the computed loss."""
    s_key, t_key, l_key = jax.random.split(jax.random.key(42), 3)
    batch_size, num_classes = 2, 20
    student_logits = self._get_dummy_logits(s_key, batch_size, num_classes)
    teacher_logits = self._get_dummy_logits(t_key, batch_size, num_classes)
    labels = self._get_dummy_labels(l_key, batch_size, num_classes)
    temp = 2.0
    alpha = 0.5

    # 1. Compute loss with FULL vocab (k=0)
    strategy_full = LogitStrategy(
        student_forward_fn=self.dummy_fn,
        teacher_forward_fn=self.dummy_fn,
        labels_fn=self.dummy_fn,
        temperature=temp,
        alpha=alpha,
        distill_top_k=0,
    )
    loss_full, _ = strategy_full.compute_loss(
        student_output=student_logits,
        teacher_output=teacher_logits,
        labels=labels,
    )

    # 2. Compute loss with Top-K (k=2)
    # Since num_classes=20, filtering to top 2 should definitely change the KL term
    strategy_k = LogitStrategy(
        student_forward_fn=self.dummy_fn,
        teacher_forward_fn=self.dummy_fn,
        labels_fn=self.dummy_fn,
        temperature=temp,
        alpha=alpha,
        distill_top_k=2,
    )
    loss_k, _ = strategy_k.compute_loss(
        student_output=student_logits,
        teacher_output=teacher_logits,
        labels=labels,
    )

    # The losses should be different because the distribution is truncated
    self.assertNotAlmostEqual(loss_full, loss_k, places=4)

  @parameterized.named_parameters(
      ("alpha_one", 1.0),
      ("alpha_half", 0.5),
      ("alpha_zero", 0.0),
  )
  def test_compute_eval_loss(self, alpha):
    # [FUNCTION DOC]
    # Verifies compute_eval_loss returns 2.49732 regardless of alpha and an empty
    # metrics dict.
    s_key, l_key = jax.random.split(jax.random.key(0), 2)
    batch_size, num_classes = 4, 10
    student_logits = self._get_dummy_logits(s_key, batch_size, num_classes)
    labels = self._get_dummy_labels(l_key, batch_size, num_classes)
    temp = 3.0
    strategy = LogitStrategy(
        student_forward_fn=self.dummy_fn,
        teacher_forward_fn=self.dummy_fn,
        labels_fn=self.dummy_fn,
        temperature=temp,
        alpha=alpha,
    )
    expected_loss = 2.49732

    # Now returns tuple (loss, metrics)
    computed_loss, metrics = strategy.compute_eval_loss(
        student_output=student_logits,
        labels=labels,
    )

    npt.assert_allclose(computed_loss, expected_loss, rtol=1e-5, atol=1e-5)
    self.assertEqual(metrics, {})

  def test_get_train_loss(self):
    # [FUNCTION DOC]
    # Verifies get_teacher_outputs and get_train_loss route inputs through the
    # forward/labels functions (pass-through lambdas reading keyword inputs) so
    # the result equals compute_loss on the same arrays with the same metric keys.
    strategy = LogitStrategy(
        student_forward_fn=lambda _, student_output, **kwargs: student_output,
        teacher_forward_fn=lambda _, teacher_output, **kwargs: teacher_output,
        labels_fn=lambda labels, **kwargs: labels,
    )
    inputs = {
        "student_output": jnp.array([1.0, 2.0, 3.0]),
        "teacher_output": jnp.array([4.0, 5.0, 6.0]),
        "labels": jnp.array([7.0, 8.0, 9.0]),
    }
    
    # Manually compute expected tuple
    expected_loss, expected_metrics = strategy.compute_loss(
        student_output=inputs["student_output"],
        teacher_output=inputs["teacher_output"],
        labels=inputs["labels"],
    )

    teacher_output = strategy.get_teacher_outputs(
        teacher_model=DummyModel(), inputs=inputs
    )
    
    # get_train_loss should return tuple now
    computed_loss, computed_metrics = strategy.get_train_loss(
        student_model=DummyModel(), teacher_output=teacher_output, inputs=inputs
    )

    npt.assert_allclose(teacher_output, inputs["teacher_output"], rtol=1e-6)
    npt.assert_allclose(computed_loss, expected_loss, rtol=1e-6)
    # Check keys match
    self.assertEqual(expected_metrics.keys(), computed_metrics.keys())

  def test_get_eval_loss(self):
    # [FUNCTION DOC]
    # Verifies get_eval_loss equals compute_eval_loss on the same inputs with
    # equal metrics.
    strategy = LogitStrategy(
        student_forward_fn=lambda _, student_output, **kwargs: student_output,
        teacher_forward_fn=lambda _, teacher_output, **kwargs: teacher_output,
        labels_fn=lambda labels, **kwargs: labels,
    )
    inputs = {
        "student_output": jnp.array([1.0, 2.0, 3.0]),
        "teacher_output": jnp.array([]),
        "labels": jnp.array([7.0, 8.0, 9.0]),
    }
    # Expected returns tuple
    expected_loss, expected_metrics = strategy.compute_eval_loss(
        student_output=inputs["student_output"],
        labels=inputs["labels"],
    )

    # get_eval_loss returns tuple
    computed_loss, computed_metrics = strategy.get_eval_loss(
        student_model=DummyModel(), inputs=inputs
    )

    npt.assert_allclose(computed_loss, expected_loss, rtol=1e-6)
    self.assertEqual(computed_metrics, expected_metrics)


if __name__ == "__main__":
  absltest.main()
