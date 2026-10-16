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
# Unit tests for tunix.distillation.distillation_trainer.DistillationTrainer,
# a PeftTrainer-style trainer that trains a student model against a frozen
# teacher using a distillation strategy (LogitStrategy or
# AttentionTransferStrategy): custom loss functions are rejected, student
# params are updated, and sharded training matches unsharded training.
# Module state: _VOCAB_SIZE (256). Sets XLA_FLAGS to force 4 host CPU devices.
# KNOWN ISSUE: XLA_FLAGS is set after importing jax; effective only if the
# backend is not yet initialized. Docstring typo 'Disttillation'.
"""Disttillation trainer unittest."""

import os
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
import optax
from tunix.distillation import distillation_trainer
from tunix.distillation import strategies
from tunix.tests import test_common as tc


os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
_VOCAB_SIZE = 256


def create_sharded_model(model_ctor, rngs, mesh):
  # [FUNCTION DOC]
  # HELPER: builds a model (ToyTransformer config with _VOCAB_SIZE) inside a
  # jitted function with sharding constraints from its partition specs, under
  # the given mesh. Outputs: (model, NamedSharding tree of the state).
  @nnx.jit(static_argnums=(0,))
  def _create_sharded_model(model_ctor, rngs):
    # [FUNCTION DOC]
    # HELPER: jitted inner builder returning (constrained model, state).
    model = model_ctor(config=tc.ModelConfig(vocab_size=_VOCAB_SIZE), rngs=rngs)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model, state

  with mesh:
    model, state = _create_sharded_model(model_ctor, rngs)
  state_sharding = nnx.get_named_sharding(state, mesh)
  return model, state_sharding


def dummy_gen_model_input_fn(x: distillation_trainer.TrainingInput):
  # [FUNCTION DOC]
  # HELPER: maps a TrainingInput to model kwargs: input_tokens, input_mask,
  # positions 0..seq_len-1 and an all-ones attention_mask.
  return {
      'input_tokens': x.input_tokens,
      'input_mask': x.input_mask,
      'positions': jnp.arange(x.input_tokens.shape[1]),
      'attention_mask': jnp.ones_like(x.input_tokens),
  }


def get_labels_fn(
    input_tokens: jax.Array,
    input_mask: jax.Array,
    **kwargs,
):
  # [FUNCTION DOC]
  # HELPER: labels function. Takes tokens shifted left by one as targets,
  # one-hot encodes them over _VOCAB_SIZE and zeroes rows where the shifted
  # input_mask is 0; extra kwargs are ignored. Output: (B, L-1, V) array.
  del kwargs
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]
  # Convert the target labels to one-hot encoded vectors.
  labels = jax.nn.one_hot(target_tokens, _VOCAB_SIZE)
  # Don't update on unwanted tokens.
  return labels * target_mask.astype(labels.dtype)[..., None]


def get_model_outputs_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
):
  # [FUNCTION DOC]
  # HELPER: forward function. Calls model(input_tokens, positions, None,
  # attention_mask), drops the cache output and returns logits without the last
  # position (B, L-1, V); input_mask is ignored.
  del input_mask
  logits, _ = model(
      input_tokens,
      positions,
      None,
      attention_mask,
  )
  # Exclude the last step as it does not appear in the targets.
  return logits[:, :-1, :]


def get_toy_logit_strategy():
  # [FUNCTION DOC]
  # HELPER: returns a LogitStrategy using the toy forward and labels functions
  # with default temperature/alpha.
  return strategies.LogitStrategy(
      student_forward_fn=get_model_outputs_fn,
      teacher_forward_fn=get_model_outputs_fn,
      labels_fn=get_labels_fn,
  )


def get_toy_attention_transfer_strategy():
  # [FUNCTION DOC]
  # HELPER: returns an AttentionTransferStrategy capturing
  # nnx.MultiHeadAttention outputs, using the toy forward/labels functions.
  return strategies.AttentionTransferStrategy(
      student_forward_fn=get_model_outputs_fn,
      teacher_forward_fn=get_model_outputs_fn,
      labels_fn=get_labels_fn,
      attention_layer=nnx.MultiHeadAttention,
  )


def dummy_datasets(batch_size: int, repeat: int = 1):
  # [FUNCTION DOC]
  # HELPER: list of TrainingInput batches from integers 0..127 reshaped to
  # (num_batches, batch_size, 16) with all-ones masks, repeated repeat times.
  # (num_batch, batch_size, seq_len)
  dummy_input = np.arange(128).reshape((-1, batch_size, 16))
  return [
      distillation_trainer.TrainingInput(
          input_tokens=x, input_mask=jnp.ones(x.shape, dtype=jnp.int32)
      )
      for x in dummy_input
  ] * repeat


class DistillationTrainerTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for DistillationTrainer.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: fixture; train and eval datasets are the same 2-batch list.
    super().setUp()
    self.eval_ds = self.train_ds = dummy_datasets(batch_size=4)

  def test_with_loss_fn_raises_exception(self):
    # [FUNCTION DOC]
    # Verifies with_loss_fn raises NotImplementedError (the strategy defines the
    # loss).
    student_rngs = nnx.Rngs(0)
    teacher_rngs = nnx.Rngs(1)
    student_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=_VOCAB_SIZE), rngs=student_rngs
    )
    teacher_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=_VOCAB_SIZE), rngs=teacher_rngs
    )
    strategy = get_toy_logit_strategy()
    config = distillation_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100
    )

    with self.assertRaises(NotImplementedError):
      distillation_trainer.DistillationTrainer(
          student_model, teacher_model, strategy, optax.sgd(1e-3), config
      ).with_loss_fn(lambda a, b: 0.0)

  def test_basic_training(self):
    # [FUNCTION DOC]
    # Verifies logit distillation training updates every student param.
    student_rngs = nnx.Rngs(0)
    teacher_rngs = nnx.Rngs(1)
    student_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=_VOCAB_SIZE), rngs=student_rngs
    )
    teacher_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=_VOCAB_SIZE), rngs=teacher_rngs
    )
    original_variables = jax.tree.map(
        jnp.copy, nnx.state(student_model, nnx.Param)
    )
    strategy = get_toy_logit_strategy()
    config = distillation_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100
    )
    trainer = distillation_trainer.DistillationTrainer(
        student_model, teacher_model, strategy, optax.sgd(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)

    trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(student_model, nnx.Param)

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

  def test_complex_strategy_training(self):
    # [FUNCTION DOC]
    # Verifies attention-transfer distillation with a deeper (6-layer) teacher
    # updates every student param.
    student_rngs = nnx.Rngs(0)
    teacher_rngs = nnx.Rngs(1)
    student_model = tc.ToyTransformer(
        config=(tc.ModelConfig(vocab_size=_VOCAB_SIZE)), rngs=student_rngs
    )
    teacher_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=_VOCAB_SIZE, num_layers=6),
        rngs=teacher_rngs,
    )
    original_variables = jax.tree.map(
        jnp.copy, nnx.state(student_model, nnx.Param)
    )
    strategy = get_toy_attention_transfer_strategy()
    config = distillation_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100
    )
    trainer = distillation_trainer.DistillationTrainer(
        student_model, teacher_model, strategy, optax.sgd(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)

    trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(student_model, nnx.Param)

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

  def test_distributed_training(self):
    # [FUNCTION DOC]
    # Verifies sharded attention-transfer training on a (2, n/2) mesh keeps the
    # ('fsdp','tp') and ('tp','fsdp') specs for layer 0 kernels, updates params,
    # and matches an unsharded run from identical inits (assert_close).
    # KNOWN ISSUE: the same strategy instance (already preprocessed with the
    # sharded models) is reused for the unsharded run.
    total_devices = jax.device_count()
    mesh = shd.Mesh(
        devices=np.array(jax.devices()).reshape(2, total_devices // 2),
        axis_names=('fsdp', 'tp'),
    )
    student_rngs = nnx.Rngs(0)
    teacher_rngs = nnx.Rngs(1)
    student_model, _ = create_sharded_model(
        tc.ToyTransformer, student_rngs, mesh
    )
    teacher_model, _ = create_sharded_model(
        tc.ToyTransformer, teacher_rngs, mesh
    )
    original_variables = jax.tree.map(
        jnp.copy, nnx.state(student_model, nnx.Param)
    )
    strategy = get_toy_attention_transfer_strategy()
    config = distillation_trainer.TrainingConfig(
        eval_every_n_steps=2, max_steps=100
    )
    trainer = distillation_trainer.DistillationTrainer(
        student_model, teacher_model, strategy, optax.sgd(1e-3), config
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)

    with mesh:
      trainer.train(self.train_ds, self.eval_ds)
    variables = nnx.state(student_model, nnx.Param)

    self.assertEqual(
        variables.layers[0].w1.kernel.value.sharding.spec,
        shd.PartitionSpec('fsdp', 'tp'),
    )
    self.assertEqual(
        variables.layers[0].w2.kernel.value.sharding.spec,
        shd.PartitionSpec('tp', 'fsdp'),
    )

    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    # compare with unsharded model
    student_rngs = nnx.Rngs(0)
    teacher_rngs = nnx.Rngs(1)
    unsharded_student_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=_VOCAB_SIZE), rngs=student_rngs
    )
    unsharded_teacher_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=_VOCAB_SIZE), rngs=teacher_rngs
    )
    trainer = distillation_trainer.DistillationTrainer(
        unsharded_student_model,
        unsharded_teacher_model,
        strategy,
        optax.sgd(1e-3),
        config,
    ).with_gen_model_input_fn(dummy_gen_model_input_fn)
    trainer.train(self.train_ds, self.eval_ds)
    unsharded_variables = nnx.state(unsharded_student_model, nnx.Param)

    self.assertIsInstance(
        unsharded_variables.layers[0].w1.kernel.value.sharding,
        jax.sharding.SingleDeviceSharding,
    )
    jax.tree.map_with_path(tc.assert_close, variables, unsharded_variables)


if __name__ == '__main__':
  absltest.main()
