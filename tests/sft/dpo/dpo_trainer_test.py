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
# Unit tests for tunix.sft.dpo.dpo_trainer: DPOTrainer training on
# pre-tokenized TrainingInput and on raw-string DataInput / dict inputs
# (tokenized with a tokenizer), the DPO loss (sigmoid loss with beta and
# label smoothing), and _prepare_inputs (concatenates prompt with chosen and
# rejected completions into a 2B batch, builds attention masks, computes
# reference log-probs, completion mask and logits_to_keep).
# Module setup: disables jax_threefry_partitionable globally so random init
# matches the hard-coded expected values.
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from grain import python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.rl import common
from tunix.sft.dpo import dpo_trainer as dpo_lib
from tunix.tests import test_common as tc

jax.config.update("jax_threefry_partitionable", False)
# jax.config.update("jax_debug_nans", True) # useful for debugging NaN


class MySource(grain.RandomAccessDataSource):
  # [CLASS DOC]
  # HELPER: minimal grain RandomAccessDataSource wrapping an indexable
  # sequence; its length controls how many dataset elements exist.

  def __init__(self, data):
    # [FUNCTION DOC]
    # HELPER: stores the backing sequence.
    self._data = data

  def __getitem__(self, idx):
    # [FUNCTION DOC]
    # HELPER: returns the element at idx.
    return self._data[idx]

  def __len__(self):
    # [FUNCTION DOC]
    # HELPER: returns the number of elements.
    return len(self._data)


def _dummy_dataset(
    source: MySource,
    prompt_ids: np.ndarray,
    prompt_mask: np.ndarray,
    chosen_ids: np.ndarray,
    chosen_mask: np.ndarray,
    rejected_ids: np.ndarray,
    rejected_mask: np.ndarray,
):
  # [FUNCTION DOC]
  # HELPER: builds a grain MapDataset over source where every element (the
  # source value is ignored) is the same TrainingInput batch built from the
  # given prompt/chosen/rejected id and mask arrays.
  return grain.MapDataset.source(source).map(
      lambda x: dpo_lib.TrainingInput(
          prompt_ids=prompt_ids,
          prompt_mask=prompt_mask,
          chosen_ids=chosen_ids,
          chosen_mask=chosen_mask,
          rejected_ids=rejected_ids,
          rejected_mask=rejected_mask,
      )
  )


def _dummy_string_dataset(
    source: MySource,
    prompts: list[str],
    chosen_responses: list[str],
    rejected_responses: list[str],
    return_dict=False,
):
  # [FUNCTION DOC]
  # HELPER: builds a grain MapDataset over source where each element is the
  # same batch of strings, either as a DataInput or (return_dict True) as a
  # dict with keys prompts, chosen_responses and rejected_responses.
  ds = grain.MapDataset.source(source)
  if return_dict:
    return ds.map(
        lambda x: {
            "prompts": prompts,
            "chosen_responses": chosen_responses,
            "rejected_responses": rejected_responses,
        }
    )
  else:
    return ds.map(
        lambda x: dpo_lib.DataInput(
            prompts=prompts,
            chosen_responses=chosen_responses,
            rejected_responses=rejected_responses,
        )
    )


class DPOTrainerTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for DPOTrainer and its loss/input preparation.

  @parameterized.named_parameters(
      dict(
          testcase_name="with_ref_model",
          prompt_ids=np.arange(0, 10).reshape(2, 5),
          prompt_mask=np.ones((2, 5)),
          chosen_ids=np.arange(10, 20).reshape(2, 5),
          chosen_mask=np.ones((2, 5)),
          rejected_ids=np.arange(20, 30).reshape(2, 5),
          rejected_mask=np.ones((2, 5)),
          use_ref_model=True,
      ),
      dict(
          testcase_name="without_ref_model",
          prompt_ids=np.arange(0, 10).reshape(2, 5),
          prompt_mask=np.ones((2, 5)),
          chosen_ids=np.arange(10, 20).reshape(2, 5),
          chosen_mask=np.ones((2, 5)),
          rejected_ids=np.arange(20, 30).reshape(2, 5),
          rejected_mask=np.ones((2, 5)),
          use_ref_model=False,
      ),
  )
  def test_dpo_trainer(
      self,
      prompt_ids,
      prompt_mask,
      chosen_ids,
      chosen_mask,
      rejected_ids,
      rejected_mask,
      use_ref_model,
  ):
    # [FUNCTION DOC]
    # Verifies DPO training (10 steps, eval every 5 on a 2-element eval set)
    # updates params, with and without a reference model, and logs
    # rewards/chosen, rewards/rejected, rewards/margin, rewards/accuracy,
    # log_probs/chosen and log_probs/rejected once per train step and 3 times
    # for eval.
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = None
    if use_ref_model:
      ref_model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    dpo_config = dpo_lib.DPOTrainingConfig(
        eval_every_n_steps=5,
        max_steps=10,
    )
    dpo_trainer = dpo_lib.DPOTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_config,
    )
    train_ds = _dummy_dataset(
        MySource(np.arange(10)),
        prompt_ids,
        prompt_mask,
        chosen_ids,
        chosen_mask,
        rejected_ids,
        rejected_mask,
    )
    eval_ds = _dummy_dataset(
        MySource(np.arange(2)),
        prompt_ids,
        prompt_mask,
        chosen_ids,
        chosen_mask,
        rejected_ids,
        rejected_mask,
    )
    dpo_trainer.train(train_ds, eval_ds=eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

    for metric_name in [
        "rewards/chosen",
        "rewards/rejected",
        "rewards/margin",
        "rewards/accuracy",
        "log_probs/chosen",
        "log_probs/rejected",
    ]:
      self.assertLen(
          dpo_trainer.metrics_logger.get_metric_history(
              "", metric_name, "train"
          ),
          dpo_trainer._train_steps,
      )
      self.assertLen(
          dpo_trainer.metrics_logger.get_metric_history(
              "", metric_name, "eval"
          ),
          3,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="dataclass_inputs",
          train_ds=_dummy_string_dataset(
              MySource(np.arange(10)),
              prompts=["Tunix", "Parallax"],
              chosen_responses=["PT", "distributed training"],
              rejected_responses=["optimizer library", "quantization"],
          ),
      ),
      dict(
          testcase_name="dict_inputs",
          train_ds=_dummy_string_dataset(
              MySource(np.arange(10)),
              prompts=["Tunix", "Parallax"],
              chosen_responses=["PT", "distributed training"],
              rejected_responses=["optimizer library", "quantization"],
              return_dict=True,
          ),
      ),
  )
  def test_dpo_trainer_with_string_inputs(self, train_ds):
    # [FUNCTION DOC]
    # Verifies training from string inputs (dataclass or dict) with MockVocab
    # tokenizer and max prompt/response length 3 updates the policy but leaves
    # the reference model unchanged, and logs reward metrics once per step.
    # KNOWN ISSUE: the 'ref_model is not None' guard is always true here.
    tokenizer = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tokenizer.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
    ref_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tokenizer.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    original_ref_variables = jax.tree.map(
        jnp.copy, nnx.state(ref_model, nnx.Param)
    )
    dpo_config = dpo_lib.DPOTrainingConfig(
        eval_every_n_steps=10,
        max_steps=10,
        max_prompt_length=3,
        max_response_length=3,
    )
    dpo_trainer = dpo_lib.DPOTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_config,
        tokenizer=tokenizer,
    )
    dpo_trainer.train(train_ds, None)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)
    if ref_model is not None:
      ref_variables = nnx.state(ref_model, nnx.Param)
      jax.tree.map_with_path(
          tc.assert_equal, original_ref_variables, ref_variables
      )

    for metric_name in [
        "rewards/chosen",
        "rewards/rejected",
        "rewards/margin",
        "rewards/accuracy",
    ]:
      self.assertLen(
          dpo_trainer.metrics_logger.get_metric_history(
              "", metric_name, "train"
          ),
          dpo_trainer._train_steps,
      )

  def test_dpo_loss_fn(self):
    # [FUNCTION DOC]
    # Verifies dpo_loss_fn against reference values: with get_per_token_logps
    # mocked to fixed seeded normal values for 8 sequences (first 4 chosen, last
    # 4 rejected) and fixed reference log-probs, beta 0.1 gives loss 0.753059
    # and label_smoothing 0.3 gives 0.925447.
    np.random.seed(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    per_token_logps = np.random.normal(0, 5, size=(8, 4))
    ref_per_token_logps = np.random.normal(0, 5, size=(8, 4)).sum(axis=-1)
    train_example = dpo_lib.TrainExample(
        input_ids=jnp.arange(0, 32).reshape(8, 4),
        positions=jnp.ones((8, 4)),
        attention_mask=jnp.ones((8, 4, 4)),
        ref_chosen_logps=ref_per_token_logps[:4],
        ref_rejected_logps=ref_per_token_logps[4:],
        logits_to_keep=4,
        completion_mask=jnp.ones((8, 4)),
    )

    with mock.patch.object(
        common, "get_per_token_logps", return_value=jnp.array(per_token_logps)
    ):
      loss, _ = dpo_lib.dpo_loss_fn(
          model, train_example, beta=0.1, label_smoothing=0
      )
      np.testing.assert_allclose(loss, 0.753059, atol=1e-5)

      loss, _ = dpo_lib.dpo_loss_fn(
          model, train_example, beta=0.1, label_smoothing=0.3
      )
      np.testing.assert_allclose(loss, 0.925447, atol=1e-5)

  def test_dpo_prepare_inputs_for_strings(self):
    # [FUNCTION DOC]
    # Verifies _prepare_inputs on strings: prompts are left-padded and
    # responses right-padded to length 3 using MockVocab ids, producing input_ids
    # rows (chosen rows then rejected rows), attention-mask sums, approximate
    # reference log-probs, completion mask and logits_to_keep 3.
    tokenizer = tc.MockVocab()

    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tokenizer.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    ref_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=tokenizer.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    dpo_trainer = dpo_lib.DPOTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_lib.DPOTrainingConfig(
            eval_every_n_steps=10,
            max_steps=10,
            max_prompt_length=3,
            max_response_length=3,
        ),
        tokenizer=tokenizer,
    )

    # These are random strings, they hold no meaning.
    training_input = dpo_lib.DataInput(
        prompts=["Tunix", "Parallax"],
        chosen_responses=["PT", "distributed training"],
        rejected_responses=["optimizer library", "quantization"],
    )
    out = dpo_trainer._prepare_inputs(training_input)

    expected_input_ids = np.array([
        [0, 1, 14, 1, 16, 0],
        [0, 1, 15, 1, 18, 19],
        [0, 1, 14, 1, 20, 17],
        [0, 1, 15, 1, 21, 0],
    ])
    np.testing.assert_array_equal(out.input_ids, expected_input_ids)
    self.assertEqual(np.sum(out.attention_mask[0]), 14)
    self.assertEqual(np.sum(out.attention_mask[1]), 15)
    self.assertEqual(np.sum(out.attention_mask[2]), 15)
    self.assertEqual(np.sum(out.attention_mask[3]), 14)
    np.testing.assert_allclose(
        out.ref_chosen_logps,
        np.array([-11.21106, -5.985622]),
        atol=1e-1,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        out.ref_rejected_logps,
        np.array([-13.020714, -5.95595]),
        atol=1e-1,
        rtol=1e-2,
    )
    expected_completion_mask = np.array(
        [[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0]]
    )
    np.testing.assert_array_equal(out.completion_mask, expected_completion_mask)
    self.assertEqual(out.logits_to_keep, 3)

  def test_dpo_prepare_inputs(self):
    # [FUNCTION DOC]
    # Verifies _prepare_inputs on token inputs: input_ids are prompt ids followed
    # by chosen (first B rows) then rejected (last B rows) completions; checks
    # attention-mask sums (causal mask over valid tokens), reference log-probs
    # (rtol 1e-2), completion mask equal to the completion masks, and
    # logits_to_keep equal to completion length 4.
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    ref_model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    dpo_trainer = dpo_lib.DPOTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optax.sgd(1e-3),
        training_config=dpo_lib.DPOTrainingConfig(
            eval_every_n_steps=10,
            max_steps=10,
        ),
    )

    training_input = dpo_lib.TrainingInput(
        prompt_ids=np.array([[1, 2, 3, 4, 5], [0, 0, 1, 2, 3]]),
        prompt_mask=np.array([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
        chosen_ids=np.array([[10, 11, 12, 0], [13, 14, 15, 16]]),
        chosen_mask=np.array([[1, 1, 1, 0], [1, 1, 1, 1]]),
        rejected_ids=np.array([[20, 21, 22, 0], [23, 0, 0, 0]]),
        rejected_mask=np.array([[1, 1, 1, 0], [1, 0, 0, 0]]),
    )
    out = dpo_trainer._prepare_inputs(training_input)
    expected_input_ids = np.array([
        [1, 2, 3, 4, 5, 10, 11, 12, 0],
        [0, 0, 1, 2, 3, 13, 14, 15, 16],
        [1, 2, 3, 4, 5, 20, 21, 22, 0],
        [0, 0, 1, 2, 3, 23, 0, 0, 0],
    ])
    np.testing.assert_array_equal(out.input_ids, expected_input_ids)
    self.assertEqual(np.sum(out.attention_mask[0]), 44)
    self.assertEqual(np.sum(out.attention_mask[1]), 28)
    self.assertEqual(np.sum(out.attention_mask[2]), 44)
    self.assertEqual(np.sum(out.attention_mask[3]), 22)
    np.testing.assert_allclose(
        out.ref_chosen_logps, np.array([-20.536058, -20.905323]), rtol=1e-2
    )
    np.testing.assert_allclose(
        out.ref_rejected_logps, np.array([-18.149311, -8.219014]), rtol=1e-2
    )
    expected_completion_mask = np.array(
        [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]]
    )
    np.testing.assert_array_equal(out.completion_mask, expected_completion_mask)
    self.assertEqual(out.logits_to_keep, 4)


if __name__ == "__main__":
  absltest.main()
