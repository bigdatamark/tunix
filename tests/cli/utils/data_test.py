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
# Tests for tunix.cli.utils.data: post_init_dataset (prompt key normalization to "prompts" with byte and
# memoryview decoding and special-token sanitization, max prompt length filtering by tokenizer.encode
# length, train/second split by fraction, num_epochs repetition, batching and num_batches limiting, and
# ValueErrors when filtering or splitting leaves the dataset empty) and get_dataset_from_module (loads a
# Python file defining create_dataset, forwards keyword arguments, optionally applies the chat template to
# "prompt" message lists). Uses in-file fake grain-like datasets and tokenizers. Side effects: writes
# temporary Python module files that are deleted on cleanup.

"""Tests for tunix.cli.utils.data.post_init_dataset."""

from __future__ import annotations

import os
import tempfile

from absl.testing import absltest
from tunix.cli.utils import data as data_lib


class _FakeTokenizer:
  # [CLASS DOC]
  # HELPER: fake tokenizer whose encode splits on whitespace.

  def encode(self, text: str):
    # [FUNCTION DOC]
    # HELPER: returns the list of whitespace-separated chunks of text (one token per word).
    # Simple tokenization: one token per whitespace-separated chunk
    return text.split()


class _FakeChatTemplateTokenizer(_FakeTokenizer):
  # [CLASS DOC]
  # HELPER: fake tokenizer adding a chat template that joins message contents.

  def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
    # [FUNCTION DOC]
    # HELPER: ignores tokenize and add_generation_prompt and returns the message contents joined by " | ".
    del tokenize
    del add_generation_prompt
    return " | ".join(message["content"] for message in messages)


class _BaseDataset:
  # [CLASS DOC]
  # HELPER: list-backed fake of a grain map dataset supporting len, indexing/slicing, filter, repeat,
  # to_iter_dataset and map.
  """Minimal dataset to mimic grain interfaces used in post_init_dataset."""

  def __init__(self, records):
    # [FUNCTION DOC]
    # HELPER: stores records as a list.
    self._records = list(records)

  def __len__(self):
    # [FUNCTION DOC]
    # HELPER: returns the number of records.
    return len(self._records)

  def __getitem__(self, idx):
    # [FUNCTION DOC]
    # HELPER: a slice returns a new fake dataset of the sliced records; an int returns that record.
    if isinstance(idx, slice):
      return _BaseDataset(self._records[idx])
    return self._records[idx]

  def filter(self, fn):
    # [FUNCTION DOC]
    # HELPER: returns a new fake dataset with records for which fn is truthy.
    return _BaseDataset([x for x in self._records if fn(x)])

  def repeat(self, n):
    # [FUNCTION DOC]
    # HELPER: returns a repeat wrapper over this dataset for n epochs.
    return _RepeatDataset(self, n)

  def to_iter_dataset(self):
    # [FUNCTION DOC]
    # HELPER: returns an iterable dataset over the records.
    return _IterDataset(self._records)

  def map(self, fn):  # Not used in tests, but kept for fidelity.
    # [FUNCTION DOC]
    # HELPER: returns a new fake dataset with fn applied to each record.
    return _BaseDataset([fn(x) for x in self._records])


class _RepeatDataset:
  # [CLASS DOC]
  # HELPER: fake repeated dataset that concatenates the base records n times.

  def __init__(self, base: _BaseDataset, n: int):
    # [FUNCTION DOC]
    # HELPER: stores the base dataset and repeat count.
    self._base = base
    self._n = n

  def __len__(self):
    # [FUNCTION DOC]
    # HELPER: returns base length times n.
    return len(self._base) * self._n

  def to_iter_dataset(self):
    # [FUNCTION DOC]
    # HELPER: returns an iterable dataset over the base records repeated n times.
    return _IterDataset(self._base._records * self._n)


class _IterDataset:
  # [CLASS DOC]
  # HELPER: fake iterable dataset supporting batch.

  def __init__(self, records):
    # [FUNCTION DOC]
    # HELPER: stores records as a list.
    self._records = list(records)

  def batch(self, batch_size: int, *, batch_fn=None):
    # [FUNCTION DOC]
    # HELPER: returns a batched dataset of the given size; a provided batch_fn is accepted but ignored.
    if batch_fn:
      # In this mock, we don't fully implement custom batch_fn,
      # but we allow it to be passed.
      pass
    return _BatchedDataset(self._records, batch_size)


class _BatchedDataset:
  # [CLASS DOC]
  # HELPER: fake batched dataset yielding consecutive list slices.

  def __init__(self, records, batch_size: int):
    # [FUNCTION DOC]
    # HELPER: stores records and batch size.
    self._records = records
    self._batch_size = batch_size

  def __iter__(self):
    # [FUNCTION DOC]
    # HELPER: yields successive slices of batch_size records (last batch may be shorter).
    for i in range(0, len(self._records), self._batch_size):
      yield self._records[i : i + self._batch_size]


class PostInitDatasetTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests for post_init_dataset and get_dataset_from_module.

  def test_get_dataset_from_module_passes_kwargs_and_templates_prompt(self):
    # [FUNCTION DOC]
    # Verifies get_dataset_from_module loads a temp module whose create_dataset takes train/eval paths,
    # forwards those kwargs, applies the chat template to the "prompt" messages producing a "prompts" string
    # "train.json | eval.parquet", drops the "prompt" key and keeps other fields.
    module_source = """
class FakeDataset:
  def __init__(self, records):
    self._records = list(records)

  def __len__(self):
    return len(self._records)

  def __getitem__(self, idx):
    return self._records[idx]

  def map(self, fn):
    return FakeDataset([fn(record) for record in self._records])


def create_dataset(train_data_path, eval_data_path):
  return FakeDataset([
      {
          "prompt": [
              {"role": "user", "content": train_data_path},
              {"role": "assistant", "content": eval_data_path},
          ],
          "meta": "kept",
      }
  ])
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write(module_source)
      module_path = f.name

    self.addCleanup(lambda: os.unlink(module_path))

    dataset = data_lib.get_dataset_from_module(
        module_path,
        tokenizer=_FakeChatTemplateTokenizer(),
        apply_chat_template_to_dataset=True,
        train_data_path="train.json",
        eval_data_path="eval.parquet",
    )

    self.assertEqual(
        dataset[0],
        {"prompts": "train.json | eval.parquet", "meta": "kept"},
    )

  def test_get_dataset_from_module_keeps_existing_prompts(self):
    # [FUNCTION DOC]
    # Verifies that with apply_chat_template_to_dataset False and a create_dataset taking no arguments,
    # records already holding "prompts" are returned unchanged.
    module_source = """
class FakeDataset:
  def __init__(self, records):
    self._records = list(records)

  def __len__(self):
    return len(self._records)

  def __getitem__(self, idx):
    return self._records[idx]

  def map(self, fn):
    return FakeDataset([fn(record) for record in self._records])


def create_dataset():
  return FakeDataset([
      {"prompts": "already formatted", "value": 1}
  ])
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
      f.write(module_source)
      module_path = f.name

    self.addCleanup(lambda: os.unlink(module_path))

    dataset = data_lib.get_dataset_from_module(
        module_path,
        tokenizer=_FakeChatTemplateTokenizer(),
        apply_chat_template_to_dataset=False,
    )

    self.assertEqual(dataset[0], {"prompts": "already formatted", "value": 1})

  def test_filters_by_prompt_length(self):
    # [FUNCTION DOC]
    # Verifies records whose prompt token count exceeds max_prompt_length are filtered out, and without a
    # fraction the second dataset is None.
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset([
        {"prompts": "short", "answer": 1},
        {"prompts": "this is too long", "answer": 2},
    ])

    first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
        batch_size=2,
        num_batches=None,
        max_prompt_length=2,  # only the first record should remain
    )

    batches = list(first)
    self.assertIsNone(second)
    self.assertLen(batches, 1)
    self.assertEqual(batches[0], [{"prompts": "short", "answer": 1}])

  def test_raises_when_prompt_length_filter_removes_all_examples(self):
    # [FUNCTION DOC]
    # Verifies a ValueError mentioning "empty after post_init_dataset filtering" when no record passes the
    # length filter.
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset([
        {"prompts": "this is too long", "answer": 1},
        {"prompts": "also too long", "answer": 2},
    ])

    with self.assertRaisesRegex(
        ValueError, "empty after post_init_dataset filtering"
    ):
      data_lib.post_init_dataset(
          dataset,
          tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
          batch_size=2,
          num_batches=None,
          max_prompt_length=2,
      )

  def test_raises_when_fraction_makes_training_split_empty(self):
    # [FUNCTION DOC]
    # Verifies a ValueError mentioning "empty after post_init_dataset split" when fraction 0.5 of a single
    # record leaves the first split empty.
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset([
        {"prompts": "short", "answer": 1},
    ])

    with self.assertRaisesRegex(ValueError, "empty after post_init_dataset split"):
      data_lib.post_init_dataset(
          dataset,
          tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
          batch_size=1,
          num_batches=None,
          max_prompt_length=None,
          fraction=0.5,
      )

  def test_limits_num_batches(self):
    # [FUNCTION DOC]
    # Verifies num_batches 2 with batch_size 3 yields exactly two full batches covering p0..p5.
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset(
        [{"prompts": f"p{i}", "answer": i} for i in range(10)]
    )

    first, _ = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
        batch_size=3,
        num_batches=2,  # keep at most 2 batches * 3 = 6 examples
        max_prompt_length=None,
    )

    batches = list(first)
    self.assertLen(batches, 2)
    self.assertEqual([len(b) for b in batches], [3, 3])
    self.assertEqual(batches[0][0]["prompts"], "p0")
    self.assertEqual(batches[-1][-1]["prompts"], "p5")

  def test_fraction_split_and_repeat(self):
    # [FUNCTION DOC]
    # Verifies fraction 0.5 splits 8 records into the first 4 and last 4, each yielding two batches of 2.
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset(
        [{"prompts": f"p{i}", "answer": i} for i in range(8)]
    )

    first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
        batch_size=2,
        num_batches=None,
        max_prompt_length=None,
        fraction=0.5,
        num_epochs=1,
    )

    first_batches = list(first)
    second_batches = list(second)

    self.assertLen(first_batches, 2)  # 4 items / batch_size 2
    self.assertLen(second_batches, 2)  # remaining 4 items / batch_size 2
    self.assertEqual(first_batches[0][0]["prompts"], "p0")
    self.assertEqual(second_batches[-1][-1]["prompts"], "p7")

  def test_normalizes_prompt_key_to_prompts(self):
    # [FUNCTION DOC]
    # Verifies prompt_key "question" adds a "prompts" copy of the question while keeping original fields.
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset([
        {"question": "short prompt", "answer": 1},
        {"question": "another prompt", "answer": 2},
    ])

    first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
        batch_size=2,
        num_batches=None,
        max_prompt_length=None,
        prompt_key="question",
    )

    self.assertIsNone(second)
    batches = list(first)
    self.assertEqual(
        batches[0],
        [
            {"question": "short prompt", "answer": 1, "prompts": "short prompt"},
            {"question": "another prompt", "answer": 2, "prompts": "another prompt"},
        ],
    )

  def test_normalizes_bytes_prompt_key_to_sanitized_prompts(self):
      # [FUNCTION DOC]
      # Verifies bytes and memoryview prompts are decoded to str and sanitized (the "<start_of_turn>" special
      # token removed) into "prompts", while original values are kept.
      tokenizer = _FakeTokenizer()
      dataset = _BaseDataset([
        {"question": b"<start_of_turn>short prompt", "answer": 1},
        {"question": memoryview(b"another prompt"), "answer": 2},
      ])

      first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
        batch_size=2,
        num_batches=None,
        max_prompt_length=None,
        prompt_key="question",
      )

      self.assertIsNone(second)
      batches = list(first)
      self.assertEqual(
        batches[0],
        [
          {
            "question": b"<start_of_turn>short prompt",
            "answer": 1,
            "prompts": "short prompt",
          },
          {
            "question": memoryview(b"another prompt"),
            "answer": 2,
            "prompts": "another prompt",
          },
        ],
      )

  def test_num_epochs_repeats_dataset(self):
    # [FUNCTION DOC]
    # Verifies num_epochs 3 repeats a 2-record dataset to six batches of size one in order p0, p1, p0, ...
    tokenizer = _FakeTokenizer()
    dataset = _BaseDataset(
        [{"prompts": "p0", "answer": 0}, {"prompts": "p1", "answer": 1}]
    )

    first, second = data_lib.post_init_dataset(
        dataset,
        tokenizer=tokenizer,  # pytype: disable=wrong-arg-types
        batch_size=1,
        num_batches=None,
        max_prompt_length=None,
        num_epochs=3,
    )

    self.assertIsNone(second)
    batches = list(first)
    # 2 items * 3 epochs = 6 batches of size 1
    self.assertLen(batches, 6)
    self.assertEqual(
        [b[0]["prompts"] for b in batches], ["p0", "p1", "p0", "p1", "p0", "p1"]
    )


if __name__ == "__main__":
  absltest.main()
