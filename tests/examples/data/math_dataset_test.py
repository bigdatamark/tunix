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
# Unit tests for tunix.examples.data.math_dataset: apply_template (renders prompts from question
# records with a tokenizer chat template or a built-in Gemma-style fallback template, strips the
# "#### " prefix from answers yielding None when absent, decodes bytes to str, and keeps only
# prompts/question/answer fields), _parse_huggingface_dataset_name (splits "name:config", defaulting
# config to "default"), and create_dataset with a HuggingFace source. Uses in-file fakes for the
# tokenizer and dataset. No side effects beyond mocks.

"""Tests for tunix.examples.data.math_dataset.apply_template."""

from __future__ import annotations

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.examples.data import math_dataset


class _FakeTokenizer:
  # [CLASS DOC]
  # HELPER: fake tokenizer recording apply_chat_template calls and returning a recognizable string.

  def __init__(self, template_suffix=""):  # suffix helps distinguish calls
    # [FUNCTION DOC]
    # HELPER: input template_suffix (string appended to rendered output, default empty). Initialises an
    # empty calls list.
    self.calls = []
    self.template_suffix = template_suffix

  def apply_chat_template(
      self, msgs, tokenize=False, add_generation_prompt=True
  ):
    # [FUNCTION DOC]
    # HELPER: records a dict of msgs, tokenize and add_generation_prompt in calls, then returns
    # "templated:" followed by the first user-role message content and the template suffix. Raises
    # StopIteration if no message has role "user".
    # Capture call arguments for assertion
    self.calls.append({
        "msgs": msgs,
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    })
    # Return a simple rendered string to check wiring
    user_msg = next(m["content"] for m in msgs if m["role"] == "user")
    return f"templated:{user_msg}{self.template_suffix}"


class _BaseDataset:
  # [CLASS DOC]
  # HELPER: list-backed dataset fake supporting map, iteration and indexing.

  def __init__(self, records):
    # [FUNCTION DOC]
    # HELPER: stores the records as a list.
    self._records = list(records)

  def map(self, fn):
    # [FUNCTION DOC]
    # HELPER: returns a new fake dataset whose records are fn applied to each record (eager).
    return _BaseDataset([fn(x) for x in self._records])

  def __iter__(self):
    # [FUNCTION DOC]
    # HELPER: iterates over the stored records.
    return iter(self._records)

  def __getitem__(self, idx):
    # [FUNCTION DOC]
    # HELPER: returns the record at the given index.
    return self._records[idx]


class ApplyTemplateTest(parameterized.TestCase):
  # [CLASS DOC]
  # Tests math_dataset.apply_template behavior.

  def test_with_tokenizer_uses_apply_chat_template_and_preserves_fields(self):
    # [FUNCTION DOC]
    # Verifies that with a tokenizer, prompts come from apply_chat_template (called once with tokenize and
    # add_generation_prompt passed through, two messages: a system message containing SYSTEM_PROMPT and the
    # user question), question is kept, answer has "#### " stripped, and extra fields are dropped.
    # KNOWN ISSUE: an inline comment says original fields are preserved, while the test asserts the extra
    # field is dropped (only the named fields are kept).
    ds = _BaseDataset([
        {"question": "Q1", "answer": "#### A1", "extra": 7},
    ])
    tok = _FakeTokenizer(template_suffix="!call")

    out = math_dataset.apply_template(
        ds, tokenizer=tok, tokenize=True, add_generation_prompt=False
    )
    result = out[0]

    # Prompts rendered via tokenizer
    self.assertEqual(result["prompts"], "templated:Q1!call")
    # Original fields preserved
    self.assertEqual(result["question"], "Q1")
    self.assertEqual(result["answer"], "A1")
    # Extra fields are dropped by apply_template
    self.assertNotIn("extra", result)

    # Tokenizer called once with expected messages and flags
    self.assertLen(tok.calls, 1)
    call = tok.calls[0]
    self.assertEqual(call["tokenize"], True)
    self.assertEqual(call["add_generation_prompt"], False)
    self.assertEqual(len(call["msgs"]), 2)
    self.assertEqual(call["msgs"][1]["content"], "Q1")
    self.assertIn(math_dataset.SYSTEM_PROMPT, call["msgs"][0]["content"])

  def test_without_tokenizer_falls_back_to_plain_template(self):
    # [FUNCTION DOC]
    # Verifies that without a tokenizer the fallback template is used: prompts contain
    # "<start_of_turn>user", the question and SYSTEM_PROMPT, and the answer is extracted.
    ds = _BaseDataset([
        {"question": "Q2", "answer": "#### A2"},
    ])

    out = math_dataset.apply_template(
        ds, tokenizer=None, tokenize=False, add_generation_prompt=True
    )
    result = out[0]

    # Prompts rendered via fallback TEMPLATE
    self.assertIn("<start_of_turn>user", result["prompts"])
    self.assertIn("Q2", result["prompts"])
    self.assertIn(math_dataset.SYSTEM_PROMPT, result["prompts"])
    self.assertEqual(result["answer"], "A2")

  def test_no_valid_answer(self):
    # [FUNCTION DOC]
    # Verifies that an answer lacking the "####" marker yields answer None while prompts still render.
    ds = _BaseDataset([
        {"question": "Q2", "answer": "A2"},
    ])

    out = math_dataset.apply_template(
        ds, tokenizer=None, tokenize=False, add_generation_prompt=True
    )
    result = out[0]

    # Prompts rendered via fallback TEMPLATE
    self.assertIn("<start_of_turn>user", result["prompts"])
    self.assertIn("Q2", result["prompts"])
    self.assertIn(math_dataset.SYSTEM_PROMPT, result["prompts"])
    self.assertIsNone(result["answer"])

  @parameterized.parameters(True, False)
  def test_bytes_are_decoded_to_str(self, use_tokenizer):
    # [FUNCTION DOC]
    # Verifies byte-string question and answer are decoded to str (answer also stripped of "#### "), both
    # with and without a tokenizer.
    ds = _BaseDataset([
        {"question": b"byte-q", "answer": b"#### byte-a"},
    ])
    tok = _FakeTokenizer() if use_tokenizer else None

    out = math_dataset.apply_template(
        ds, tokenizer=tok, tokenize=False, add_generation_prompt=True
    )
    result = out[0]

    self.assertIsInstance(result["question"], str)
    self.assertIsInstance(result["answer"], str)
    self.assertEqual(result["question"], "byte-q")
    self.assertEqual(result["answer"], "byte-a")


class HuggingFaceDatasetTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests HuggingFace dataset name parsing and loading in math_dataset.

  def test_parse_huggingface_dataset_name_supports_gsm8k_alias(self):
    # [FUNCTION DOC]
    # Verifies "openai/gsm8k" parses to dataset "openai/gsm8k" with config "default".
    dataset_name, config_name = math_dataset._parse_huggingface_dataset_name(
        "openai/gsm8k"
    )

    self.assertEqual(dataset_name, "openai/gsm8k")
    self.assertEqual(config_name, "default")

  def test_parse_huggingface_dataset_name_supports_explicit_config(self):
    # [FUNCTION DOC]
    # Verifies "openai/gsm8k:main" parses to dataset "openai/gsm8k" with config "main".
    dataset_name, config_name = math_dataset._parse_huggingface_dataset_name(
        "openai/gsm8k:main"
    )

    self.assertEqual(dataset_name, "openai/gsm8k")
    self.assertEqual(config_name, "main")

  def test_create_dataset_uses_huggingface_loader(self):
    # [FUNCTION DOC]
    # Verifies create_dataset with data_source "huggingface" calls get_huggingface_dataset once with
    # dataset_name and split keywords, and applies the template to the result (question kept, answer "42",
    # prompt containing the question).
    raw_dataset = _BaseDataset([
        {"question": "Q3", "answer": "#### 42"},
    ])

    with mock.patch.object(
        math_dataset,
        "get_huggingface_dataset",
        return_value=raw_dataset,
    ) as mock_loader:
      dataset = math_dataset.create_dataset(
          data_source="huggingface",
          dataset="openai/gsm8k",
          split="test",
          apply_chat_template_to_dataset=True,
      )

    mock_loader.assert_called_once_with(
        dataset_name="openai/gsm8k",
        split="test",
    )
    self.assertEqual(dataset[0]["question"], "Q3")
    self.assertEqual(dataset[0]["answer"], "42")
    self.assertIn("Q3", dataset[0]["prompts"])


if __name__ == "__main__":
  absltest.main()
