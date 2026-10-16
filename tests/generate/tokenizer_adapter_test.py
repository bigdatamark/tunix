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
# Unit tests for tunix.generate.tokenizer_adapter (TokenizerAdapter and Tokenizer).
# Inputs: none beyond the test framework; one test downloads the HuggingFace tokenizer
# "meta-llama/Meta-Llama-3-8B-Instruct" via transformers AutoTokenizer (needs network/credentials).
# Outputs: absltest pass/fail results. Side effects: possible network download and HF cache writes.

#

from absl.testing import absltest
from unittest import mock
import transformers
from tunix.generate import tokenizer_adapter as adapter


AutoTokenizer = transformers.AutoTokenizer


class TokenizerAdapterTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests TokenizerAdapter wrapping either a mock tokenizer or a real HuggingFace tokenizer.

  def test_forwards_wrapped_tokenizer_attributes(self):
    # [FUNCTION DOC]
    # Verifies that TokenizerAdapter exposes the wrapped tokenizer's bos_token and eos_token attributes.
    # Builds a mock tokenizer with encode/decode/bos_id/eos_id/pad_id return values and string
    # bos_token "<bos>"/eos_token "<eos>", wraps it, and asserts the adapter's bos_token/eos_token match.
    # Raises AssertionError on mismatch.
    tokenizer = mock.Mock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = 'decoded'
    tokenizer.bos_id.return_value = 11
    tokenizer.eos_id.return_value = 12
    tokenizer.pad_id.return_value = 0
    tokenizer.bos_token = '<bos>'
    tokenizer.eos_token = '<eos>'

    tokenizer_adapter = adapter.TokenizerAdapter(tokenizer)

    self.assertEqual(tokenizer_adapter.bos_token, '<bos>')
    self.assertEqual(tokenizer_adapter.eos_token, '<eos>')

  def test_hf_tokenizer_adapter(self):
    # [FUNCTION DOC]
    # Verifies that wrapping a HuggingFace tokenizer is detected as TokenizerType.HF and that
    # bos_id, eos_id and pad_id are non-None, and that encoding "test" without special tokens and
    # decoding the result round-trips to exactly "test". The model name defaults to the Llama-3-8B
    # Instruct HF checkpoint (a None placeholder exists for internal override).
    # Side effects: downloads the tokenizer from the HF hub. Raises AssertionError on failure.
    # Additional assignment to handle google internal logics.
    model = None  # pylint: disable=unused-variable
    #
    if model is None:
      model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    hf_tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer_adapter = adapter.TokenizerAdapter(hf_tokenizer)
    self.assertEqual(
        tokenizer_adapter._tokenizer_type, adapter.TokenizerType.HF
    )
    self.assertIsNotNone(tokenizer_adapter.bos_id())
    self.assertIsNotNone(tokenizer_adapter.eos_id())
    self.assertIsNotNone(tokenizer_adapter.pad_id())
    encoded = tokenizer_adapter.encode('test', add_special_tokens=False)
    self.assertIsNotNone(encoded)
    decoded = tokenizer_adapter.decode(encoded)
    self.assertIsNotNone(decoded)
    self.assertEqual(decoded, 'test')


class TokenizerTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests the default construction of the tunix Tokenizer class.

  def test_default_tokenizer(self):
    # [FUNCTION DOC]
    # Verifies that a Tokenizer constructed with no arguments reports tokenizer_type "sentencepiece".
    # KNOWN ISSUE: the second check calls assertIsNotNone with two arguments (the private
    # _tokenizer_type and TokenizerType.SP), so the second value is treated as the failure message
    # and the check never compares equality with SP; an assertEqual was likely intended.
    tokenizer = adapter.Tokenizer()
    self.assertEqual(tokenizer.tokenizer_type, 'sentencepiece')
    self.assertIsNotNone(tokenizer._tokenizer_type, adapter.TokenizerType.SP)


if __name__ == '__main__':
  absltest.main()
