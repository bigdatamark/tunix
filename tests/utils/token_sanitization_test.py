# Copyright 2026 Google LLC
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
# Unit tests for tunix.utils.token_sanitization.sanitize_control_tokens,
# which repeatedly removes known chat-template control tokens (Gemma, Llama,
# ChatML) plus optional extra tokens until no more occur, passing through
# empty strings and None unchanged.
from absl.testing import absltest
from absl.testing import parameterized
from tunix.utils import token_sanitization


class TokenSanitizationTest(parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for sanitize_control_tokens.

  @parameterized.parameters(
      ('', ''),
      (None, None),
      ('hello world', 'hello world'),
      ('<start_of_turn>user<end_of_turn>', 'user'),
      ('<bos>Hello<eos>', 'Hello'),
      ('<|begin_of_text|>System message<|end_of_text|>', 'System message'),
      ('<|start_header_id|>assistant<|end_header_id|><|eot_id|>', 'assistant'),
      ('<|im_start|>user\nHello!<|im_end|>', 'user\nHello!'),
      ('Nested <start_of_turn><bos>tokens<eos><end_of_turn>', 'Nested tokens'),
      ('<start_<start_of_turn>of_turn>', ''),
      ('Recursive <bos<bos>>', 'Recursive '),
  )
  def test_sanitize_control_tokens(self, content, expected):
    # [FUNCTION DOC]
    # Verifies default-token stripping for many inputs, including None/empty
    # passthrough, nested tokens, and tokens that reappear after an inner token
    # is removed (removal is repeated until a fixed point).
    # Inputs: content and expected output.
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(content), expected
    )

  def test_sanitize_control_tokens_with_extra(self):
    # [FUNCTION DOC]
    # Verifies extra_tokens are stripped in addition to defaults.
    content = '[CUSTOM]user\nHello![/CUSTOM]'
    extra_tokens = ['[CUSTOM]', '[/CUSTOM]']
    expected = 'user\nHello!'
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(
            content, extra_tokens=extra_tokens
        ),
        expected,
    )

  def test_sanitize_control_tokens_recursive_with_extra(self):
    # [FUNCTION DOC]
    # Verifies recursive removal also applies to extra tokens.
    content = '[CUSTOM[CUSTOM]]nested[/CUSTOM]'
    extra_tokens = ['[CUSTOM]', '[/CUSTOM]']
    expected = 'nested'
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(
            content, extra_tokens=extra_tokens
        ),
        expected,
    )

  def test_sanitize_control_tokens_with_empty_extra(self):
    # [FUNCTION DOC]
    # Verifies empty-string and None entries in extra_tokens are ignored.
    content = 'hello world'
    extra_tokens = ['', None]
    expected = 'hello world'
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(
            content, extra_tokens=extra_tokens
        ),
        expected,
    )

  def test_sanitize_control_tokens_no_default(self):
    # [FUNCTION DOC]
    # Verifies include_default False strips only extra tokens, keeping <bos>/<eos>.
    content = '<bos>[CUSTOM]hello[/CUSTOM]<eos>'
    extra_tokens = ['[CUSTOM]', '[/CUSTOM]']
    # <bos> and <eos> should remain
    expected = '<bos>hello<eos>'
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(
            content, extra_tokens=extra_tokens, include_default=False
        ),
        expected,
    )

  def test_sanitize_control_tokens_no_default_no_extra(self):
    # [FUNCTION DOC]
    # Verifies include_default False with no extras leaves content unchanged.
    content = '<bos>hello<eos>'
    expected = '<bos>hello<eos>'
    self.assertEqual(
        token_sanitization.sanitize_control_tokens(
            content, include_default=False
        ),
        expected,
    )


if __name__ == '__main__':
  absltest.main()
