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
# Unit tests for chat template parsers in
# tunix.rl.agentic.parser.chat_template_parser.parser: the default parser
# (delegates to tokenizer.apply_chat_template), and Qwen, Llama and Gemma
# hand-written templates. Messages are dicts with role and content; parse
# returns the rendered prompt string. Tokenizers are mocks. No side effects.

"""Tests for chat_template_parser."""

from unittest import mock

from absl.testing import absltest

from tunix.rl.agentic.parser.chat_template_parser import parser


class DefaultChatTemplateParserTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests for DefaultChatTemplateParser.

  def test_parse_uses_tokenizer_template(self):
    # [FUNCTION DOC]
    # Verifies parse calls tokenizer.apply_chat_template exactly once with
    # the messages, tokenize False and the given add_generation_prompt.
    mock_tokenizer = mock.Mock()
    messages = [{'role': 'user', 'content': 'Hello'}]

    p = parser.DefaultChatTemplateParser(mock_tokenizer)
    p.parse(messages, add_generation_prompt=True)

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        messages, tokenize=False, add_generation_prompt=True
    )


class QwenChatTemplateParserTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests for QwenChatTemplateParser ChatML-style rendering.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: creates a mock tokenizer with bos "<bos>" and eos "<eos>".
    super().setUp()
    self.mock_tokenizer = mock.Mock()
    self.mock_tokenizer.bos_token = '<bos>'
    self.mock_tokenizer.eos_token = '<eos>'

  def test_parse_with_system_message(self):
    # [FUNCTION DOC]
    # Verifies a continuation chunk (is_first_msg False) starts with a
    # newline separator, renders each message as im_start role newline
    # content im_end, joins messages with newlines, and has no trailing
    # newline; no default system prompt is added.
    # ``is_first_msg=False`` (default) renders a continuation chunk: leading
    # separator, no trailing separator. eot_token is ``<|im_end|>`` (no
    # trailing ``\n``); the inter-message ``\n`` is emitted by joining the
    # separator between parts.
    p = parser.QwenChatTemplateParser(self.mock_tokenizer)
    messages = [
        {'role': 'system', 'content': 'You are Qwen.'},
        {'role': 'user', 'content': 'Hello'},
    ]
    result = p.parse(messages)
    expected = ('\n<|im_start|>system\nYou are Qwen.<|im_end|>\n'
                '<|im_start|>user\nHello<|im_end|>')
    self.assertEqual(result, expected)

  def test_parse_without_system_message_and_is_first_msg(self):
    # [FUNCTION DOC]
    # Verifies the first chunk has no leading separator and prepends the
    # default Qwen system prompt when the first message is not a system one.
    # ``is_first_msg=True`` is the opening of a conversation: no leading
    # separator. Default system prompt is prepended because the first
    # message is not a system message.
    p = parser.QwenChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'user', 'content': 'Hello'}]
    result = p.parse(messages, is_first_msg=True)
    expected = ('<|im_start|>system\n'
                'You are Qwen, created by Alibaba Cloud. You are a helpful'
                ' assistant.<|im_end|>\n'
                '<|im_start|>user\nHello<|im_end|>')
    self.assertEqual(result, expected)

  def test_parse_with_add_generation_prompt(self):
    # [FUNCTION DOC]
    # Verifies add_generation_prompt appends a newline and an open
    # assistant header after the last message.
    p = parser.QwenChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'user', 'content': 'Hello'}]
    result = p.parse(messages, add_generation_prompt=True)
    expected = ('\n<|im_start|>user\nHello<|im_end|>\n'
                '<|im_start|>assistant\n')
    self.assertEqual(result, expected)

  def test_parse_with_tool_message(self):
    # [FUNCTION DOC]
    # Verifies a tool message renders as a user turn wrapping the content in
    # tool_response tags on their own lines.
    p = parser.QwenChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'tool', 'content': 'Tool output'}]
    result = p.parse(messages)
    expected = ('\n<|im_start|>user\n'
                '<tool_response>\nTool output\n</tool_response>'
                '<|im_end|>')
    self.assertEqual(result, expected)

  def test_parse_with_disable_thinking(self):
    # [FUNCTION DOC]
    # Verifies enable_thinking False inserts an empty think block (think
    # tags separated by blank lines) before assistant content and after the
    # generation-prompt assistant header.
    p = parser.QwenChatTemplateParser(
        self.mock_tokenizer, enable_thinking=False
    )
    messages = [{'role': 'assistant', 'content': 'Thinking...'}]
    result = p.parse(messages, add_generation_prompt=True)
    expected = (
        '\n<|im_start|>assistant\n<think>\n\n</think>\n\nThinking...<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    )
    self.assertEqual(result, expected)


class LlamaChatTemplateParserTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests for LlamaChatTemplateParser (Llama 3 header/eot format).

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: creates a plain mock tokenizer.
    super().setUp()
    self.mock_tokenizer = mock.Mock()

  def test_parse_with_system_message_and_is_first_msg(self):
    # [FUNCTION DOC]
    # Verifies the first chunk starts with begin_of_text and each message
    # renders as start_header role end_header, two newlines, content, eot.
    p = parser.LlamaChatTemplateParser(self.mock_tokenizer)
    messages = [
        {'role': 'system', 'content': 'You are Llama.'},
        {'role': 'user', 'content': 'Hello'},
    ]
    result = p.parse(messages, is_first_msg=True)
    expected = (
        '<|begin_of_text|>'
        '<|start_header_id|>system<|end_header_id|>\n\nYou are Llama.<|eot_id|>'
        '<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>'
    )
    self.assertEqual(result, expected)

  def test_parse_without_is_first_msg(self):
    # [FUNCTION DOC]
    # Verifies a non-first chunk omits begin_of_text.
    p = parser.LlamaChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'user', 'content': 'Hello'}]
    result = p.parse(messages, is_first_msg=False)
    expected = '<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>'
    self.assertEqual(result, expected)

  def test_parse_with_add_generation_prompt(self):
    # [FUNCTION DOC]
    # Verifies add_generation_prompt appends an assistant header followed by
    # two newlines.
    p = parser.LlamaChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'user', 'content': 'Hello'}]
    result = p.parse(messages, add_generation_prompt=True)
    expected = ('<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>'
                '<|start_header_id|>assistant<|end_header_id|>\n\n')
    self.assertEqual(result, expected)

  def test_parse_with_tool_message(self):
    # [FUNCTION DOC]
    # Verifies a tool message renders as a user header wrapping a nested
    # tool_response header block, with two eot tokens at the end.
    # KNOWN ISSUE: the expected output pins a nested header with a doubled
    # eot token, which looks like an artifact of the implementation rather
    # than a standard Llama 3 tool-response format.
    p = parser.LlamaChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'tool', 'content': 'Tool output'}]
    result = p.parse(messages)
    expected = ('<|start_header_id|>user<|end_header_id|>\n\n'
                '<|start_header_id|>tool_response<|end_header_id|>\n\n'
                'Tool output'
                '<|eot_id|>'
                '<|eot_id|>')
    self.assertEqual(result, expected)

  def test_parse_with_unsupported_role_raises_error(self):
    # [FUNCTION DOC]
    # Verifies an unknown role raises NotImplementedError.
    p = parser.LlamaChatTemplateParser(self.mock_tokenizer)
    messages = [{'role': 'unsupported', 'content': 'Hello'}]
    with self.assertRaises(NotImplementedError):
      p.parse(messages)


class GemmaChatTemplateParserTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests for GemmaChatTemplateParser, which has no system role and folds
  # system text into the following user message.

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: creates a mock tokenizer and self.parser for Gemma.
    super().setUp()
    self.mock_tokenizer = mock.Mock()
    self.parser = parser.GemmaChatTemplateParser(self.mock_tokenizer)

  def test_preprocess_with_system_and_user(self):
    # [FUNCTION DOC]
    # Verifies a system plus user pair becomes one user message whose
    # content is system text, newline, user text.
    messages = [
        {'role': 'system', 'content': 'System prompt.'},
        {'role': 'user', 'content': 'User prompt.'},
    ]
    processed = self.parser.preprocess_messages(messages)
    self.assertEqual(len(processed), 1)
    self.assertEqual(processed[0]['role'], 'user')
    self.assertEqual(
        processed[0]['content'], 'System prompt.\nUser prompt.'
    )

  def test_preprocess_with_system_only(self):
    # [FUNCTION DOC]
    # Verifies a lone system message is converted to a user message with
    # the same content.
    messages = [{'role': 'system', 'content': 'System prompt.'}]
    processed = self.parser.preprocess_messages(messages)
    self.assertEqual(len(processed), 1)
    self.assertEqual(processed[0]['role'], 'user')
    self.assertEqual(processed[0]['content'], 'System prompt.')

  def test_preprocess_with_no_system_message(self):
    # [FUNCTION DOC]
    # Verifies messages without a system role pass through unchanged.
    messages = [
        {'role': 'user', 'content': 'User prompt.'},
        {'role': 'assistant', 'content': 'Assistant response.'},
    ]
    processed = self.parser.preprocess_messages(messages)
    self.assertEqual(processed, messages)

  def test_parse_raises_error_for_direct_system_message(self):
    # [FUNCTION DOC]
    # Verifies the private per-message renderer raises ValueError for a
    # system message that bypassed preprocessing.
    message = {'role': 'system', 'content': 'System prompt.'}
    with self.assertRaises(ValueError):
      # Directly test _parse_message to ensure it raises for system messages
      # that might bypass the preprocessing in parse().
      self.parser._parse_message(message)


if __name__ == '__main__':
  absltest.main()
