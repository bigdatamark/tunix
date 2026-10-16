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
# HELPER: Small safety utility module that strips chat-template control tokens
# (Gemma, Llama 3, ChatML styles) from untrusted text, to prevent prompt
# injection in RL agent environments/tool outputs.
# Module state: a private list of default control token strings
# ("<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "<|begin_of_text|>",
# "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
# "<|im_start|>", "<|im_end|>") and a precompiled regex alternation of their
# escaped literal forms.
"""Safety utilities for RL agents."""

import re
from typing import Iterable

_CONTROL_TOKENS = [
    '<start_of_turn>',
    '<end_of_turn>',
    '<bos>',
    '<eos>',
    '<|begin_of_text|>',
    '<|end_of_text|>',
    '<|start_header_id|>',
    '<|end_header_id|>',
    '<|eot_id|>',
    '<|im_start|>',
    '<|im_end|>',
]

_CONTROL_TOKENS_RE = re.compile('|'.join(map(re.escape, _CONTROL_TOKENS)))


def sanitize_control_tokens(
    content: str | bytes,
    extra_tokens: Iterable[str] | None = None,
    include_default: bool = True,
) -> str:
  # [FUNCTION DOC]
  # Removes control tokens from content.
  # Inputs: content - str, bytes/bytearray, memoryview, any object with a
  #   decode method, or other objects; extra_tokens - optional iterable of extra
  #   literal token strings (empty strings are ignored); include_default -
  #   whether the default token list is included.
  # Behavior: normalize content to text: bytes/bytearray decoded as UTF-8 with
  #   replacement of invalid bytes; memoryview converted to bytes then decoded;
  #   other non-str objects with a decode method are decoded (falling back to
  #   string conversion if decoding raises); other non-None objects are
  #   string-converted. If the result is falsy (empty or None), return it as is.
  #   Choose the regex: the precompiled default one when defaults are included
  #   and no extra tokens are given; otherwise the union of default (if enabled)
  #   and non-empty extra tokens, sorted, escaped, joined as an alternation;
  #   if that union is empty, return the content unchanged. Then repeatedly
  #   delete all matches until a fixed point, so tokens reassembled by removal
  #   of inner tokens (nesting tricks) are also removed.
  # Output: sanitized string (or None/empty input returned unchanged).
  # Exceptions: none raised deliberately. Side effects: none.
  # KNOWN ISSUE: the return annotation says str but None is returned when
  # content is None.
  """Sanitize control tokens from the content.

  Args:
    content: The content to sanitize.
    extra_tokens: Additional tokens to sanitize.
    include_default: Whether to include the default control tokens.

  Returns:
    The sanitized content.
  """
  if isinstance(content, (bytes, bytearray)):
    content = content.decode('utf-8', errors='replace')
  elif isinstance(content, memoryview):
    content = content.tobytes().decode('utf-8', errors='replace')
  elif not isinstance(content, str):
    if hasattr(content, 'decode'):
      try:
        content = content.decode('utf-8', errors='replace')
      except Exception:  # pylint: disable=broad-except
        content = str(content)
    elif content is not None:
      content = str(content)

  if not content:
    return content  # pytype: disable=bad-return-type

  if include_default and not extra_tokens:
    regex = _CONTROL_TOKENS_RE
  else:
    all_tokens = set()
    if include_default:
      all_tokens.update(_CONTROL_TOKENS)
    if extra_tokens:
      all_tokens.update(t for t in extra_tokens if t)

    if not all_tokens:
      return content

    regex = re.compile('|'.join(map(re.escape, sorted(all_tokens))))

  # Strip known model control tokens to prevent prompt injection.
  # We use a loop to handle cases where tokens are nested to bypass sanitization.
  sanitized = content
  while True:
    new_content = regex.sub('', sanitized)
    if new_content == sanitized:
      break
    sanitized = new_content
  return sanitized
