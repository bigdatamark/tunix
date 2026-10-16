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
# Adapts heterogeneous tokenizers (SentencePiece processors, HuggingFace
# PreTrainedTokenizerBase subclasses, or any duck-typed object with encode,
# decode, bos_id, eos_id, pad_id methods) to one interface used by samplers.
# Contents: TokenizerType enum, TokenizerAdapter wrapper (normalized
# encode/decode/special-id access, chat templating with a Gemma fallback,
# attribute delegation), and Tokenizer, a convenience subclass that loads a
# tokenizer from a path and offers a tokenize function returning numpy ids.
# User-provided text is passed through token_sanitization.sanitize_control_tokens
# before chat templating and in Tokenizer.tokenize to neutralize injected
# control tokens. Imports sentencepiece at import time; transformers is imported
# lazily.
"""Adapt tokenizers to a common interface."""

import enum
import inspect
from typing import Any

from etils import epath
import numpy as np
from tunix.utils import token_sanitization

import sentencepiece as spm


class TokenizerType(enum.Enum):
  # [CLASS DOC]
  # Enum of supported tokenizer kinds: SP (value sp, SentencePiece), HF (value
  # hf, HuggingFace), NONE (value none, a duck-typed custom tokenizer exposing
  # the required methods).
  SP: str = 'sp'  # sentencepiece tokenizer
  HF: str = 'hf'  # huggingface tokenizer
  NONE: str = 'none'  # Represents no tokenizer


class TokenizerAdapter:
  # [CLASS DOC]
  # Wrapper normalizing tokenizer access. Holds the wrapped object in _tokenizer
  # and its detected kind in _tokenizer_type. Unknown attribute lookups are
  # forwarded to the wrapped tokenizer via __getattr__.
  """Wrapper for different tokenizers used in sampler."""

  def __init__(self, tokenizer: Any):
    # [FUNCTION DOC]
    # Stores the tokenizer and detects its type: SP if it is an instance of
    # SentencePieceProcessor; else HF if its class MRO includes
    # transformers.tokenization_utils_base.PreTrainedTokenizerBase (checked by
    # name, so transformers need not be imported); else NONE if it has all of
    # encode, decode, bos_id, eos_id, pad_id attributes.
    # Raises ValueError listing the type and the missing method names otherwise.
    self._tokenizer = tokenizer

    missing_methods = self._missing_methods()

    if isinstance(self._tokenizer, spm.SentencePieceProcessor):
      self._tokenizer_type = TokenizerType.SP
    elif self._is_hf_tokenizer():
      self._tokenizer_type = TokenizerType.HF
    elif not missing_methods:
      self._tokenizer_type = TokenizerType.NONE
    else:
      raise ValueError(
          'Your tokenizer should either be a `spm.SentencePieceProcessor` '
          'tokenizer, a HuggingFace tokenizer, or it should have '
          'the following methods: '
          '`["encode", "decode", "bos_id", "eos_id", "pad_id"]`. Received: '
          f'`type(tokenizer)` = {type(tokenizer)}, with missing methods: '
          f'{missing_methods}.'
      )

  def encode(self, text: str, **kwargs) -> list[int]:
    # [FUNCTION DOC]
    # Encodes text to a list of token ids, forwarding keyword arguments. SP uses
    # EncodeAsIds; HF and NONE use encode.
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.EncodeAsIds(text, **kwargs)
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.encode(text, **kwargs)
    else:
      return self._tokenizer.encode(text, **kwargs)

  def decode(self, ids: list[int], **kwargs) -> str:
    # [FUNCTION DOC]
    # Decodes a list of ids to a string, forwarding keyword arguments. SP uses
    # DecodeIds; HF and NONE use decode.
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.DecodeIds(ids, **kwargs)
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.decode(ids, **kwargs)
    else:
      return self._tokenizer.decode(ids, **kwargs)

  def bos_id(self) -> int:
    # [FUNCTION DOC]
    # Returns the beginning-of-sequence id: SP and NONE call bos_id(); HF reads
    # bos_token_id (may be None).
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.bos_id()
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.bos_token_id
    else:
      return self._tokenizer.bos_id()

  def eos_id(self) -> int:
    # [FUNCTION DOC]
    # Returns the end-of-sequence id: SP and NONE call eos_id(); HF reads
    # eos_token_id (may be None).
    if self._tokenizer_type == TokenizerType.SP:
      return self._tokenizer.eos_id()
    elif self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.eos_token_id
    else:
      return self._tokenizer.eos_id()

  def pad_id(self) -> int:
    # [FUNCTION DOC]
    # Returns the padding id. SP: pad_id(), raising ValueError if it is -1
    # (undefined). HF: if pad_token_id is None, sets the tokenizer's pad_token to
    # its eos_token (side effect: mutates the wrapped tokenizer), then returns
    # pad_token_id. NONE: pad_id().
    """Returns the pad token id."""
    if self._tokenizer_type == TokenizerType.SP:
      ret_id = self._tokenizer.pad_id()
      if ret_id == -1:
        raise ValueError('SentencePiece tokenizer has a undefined pad_id.')
      return ret_id
    elif self._tokenizer_type == TokenizerType.HF:
      # e.g. llama3 HF tokenizers do not have pad_id
      if self._tokenizer.pad_token_id is None:
        self._tokenizer.pad_token = self._tokenizer.eos_token
      return self._tokenizer.pad_token_id
    else:
      return self._tokenizer.pad_id()

  def dedup_bos_ids(self, ids: list[int]) -> list[int]:
    # [FUNCTION DOC]
    # Removes repeated leading BOS tokens so at most one remains. Advances an index
    # while the current and next ids are both equal to bos_id, then returns the
    # slice from that index. Lists without duplicate leading BOS are returned as an
    # equal slice (a copy for lists).
    """Deduplicates the bos_id at the beginning of the list."""
    i = 0
    while i < len(ids) - 1 and ids[i] == ids[i + 1] == self.bos_id():
      i += 1
    return ids[i:]

  def _missing_methods(self) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: returns the names among encode, decode, bos_id, eos_id, pad_id that
    # the wrapped tokenizer lacks (attribute existence only, not callability).
    """Checks if the tokenizer has any missing methods."""
    required_methods = ['encode', 'decode', 'bos_id', 'eos_id', 'pad_id']
    missing_methods = []
    for method in required_methods:
      if not hasattr(self._tokenizer, method):
        missing_methods.append(method)
    return missing_methods

  def _is_hf_tokenizer(self) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns True if any class in the wrapped tokenizer's MRO has the
    # fully qualified name
    # transformers.tokenization_utils_base.PreTrainedTokenizerBase, else False.
    """Checks if the tokenizer is a huggingface tokenizer."""
    baseclasses = inspect.getmro(type(self._tokenizer))
    baseclass_names = [
        baseclass.__module__ + '.' + baseclass.__name__
        for baseclass in baseclasses
    ]
    if (
        'transformers.tokenization_utils_base.PreTrainedTokenizerBase'
        in baseclass_names
    ):
      return True
    return False

  @property
  def tokenizer(self) -> Any:
    # [FUNCTION DOC]
    # Read-only property returning the wrapped tokenizer object.
    return self._tokenizer

  def __getattr__(self, name: str) -> Any:
    # [FUNCTION DOC]
    # Called only when normal attribute lookup fails; returns the same-named
    # attribute of the wrapped tokenizer, raising AttributeError if absent there.
    # KNOWN ISSUE: if accessed before _tokenizer is set (e.g. in a subclass init
    # before super init, or during unpickling/copy), looking up _tokenizer itself
    # re-enters this method and recurses until RecursionError.
    """Delegate unknown attributes to the wrapped tokenizer.

    This keeps the adapter compatible with callers that expect Hugging Face
    tokenizer attributes such as bos_token/eos_token while still using the
    normalized adapter interface for encode/decode/id helpers.
    """
    return getattr(self._tokenizer, name)

  def apply_chat_template(
      self,
      messages: list[dict[str, str]],
      add_generation_prompt: bool = True,
      tokenize: bool = False,
      **kwargs,
  ) -> str | list[int]:
    # [FUNCTION DOC]
    # Formats chat messages (dicts with role and content, other keys preserved)
    # into a prompt string or token ids. First copies each message with its
    # content sanitized via sanitize_control_tokens (raises KeyError if a message
    # lacks content). HF: delegates to the tokenizer's apply_chat_template with
    # add_generation_prompt, tokenize and extra kwargs. SP: uses the Gemma template
    # helper (kwargs ignored). NONE: delegates to the tokenizer's
    # apply_chat_template if it has one, else the Gemma template helper.
    # KNOWN ISSUE: the docstring says NotImplementedError is raised for
    # unsupported tokenizers, but the code never raises it; it falls back to the
    # Gemma template instead.
    """Applies a chat template to format a list of messages.

    Primarily for HuggingFace tokenizers, this formats conversation history
    into a single string or token sequence.

    Args:
      messages: Conversation turns, each with 'role' and 'content'.
      add_generation_prompt: Whether to append a generation prompt.
      tokenize: If True, returns token IDs; otherwise, returns a string.
      **kwargs: Additional args for the underlying `apply_chat_template`.

    Returns:
      The formatted chat as a string or list of token IDs.

    Raises:
      NotImplementedError: If chat templating is not supported by the tokenizer.
    """
    messages = [
        {
            **m,
            'content': token_sanitization.sanitize_control_tokens(m['content']),
        }
        for m in messages
    ]
    if self._tokenizer_type == TokenizerType.HF:
      return self._tokenizer.apply_chat_template(
          messages,
          add_generation_prompt=add_generation_prompt,
          tokenize=tokenize,
          **kwargs,
      )
    elif self._tokenizer_type == TokenizerType.SP:
      # TODO(haoyugao): The tokenizer type is SP but the template is for
      # Gemma. This should be aligned, for example by adding a 'gemma'
      # tokenizer type or a flag. For now, all sentencepiece tokenizers default
      # to using the Gemma chat template.
      return self._apply_gemma_chat_template(
          messages, add_generation_prompt, tokenize
      )
    else:
      if hasattr(self._tokenizer, 'apply_chat_template'):
        return self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )
      # Implements the Gemma chat template format as a fallback.
      return self._apply_gemma_chat_template(
          messages, add_generation_prompt, tokenize
      )

  def _apply_gemma_chat_template(
      self,
      messages: list[dict[str, str]],
      add_generation_prompt: bool,
      tokenize: bool,
  ) -> str | list[int]:
    # [FUNCTION DOC]
    # HELPER: builds a Gemma-format chat string. For each message whose role is
    # user or model, appends start_of_turn marker, role, newline, content,
    # end_of_turn marker and newline; other roles (e.g. system, assistant) are
    # silently dropped. If add_generation_prompt, appends the start_of_turn marker
    # followed by model and a newline. Returns encode(chat_str) if tokenize, else
    # the string.
    # KNOWN ISSUE: messages with role assistant or system are dropped, which can
    # silently lose content for OpenAI-style conversations.
    """Applies the Gemma chat template format."""
    chat_str = ''
    for message in messages:
      role = message.get('role')
      content = message.get('content')
      if role in ('user', 'model'):
        chat_str += f'<start_of_turn>{role}\n{content}<end_of_turn>\n'

    if add_generation_prompt:
      chat_str += '<start_of_turn>model\n'

    if tokenize:
      return self.encode(chat_str)
    return chat_str


class Tokenizer(TokenizerAdapter):
  # [CLASS DOC]
  # TokenizerAdapter subclass that constructs the underlying tokenizer itself
  # from a type name and a path, and adds a tokenize method producing an int32
  # numpy array. Stores the requested type string in the public attribute
  # tokenizer_type (huggingface or sentencepiece), separate from the adapter's
  # detected _tokenizer_type.
  """Tokenizing and encoding/decoding text using TokenizerAdapter."""

  def __init__(
      self,
      tokenizer_type: str = 'sentencepiece',
      tokenizer_path: str = 'gs://gemma-data/tokenizers/tokenizer_gemma2.model',
      add_bos: bool | None = True,
      add_eos: bool | None = True,
      hf_access_token: str | None = None,
  ):
    # [FUNCTION DOC]
    # Loads a tokenizer. Inputs: tokenizer_type (huggingface or sentencepiece,
    # default sentencepiece), tokenizer_path (default a GCS path to the Gemma 2
    # SentencePiece model), add_bos, add_eos, optional hf_access_token.
    # huggingface: lazily imports transformers and calls
    # AutoTokenizer.from_pretrained with the path, add_bos_token, add_eos_token,
    # token and an empty extra_special_tokens dict.
    # sentencepiece: reads the model bytes via epath (local or GCS), loads into a
    # new SentencePieceProcessor, and if add_bos and/or add_eos are set, sets
    # encode extra options to bos, eos, or bos colon eos.
    # Then initializes the adapter with the tokenizer.
    # Raises ValueError for any other type; propagates IO/download errors.
    # Side effects: file or network reads.

    self.tokenizer_type = tokenizer_type
    if tokenizer_type == 'huggingface':
      import transformers  # pylint: disable=g-import-not-at-top

      tokenizer = transformers.AutoTokenizer.from_pretrained(
          pretrained_model_name_or_path=tokenizer_path,
          add_bos_token=add_bos,
          add_eos_token=add_eos,
          token=hf_access_token,
          extra_special_tokens={},
      )
    elif tokenizer_type == 'sentencepiece':
      model_proto = epath.Path(tokenizer_path).read_bytes()
      tokenizer = spm.SentencePieceProcessor()
      tokenizer.LoadFromSerializedProto(model_proto)
      options = []
      if add_bos:
        options.append('bos')
      if add_eos:
        options.append('eos')

      extra_options_str = ':'.join(options)
      if extra_options_str:
        tokenizer.SetEncodeExtraOptions(extra_options_str)
    else:
      raise ValueError(f'Unsupported tokenizer_type: {tokenizer_type}')
    super().__init__(tokenizer)

  def tokenize(
      self,
      example: str,
      prefix: str = '',
      suffix: str = '',
      add_eos: bool = True,
  ) -> np.ndarray:
    # [FUNCTION DOC]
    # Tokenizes prefix + sanitized example + suffix into an int32 numpy array.
    # Sanitizes example (not prefix/suffix) with sanitize_control_tokens. Prepends
    # bos_id if it is truthy. Encodes: huggingface type via encode with
    # add_special_tokens False; otherwise via the SentencePiece EncodeAsIds. If
    # add_eos, appends eos_id.
    # KNOWN ISSUE: for sentencepiece with add_bos/add_eos enabled in the
    # constructor, EncodeAsIds already adds BOS/EOS via extra options, so BOS and
    # EOS end up duplicated. A BOS id of 0 is treated as absent because of the
    # truthiness check.
    """The tokenization function.

    Args:
      example: Input string to tokenize.
      prefix:  Prefix to add to the input string.
      suffix:  Suffix to add to the input string.
      add_eos: If True, add an "end of sentence" token at the end of the output
        sequence.

    Returns:
      Tokens corresponding to the input string.
    """
    example = token_sanitization.sanitize_control_tokens(example)
    int_list = []
    if self.bos_id():
      int_list.append(self.bos_id())
    if self.tokenizer_type == 'huggingface':
      int_list.extend(
          self.encode(prefix + example + suffix, add_special_tokens=False)
      )
    else:
      # sentencepiece
      int_list.extend(self.tokenizer.EncodeAsIds(prefix + example + suffix))
    if add_eos:
      int_list.append(self.eos_id())
    return np.array(int_list, dtype=np.int32)
