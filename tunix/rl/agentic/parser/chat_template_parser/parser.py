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
# Chat-template rendering for agentic RL. Converts a list of chat messages (dicts
# with "role" and "content" string keys) into the exact prompt string a given
# model family expects, so that rendering can be done incrementally per message
# and concatenated with previously tokenized text.
# Contents: TokenConfig (dataclass of the special strings of a template),
# BaseChatTemplateParser (abstract renderer with role dispatch and control-token
# sanitization), and concrete parsers: Default (delegates to the tokenizer's
# apply_chat_template), Qwen (ChatML with optional empty think block), Llama 3
# header-style, and Gemma (turn-style, system merged into first user turn).
# Module aliases dataclass, abstractmethod and ABC are defined for brevity.
# Depends on tunix.utils.token_sanitization for stripping control tokens from
# message content. No import-time side effects.
"""Base classes and factory for parsing chat messages into model-specific formats."""

import abc
import dataclasses
from typing import Dict, List
from tunix.utils import token_sanitization


dataclass = dataclasses.dataclass
abstractmethod = abc.abstractmethod
ABC = abc.ABC


@dataclass
class TokenConfig:
  # [CLASS DOC]
  # Dataclass holding the literal strings used by a chat template. All fields are
  # strings defaulting to the empty string: bos_token, eos_token, eot_token (end of
  # turn, appended after each message), system_token, user_token, assistant_token
  # (role prefixes), tool_start_token/tool_end_token (wrapping tool calls),
  # tool_response_start_token/tool_response_end_token (wrapping tool results), and
  # message_separator (inserted between consecutive rendered messages by the base
  # parse and prefixed to continuation renders).
  """Token configuration for different chat templates."""

  bos_token: str = ""
  eos_token: str = ""
  eot_token: str = ""
  system_token: str = ""
  user_token: str = ""
  assistant_token: str = ""
  tool_start_token: str = ""
  tool_end_token: str = ""
  tool_response_start_token: str = ""
  tool_response_end_token: str = ""
  # Separator inserted between consecutive messages by `parse()`. Match what
  # the model's `apply_chat_template` writes between successive messages so
  # incremental per-message rendering can be concatenated without a fixup.
  message_separator: str = ""


class BaseChatTemplateParser(ABC):
  # [CLASS DOC]
  # Abstract renderer of chat messages to a template string.
  # Subclasses must implement _init_tokens (returning a TokenConfig) and
  # _init_generation_prompt (the text that opens the assistant turn). Rendering of
  # each role is "role prefix token + sanitized content + eot token"; tool messages
  # are rendered as a user turn wrapping content in the tool-response tokens.
  # Message content is sanitized so that it cannot contain any of the template's
  # own non-empty control strings.
  """Abstract base class for chat template parsers."""

  def __init__(self, tokenizer, enable_thinking: bool = True):
    # [FUNCTION DOC]
    # Construct the parser.
    # Inputs: tokenizer (any object; concrete subclasses read bos/eos tokens or call
    # apply_chat_template on it); enable_thinking (bool, default True) used by
    # subclasses such as Qwen.
    # Behavior: stores both, then calls _init_tokens to set self.tokens and
    # _init_generation_prompt to set self.generation_prompt (in that order, so the
    # latter may read self.tokens). Builds self._tokens_to_sanitize: the set of all
    # non-empty string values of the TokenConfig fields except message_separator.
    # Outputs: none. Side effects: sets attributes tokenizer, enable_thinking,
    # tokens, generation_prompt, _tokens_to_sanitize.
    # Exceptions: whatever the subclass init hooks raise.
    self.tokenizer = tokenizer
    self.enable_thinking = enable_thinking
    self.tokens = self._init_tokens()
    self.generation_prompt = self._init_generation_prompt()
    # message_separator is a per-turn formatting hint (e.g. "\n"), not a
    # control token; including it would strip every newline from message
    # content.
    self._tokens_to_sanitize = {
        v
        for k, v in dataclasses.asdict(self.tokens).items()
        if k != "message_separator" and isinstance(v, str) and v
    }

  @abstractmethod
  def _init_tokens(self) -> TokenConfig:
    # [FUNCTION DOC]
    # HELPER: Abstract hook returning the TokenConfig for this template.
    # Base implementation returns None; subclasses must override.
    """Initialize token configuration."""
    pass

  @abstractmethod
  def _init_generation_prompt(self) -> str:
    # [FUNCTION DOC]
    # HELPER: Abstract hook returning the generation prompt string (the text that
    # starts an assistant turn). Called after self.tokens is set. Base returns None.
    """Initialize generation prompt."""
    pass

  def parse(
      self,
      messages: List[Dict[str, str]],
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    # [FUNCTION DOC]
    # Render messages into a template string.
    # Inputs: messages, list of dicts with "role" and "content"; add_generation_prompt
    # (bool, default False); is_first_msg (bool, default False) meaning the render
    # starts a fresh conversation.
    # Behavior: builds a list of parts. If is_first_msg, calls _handle_first_message
    # and appends its result when non-empty. Appends the rendering of each message via
    # _parse_message. If add_generation_prompt, appends self.generation_prompt. Joins
    # parts with tokens.message_separator. If not is_first_msg and the result is
    # non-empty, prefixes the separator once more so the chunk can be appended
    # directly after previous turns.
    # Outputs: the rendered string.
    # Exceptions: NotImplementedError for unknown roles; KeyError if a message lacks
    # "role" or "content".
    """Parse messages into chat template format.

    When `is_first_msg=False` the call renders a continuation of an existing
    conversation, so the result is prefixed with `message_separator` to
    re-introduce the inter-message boundary that the previous turn's eot did
    not emit. This lets per-message incremental rendering be concatenated
    directly to prior tokens without external fixups.
    """
    parts = []

    if is_first_msg:
      first_chunk = self._handle_first_message(messages)
      if first_chunk:
        parts.append(first_chunk)

    for message in messages:
      parts.append(self._parse_message(message))

    if add_generation_prompt:
      parts.append(self.generation_prompt)

    sep = self.tokens.message_separator
    result = sep.join(parts)
    if not is_first_msg and result:
      result = sep + result
    return result

  def _handle_first_message(self, messages: List[Dict[str, str]]) -> str:
    # [FUNCTION DOC]
    # HELPER: Produce the text that precedes the first message of a conversation.
    # Inputs: messages (unused here). Outputs: tokens.bos_token. No side effects.
    """Handle special logic for first message."""
    del messages  # Unused in the base implementation.
    return self.tokens.bos_token

  def _parse_message(self, message: Dict[str, str]) -> str:
    # [FUNCTION DOC]
    # HELPER: Render one message according to its role.
    # Inputs: message dict with "role" and "content".
    # Behavior: sanitizes content with token_sanitization.sanitize_control_tokens,
    # passing the template's control strings as extra tokens and enabling the
    # library's default token list only when the template has no control strings.
    # Dispatches on role: "system", "user", "assistant", "tool" to the matching
    # _parse_* method with the sanitized content.
    # Outputs: rendered string.
    # Exceptions: NotImplementedError("Unsupported message role: ...") for other
    # roles; KeyError if keys are missing.
    """Parse a single message based on its role."""
    role = message["role"]
    content = token_sanitization.sanitize_control_tokens(
        message["content"],
        extra_tokens=self._tokens_to_sanitize,
        include_default=not self._tokens_to_sanitize,
    )

    parser_map = {
        "system": self._parse_system,
        "user": self._parse_user,
        "assistant": self._parse_assistant,
        "tool": self._parse_tool,
    }

    if role not in parser_map:
      raise NotImplementedError(f"Unsupported message role: {role}")

    return parser_map[role](content)

  def _parse_system(self, content: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Return system_token + content + eot_token.
    return self.tokens.system_token + content + self.tokens.eot_token

  def _parse_user(self, content: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Return user_token + content + eot_token.
    return self.tokens.user_token + content + self.tokens.eot_token

  def _parse_assistant(self, content: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Return assistant_token + content + eot_token.
    return self.tokens.assistant_token + content + self.tokens.eot_token

  def _parse_tool(self, content: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Render a tool result as a user turn: user_token +
    # tool_response_start_token + content + tool_response_end_token + eot_token.
    return (
        self.tokens.user_token
        + self.tokens.tool_response_start_token
        + content
        + self.tokens.tool_response_end_token
        + self.tokens.eot_token
    )


class DefaultChatTemplateParser(BaseChatTemplateParser):
  # [CLASS DOC]
  # Parser that delegates rendering entirely to the tokenizer's own chat template
  # (tokenizer.apply_chat_template). Uses an all-empty TokenConfig and an empty
  # generation prompt.
  """Default parser using tokenizer's built-in chat template."""

  def _init_tokens(self) -> TokenConfig:
    # [FUNCTION DOC]
    # HELPER: Return a TokenConfig with all fields empty.
    return TokenConfig()

  def _init_generation_prompt(self) -> str:
    # [FUNCTION DOC]
    # HELPER: Return the empty string (no explicit generation prompt).
    return ""

  def parse(
      self,
      messages: List[Dict[str, str]],
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    # [FUNCTION DOC]
    # Render via tokenizer.apply_chat_template with tokenize disabled and the given
    # add_generation_prompt flag.
    # Inputs: messages, add_generation_prompt, is_first_msg (ignored).
    # Outputs: the string returned by the tokenizer. No sanitization is applied.
    # Exceptions: whatever the tokenizer raises (e.g. if it has no chat template).
    # KNOWN ISSUE: is_first_msg is ignored, so incremental (continuation) renders
    # include the full template preamble (e.g. BOS / default system text) again and
    # no separator prefix; behavior is inconsistent with the base class contract.
    return self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


class QwenChatTemplateParser(BaseChatTemplateParser):
  # [CLASS DOC]
  # Parser for Qwen ChatML templates: turns are "<|im_start|>ROLE\n" + content +
  # "<|im_end|>", separated by a newline. When thinking is disabled the assistant
  # prefix also contains an empty think block. A default Qwen system prompt is
  # inserted if a conversation does not start with a system message.
  """Parser for Qwen models."""

  def _init_tokens(self) -> TokenConfig:
    # [FUNCTION DOC]
    # HELPER: Build the Qwen TokenConfig: bos/eos taken from the tokenizer's bos_token
    # and eos_token attributes; eot "<|im_end|>"; system/user prefixes
    # "<|im_start|>system\n" and "<|im_start|>user\n"; assistant prefix from
    # _get_assistant_token; tool call wrapped by "\n<tool_call>\n" and
    # "\n</tool_call>"; tool response wrapped by "<tool_response>\n" and
    # "\n</tool_response>"; message_separator "\n".
    # Exceptions: AttributeError if the tokenizer lacks bos_token/eos_token.
    return TokenConfig(
        bos_token=self.tokenizer.bos_token,
        eos_token=self.tokenizer.eos_token,
        eot_token="<|im_end|>",
        system_token="<|im_start|>system\n",
        user_token="<|im_start|>user\n",
        assistant_token=self._get_assistant_token(),
        tool_start_token="\n<tool_call>\n",
        tool_end_token="\n</tool_call>",
        tool_response_start_token="<tool_response>\n",
        tool_response_end_token="\n</tool_response>",
        message_separator="\n",
    )

  def _get_assistant_token(self) -> str:
    # [FUNCTION DOC]
    # HELPER: Return "<|im_start|>assistant\n", followed by "<think>\n\n</think>\n\n"
    # when enable_thinking is False.
    token = "<|im_start|>assistant\n"
    if not self.enable_thinking:
      token += "<think>\n\n</think>\n\n"
    return token

  def _init_generation_prompt(self) -> str:
    # [FUNCTION DOC]
    # HELPER: Generation prompt equals the assistant prefix token.
    return self.tokens.assistant_token

  def _handle_first_message(self, messages: List[Dict[str, str]]) -> str:
    # [FUNCTION DOC]
    # HELPER: If the first message's role is not "system", return the rendered default
    # system message "You are Qwen, created by Alibaba Cloud. You are a helpful
    # assistant."; otherwise return the empty string (no BOS is emitted).
    # Exceptions: IndexError if messages is empty.
    """Add default system message if first message is not system."""
    if messages[0]["role"] != "system":
      return self._parse_system(
          "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
      )
    return ""


class LlamaChatTemplateParser(BaseChatTemplateParser):
  # [CLASS DOC]
  # Parser for Llama 3 header-style templates: each turn starts with
  # "<|start_header_id|>ROLE<|end_header_id|>\n\n" and ends with "<|eot_id|>";
  # BOS is "<|begin_of_text|>". No message separator.
  """Parser for Llama models."""

  def _init_tokens(self) -> TokenConfig:
    # [FUNCTION DOC]
    # HELPER: Build the Llama TokenConfig with the header-style role prefixes for
    # system, user, assistant, tool start ("tool" header) and tool response
    # ("tool_response" header); eot and tool_end and tool_response_end all
    # "<|eot_id|>"; eos left empty.
    return TokenConfig(
        bos_token="<|begin_of_text|>",
        eot_token="<|eot_id|>",
        system_token="<|start_header_id|>system<|end_header_id|>\n\n",
        user_token="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant_token="<|start_header_id|>assistant<|end_header_id|>\n\n",
        tool_start_token="<|start_header_id|>tool<|end_header_id|>\n\n",
        tool_end_token="<|eot_id|>",
        tool_response_start_token=(
            "<|start_header_id|>tool_response<|end_header_id|>\n\n"
        ),
        tool_response_end_token="<|eot_id|>",
    )

  def _init_generation_prompt(self) -> str:
    # [FUNCTION DOC]
    # HELPER: Generation prompt equals the assistant header token.
    return self.tokens.assistant_token


class GemmaChatTemplateParser(BaseChatTemplateParser):
  # [CLASS DOC]
  # Parser for Gemma turn templates: "<start_of_turn>user\n" / "<start_of_turn>model\n"
  # prefixes, "<end_of_turn>\n" as eot, "<bos>" as BOS. Gemma has no system role:
  # a leading system message is merged into the first user message.
  """Parser for Gemma models."""

  def _init_tokens(self) -> TokenConfig:
    # [FUNCTION DOC]
    # HELPER: Build the Gemma TokenConfig (bos, eot, user and assistant tokens; all
    # other fields empty).
    return TokenConfig(
        bos_token="<bos>",
        eot_token="<end_of_turn>\n",
        user_token="<start_of_turn>user\n",
        assistant_token="<start_of_turn>model\n",
    )

  def _parse_assistant(self, content: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Render an assistant message as assistant_token + content, without an
    # end-of-turn token (so the text matches what the model generates up to EOS).
    return self.tokens.assistant_token + content

  def _parse_system(self, content: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Always raises ValueError, because Gemma has no system role; system
    # content must be merged via preprocess_messages before rendering.
    # This should not be called if parse() is used, as it handles the system
    # prompt by merging it. Raise error for unexpected system messages.
    raise ValueError(
        "Gemma models do not support system messages directly. The system"
        " prompt should be the first message and is handled by merging with"
        " the first user message."
    )

  def preprocess_messages(
      self, messages: List[Dict[str, str]]
  ) -> List[Dict[str, str]]:
    # [FUNCTION DOC]
    # Merge a leading system message into the first user message.
    # Inputs: messages list. Behavior: makes a shallow copy of the list. If the first
    # message is a system message it is removed; if the next message is a user
    # message, its content becomes system content + "\n" + user content; otherwise a
    # new user message with the system content is inserted at the front.
    # Outputs: the new list.
    # Side effects: KNOWN ISSUE: the list is copied but the message dicts are not, so
    # the caller's first user message dict is mutated in place (its content gets the
    # system prompt prepended).
    """Preprocesses messages, merging system prompt into the first user prompt."""
    processed_messages = list(messages)
    if processed_messages and processed_messages[0]["role"] == "system":
      system_message = processed_messages.pop(0)
      if processed_messages and processed_messages[0]["role"] == "user":
        processed_messages[0]["content"] = (
            system_message["content"] + "\n" + processed_messages[0]["content"]
        )
      else:
        processed_messages.insert(
            0, {"role": "user", "content": system_message["content"]}
        )
    return processed_messages

  def parse(
      self,
      messages: List[Dict[str, str]],
      add_generation_prompt: bool = False,
      is_first_msg: bool = False,
  ) -> str:
    # [FUNCTION DOC]
    # Render messages for Gemma.
    # Inputs: messages, add_generation_prompt, is_first_msg (ignored).
    # Behavior: preprocesses messages (system merge), concatenates _parse_message of
    # each with no separator, then appends the generation prompt if requested.
    # Outputs: rendered string. Exceptions: as _parse_message.
    # KNOWN ISSUE: is_first_msg is ignored, so the BOS token is never emitted by this
    # parser, unlike the base class behavior.
    """Parses messages for Gemma. System prompt is prepended to user prompt."""
    processed_messages = self.preprocess_messages(messages)

    result = ""

    for message in processed_messages:
      result += self._parse_message(message)

    if add_generation_prompt:
      result += self.generation_prompt

    return result

  def _init_generation_prompt(self) -> str:
    # [FUNCTION DOC]
    # HELPER: Generation prompt equals the assistant ("model") turn prefix.
    return self.tokens.assistant_token
