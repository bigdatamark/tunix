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
# Shared utilities for the agentic RL stack: numpy padding helpers, extraction of
# the latest assistant turn and following environment messages from a chat
# history, per-message tokenization with assistant loss masks using a chat
# template parser, a reader/writer style lock (RolloutSyncLock) that lets many
# rollouts run concurrently but gives exclusive, preferred access to weight
# syncs, and an event-loop getter. Depends on numpy, threading, asyncio, the
# tokenizer adapter and the chat template parser module.
"""Utility functions for agentic models."""

import asyncio
import threading
from typing import Any, Optional

import numpy as np
import tunix.generate.tokenizer_adapter as tok_adapter
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_template_parser


def left_pad(x, length, pad, dtype=np.int32):
  # [FUNCTION DOC]
  # HELPER: Left-pad or left-truncate a 1D sequence to exactly length.
  # Inputs: x (array-like), length (int), pad (fill value), dtype (default int32).
  # Behavior: converts x to a numpy array of dtype; if it has at least length
  # elements returns its last length elements; otherwise prepends pad values.
  # Outputs: numpy array of size length. Assumes 1D input (size is used).
  x = np.asarray(x, dtype=dtype)
  if x.size >= length:
    return x[-length:]
  pad_part = np.full(length - x.size, pad, dtype)
  return np.concatenate([pad_part, x], axis=0)


def right_pad(x, length, pad, dtype=np.int32):
  # [FUNCTION DOC]
  # HELPER: Right-pad or right-truncate a 1D sequence to exactly length: keeps the
  # first length elements if too long, otherwise appends pad values. Same inputs and
  # dtype handling as left_pad.
  x = np.asarray(x, dtype=dtype)
  if x.size >= length:
    return x[:length]
  pad_part = np.full(length - x.size, pad, dtype)
  return np.concatenate([x, pad_part], axis=0)


def pad_prompt_and_completion(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    max_prompt_length: int,
    max_generation_steps: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  # [FUNCTION DOC]
  # Pad a prompt and completion to fixed lengths.
  # Inputs: prompt token list, completion token list, max_prompt_length,
  # max_generation_steps, pad_id.
  # Behavior: left-pads/truncates the prompt to max_prompt_length (keeping its end),
  # right-pads/truncates the completion to max_generation_steps (keeping its start),
  # and concatenates them.
  # Outputs: tuple (padded prompt, padded completion, concatenated), int32 arrays.
  """Pads prompt tokens to the left and completion tokens to the right."""
  left_padded_prompt_tokens = left_pad(prompt_tokens, max_prompt_length, pad_id)
  right_padded_completion_tokens = right_pad(
      completion_tokens, max_generation_steps, pad_id
  )
  tokens = np.concatenate(
      [left_padded_prompt_tokens, right_padded_completion_tokens], axis=0
  )

  return left_padded_prompt_tokens, right_padded_completion_tokens, tokens


def get_recent_assistant_user_messages(
    chat_completions_messages: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
  # [FUNCTION DOC]
  # Find the latest assistant message and the env messages after it.
  # Inputs: list of message dicts.
  # Behavior: scans from the end; collects messages whose role is "user" or "tool"
  # until the first assistant message is met (other roles are skipped); stops at
  # that assistant message.
  # Outputs: tuple (assistant message dict or None, list of the collected user/tool
  # messages in chronological order). If there is no assistant message, all
  # user/tool messages are returned.
  """Extracts the most recent assistant message and environment messages (user/tool) from a chat completions list.

  Args:
      chat_completions_messages (list[dict]): List of message dictionaries from
        chat completions.

  Returns:
      tuple[dict, list[dict]]: A tuple containing:
          - The most recent assistant message (or None if not found)
          - A list of environment messages (user/tool) that occurred after the
          last assistant message, in chronological order.
  """
  # Loop backwards to get the last assistant message and environment messages
  env_messages = []
  assistant_message = None
  for message in reversed(chat_completions_messages):
    role = message.get("role", None)
    if role == "assistant":
      assistant_message = message
      break
    elif role in ["user", "tool"] and assistant_message is None:
      env_messages.append(message)
  # Reverse the env_messages to maintain chronological order
  env_messages = list(reversed(env_messages))

  return assistant_message, env_messages


def convert_messages_to_string(
    message: dict[str, Any],
) -> dict[str, str]:
  # [FUNCTION DOC]
  # Normalize a message dict so all values are strings.
  # Inputs: message dict. Behavior: for each key, numpy array values are replaced by
  # their single scalar (item()); a None "content" becomes ""; other values kept.
  # Each resulting value must be a str.
  # Outputs: new dict with the same keys.
  # Exceptions: ValueError naming the field and type when a value is not a string;
  # ValueError from item() for arrays with more than one element.
  """Converts non-string message values to strings.

  Specifically, numpy array values are converted to their item representation.
  Raises an error if, after processing, any value is not a string.

  Args:
    message: A dictionary representing a message.

  Returns:
    A new dictionary with all values ensured to be strings.

  Raises:
    ValueError: If any message field is not a string after preprocessing.
  """
  # Parse message to text
  processed_messages = {}
  for k, v in message.items():
    if isinstance(v, np.ndarray):
      processed_messages[k] = v.item()
    elif k == "content" and v is None:
      processed_messages[k] = ""
    else:
      processed_messages[k] = v

    if not isinstance(processed_messages[k], str):
      raise ValueError(
          f"Message field {k} must be a string after preprocessing, got"
          f" {type(processed_messages[k])}."
      )
  return processed_messages


def convert_single_message(
    msg: dict[str, str],
    tokenizer: Any,
    parser: chat_template_parser.BaseChatTemplateParser,
    is_first: bool = False,
    is_generation: bool = False,
) -> tuple[list[int], list[int]]:
  # [FUNCTION DOC]
  # Tokenize one message and build its loss mask.
  # Inputs: msg dict, tokenizer (with encode), parser (BaseChatTemplateParser),
  # is_first (message starts the conversation), is_generation (append generation
  # prompt).
  # Behavior: normalizes the message with convert_messages_to_string, renders it
  # alone via parser.parse with add_generation_prompt=is_generation and
  # is_first_msg=is_first. For assistant messages, if the parser has an attribute
  # named assistant_token and the text starts with it, that prefix is removed
  # (since it was emitted as the previous generation prompt). Encodes the text
  # with add_special_tokens=False, falling back to encode(text) when the tokenizer
  # does not accept that keyword (TypeError). Mask is 1 for every token of an
  # assistant message, 0 otherwise.
  # Outputs: (token list, mask list) of equal length.
  # KNOWN ISSUE: parsers store the assistant prefix at parser.tokens.assistant_token,
  # not parser.assistant_token, so the prefix-stripping branch never runs and
  # assistant tokens include the role prefix (masked as 1). Also a TypeError
  # raised inside encode for other reasons triggers the fallback call.
  """Converts a single message to tokens and a mask.

  Args:
    msg: The message to convert.
    tokenizer: The tokenizer to use.
    parser: The chat template parser.
    is_first: Whether this is the first message in a sequence.
    is_generation: Whether this message is for generation.

  Returns:
    A tuple containing (tokens, mask).
  """
  processed_messages = convert_messages_to_string(msg)
  msg_text = parser.parse(
      messages=[processed_messages],
      add_generation_prompt=is_generation,
      is_first_msg=is_first,
  )

  # Remove assistant token if present (it's in the prior generation prompt).
  if msg["role"] == "assistant" and hasattr(parser, "assistant_token"):
    assistant_token = parser.assistant_token
    if msg_text.startswith(assistant_token):
      msg_text = msg_text[len(assistant_token) :]

  # Tokenize
  # Some tokenizers (like raw sentencepiece) have an `encode` method that
  # doesn't accept `add_special_tokens`.
  try:
    tokens = tokenizer.encode(msg_text, add_special_tokens=False)
  except TypeError:
    tokens = tokenizer.encode(msg_text)

  # Create mask (1 for assistant, 0 for others)
  mask_value = 1 if msg["role"] == "assistant" else 0
  masks = [mask_value] * len(tokens)

  return tokens, masks


def tokenize_and_generate_masks(
    messages: list[dict[str, str]],
    tokenizer: tok_adapter.TokenizerAdapter,
    parser: chat_template_parser.BaseChatTemplateParser,
    contains_first_msg: bool = False,
    contains_generation_msg: bool = False,
) -> tuple[list[int], list[int]]:
  # [FUNCTION DOC]
  # Tokenize a list of messages into concatenated tokens and masks.
  # Inputs: messages, tokenizer (TokenizerAdapter, must provide dedup_bos_ids),
  # parser, contains_first_msg (the first message begins the conversation),
  # contains_generation_msg (add the generation prompt after the last message).
  # Behavior: if the parser defines preprocess_messages (Gemma), applies it first.
  # Converts each message with convert_single_message, marking only index 0 as first
  # (when contains_first_msg) and only the last as generation (when
  # contains_generation_msg), and concatenates results. Finally passes the token
  # list through tokenizer.dedup_bos_ids.
  # Outputs: (tokens, masks).
  # KNOWN ISSUE: masks are not adjusted after dedup_bos_ids, so if a duplicate BOS
  # is removed the token and mask lists differ in length / misalign.
  """Converts multiple messages to tokens and masks.

  Args:
    messages: The messages to convert.
    tokenizer: The tokenizer to use.
    parser: The chat template parser.
    contains_first_msg: Whether the first message in `messages` is the absolute
      first message of the conversation. This is used by the parser to add
      special beginning-of-sequence tokens.
    contains_generation_msg: Whether to add a generation prompt after the last
      message in `messages`. This signals the model to start generating a
      response (e.g., by adding an assistant role token).

  Returns:
    A tuple containing (all_tokens, all_masks).
  """
  # For parsers that require preprocessing (e.g., merging system messages),
  # apply it before iterating message by message.
  if hasattr(parser, "preprocess_messages"):
    messages = parser.preprocess_messages(messages)

  all_tokens = []
  all_masks = []

  # Process each message
  for i, msg in enumerate(messages):
    is_first = contains_first_msg and i == 0
    is_generation = contains_generation_msg and i == len(messages) - 1
    tokens, masks = convert_single_message(
        msg, tokenizer, parser, is_first, is_generation
    )
    all_tokens.extend(tokens)
    all_masks.extend(masks)

  all_tokens = tokenizer.dedup_bos_ids(all_tokens)
  return all_tokens, all_masks


class RolloutSyncLock:
  # [CLASS DOC]
  # Thread-based many-readers / one-writer lock with writer preference. Rollouts
  # (readers) may hold it concurrently; a weight sync (writer) needs exclusive
  # access, and once a sync is waiting new rollouts block. State: a mutex with two
  # condition variables (can_rollout, can_weight_sync), counters of active
  # rollouts, active syncs (0 or 1) and waiting syncs.
  """A lock object that allows many simultaneous rollouts or one exclusive weight sync.

  Sync preference: Once a weight sync requests the lock, new rollouts will be
  blocked until the sync has acquired and released the lock.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Create the mutex, the two conditions sharing it, and zero all counters.
    self._mutex = threading.Lock()
    self._can_rollout = threading.Condition(self._mutex)
    self._can_weight_sync = threading.Condition(self._mutex)
    self._rollouts = 0
    # Number of active weight syncs. This should only be 0 or 1.
    self._weight_syncs = 0
    self._weight_syncs_waiting = 0

  def acquire_rollout(self):
    # [FUNCTION DOC]
    # Block while a sync is active or waiting, then increment the active rollout
    # count.
    """Acquire a rollout lock."""
    with self._mutex:
      while self._weight_syncs > 0 or self._weight_syncs_waiting > 0:
        self._can_rollout.wait()
      self._rollouts += 1

  def release_rollout(self):
    # [FUNCTION DOC]
    # Decrement the active rollout count; when it reaches zero and a sync is waiting,
    # notify one waiting sync. No check against releasing more than acquired.
    """Release a rollout lock."""
    with self._mutex:
      self._rollouts -= 1
      if self._rollouts == 0 and self._weight_syncs_waiting > 0:
        self._can_weight_sync.notify()

  def acquire_weight_sync(self):
    # [FUNCTION DOC]
    # Block while any rollout or sync is active (counting itself as waiting while
    # blocked), then mark one sync active.
    """Acquire a weight sync lock."""
    with self._mutex:
      while self._rollouts > 0 or self._weight_syncs > 0:
        self._weight_syncs_waiting += 1
        self._can_weight_sync.wait()
        self._weight_syncs_waiting -= 1
      self._weight_syncs += 1

  def release_weight_sync(self):
    # [FUNCTION DOC]
    # Mark the sync inactive, notify one waiting sync, and wake all waiting rollouts
    # (rollouts re-check and stay blocked if another sync is still waiting).
    """Release a weight sync lock."""
    with self._mutex:
      self._weight_syncs -= 1
      # Notify waiting sync first, then all rollouts
      self._can_weight_sync.notify()
      self._can_rollout.notify_all()


def get_or_create_loop():
  # [FUNCTION DOC]
  # Return asyncio.get_event_loop(); if that raises RuntimeError (no loop in this
  # thread), create a new event loop, set it as current for the thread, and return
  # it. Side effects: may install a new event loop.
  """Returns the current event loop or creates a new one if needed."""
  try:
    loop = asyncio.get_event_loop()
  except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
  return loop
