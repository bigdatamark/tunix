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
# GSM8K-style reward functions for the CLI GRPO pipeline.
# Module constants: tag strings for the reasoning block ("<reasoning>"/"</reasoning>") and the
# answer block ("<answer>"/"</answer>"); a type alias describing the reward-function signature
# (any arguments, returns a list of floats); and a compiled regex (multiline + dot-matches-newline)
# that matches: reasoning start tag, at least one char (non-greedy), reasoning end tag, any
# chars, answer start tag, then captures (group 1) a non-greedy non-empty answer up to either
# the answer end tag or end of string.
# All reward functions take the batch of prompts, the batch of completion strings and arbitrary
# keyword arguments (some also take the gold "answer" list) and return one score per completion.
# Side effects: none at import time; check_answer may log when an env var is set.

"""Reward functions for the RLHF pipeline."""

import os
import re
from typing import Callable, List
from absl import logging


# Define the expected signature with type hints
ExpectedSignature = Callable[..., List[float]]

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"
match_format = re.compile(
    rf"{re.escape(reasoning_start)}.+?{re.escape(reasoning_end)}.*?"
    rf"{re.escape(solution_start)}(.+?)(?:{re.escape(solution_end)}|$)",
    flags=re.MULTILINE | re.DOTALL,
)


# All reward functions must have this signature.
# range: [0, 3]
def match_format_exactly(prompts, completions, **kwargs):
  # [FUNCTION DOC]
  # Format reward: for each completion returns 3.0 if the module-level format regex finds a match
  # anywhere in the completion, else integer 0. Inputs: prompts (unused), completions (list of
  # strings), extra kwargs ignored. Output: list of scores in range [0, 3]. No exceptions/side effects.
  return [
      0 if match_format.search(response) is None else 3.0
      for response in completions
  ]


# range: [-2, 2]
def match_format_approximately(prompts, completions, **kwargs):
  # [FUNCTION DOC]
  # Approximate-format reward: for each completion starts from 0 and, for each of the four tags
  # (reasoning start, reasoning end, answer start, answer end), adds 0.5 if the tag occurs exactly
  # once in the completion, otherwise subtracts 0.5. Inputs: prompts (unused), completions (list of
  # strings), kwargs ignored. Output: list of scores in [-2, 2]. No exceptions/side effects.
  scores = []

  for completion in completions:
    score = 0
    response = completion
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += 0.5 if response.count(reasoning_start) == 1 else -0.5
    score += 0.5 if response.count(reasoning_end) == 1 else -0.5
    score += 0.5 if response.count(solution_start) == 1 else -0.5
    score += 0.5 if response.count(solution_end) == 1 else -0.5
    scores.append(score)
  return scores


# range: [-1, 3]
def check_answer(prompts, completions, answer, **kwargs):
  # [FUNCTION DOC]
  # Answer-correctness reward. Inputs: prompts, completions (list of strings), answer (list of gold
  # answers, any type), kwargs ignored.
  # Logic: gold answers are stringified, stripped and have commas removed. If the environment
  # variable LOG_GRPO_ROLLOUT equals "1", logs prompts, completions, raw and cleaned gold answers at
  # INFO level. For each completion extracts the format regex's capture group (stripped, commas
  # removed) or None when there is no match. Scoring, pairwise with gold answers (zip truncates to
  # the shorter list): no extraction gives 0; exact string match gives 3.0; otherwise both are
  # parsed as floats: numerically equal gives 3.0; gold equal to zero gives -1.0 if guess is nonzero
  # else 0.0; otherwise the ratio guess/gold in [0.9, 1.1] gives 0.5, in [0.8, 1.2] gives 0.25,
  # anything else -1.0. A float-conversion failure gives -0.5.
  # Output: list of scores in [-1, 3]. Exceptions: none escape (ValueError/ZeroDivisionError caught).
  # Side effects: optional logging.
  """Checks if the extracted response matches the gold answer."""
  # Clean formatting (e.g. stripping spaces and commas).
  clean_answers = [str(a).strip().replace(",", "") for a in answer]

  log_rollout = os.environ.get("LOG_GRPO_ROLLOUT", "0") == "1"
  if log_rollout:
    logging.info("GRPO Rollout Prompts: %r", prompts)
    logging.info("GRPO Rollout Completions: %r", completions)
    logging.info("GRPO Rollout Gold Answers (Raw): %r", answer)
    logging.info("GRPO Rollout Gold Answers (Cleaned): %s", clean_answers)

  extracted_responses = [
      guess.group(1).strip().replace(",", "")
      if (guess := match_format.search(r)) is not None
      else None
      for r in completions
  ]

  scores = []
  for guess, true_answer in zip(extracted_responses, clean_answers):
    if guess is None:
      scores.append(0)
      continue

    # Correct answer gets 3 points!
    if guess == true_answer:
      scores.append(3.0)
      continue

    # If string comparison fails, try numerical equivalence.
    try:
      guess_float = float(guess)
      true_answer_float = float(true_answer)

      if guess_float == true_answer_float:
        scores.append(3.0)
      elif true_answer_float == 0:
        # Cannot compute ratio. Penalize if guess is not also 0.
        scores.append(-1.0 if guess_float != 0 else 0.0)
      else:
        ratio = guess_float / true_answer_float
        if 0.9 <= ratio <= 1.1:
          scores.append(0.5)
        elif 0.8 <= ratio <= 1.2:
          scores.append(0.25)
        else:
          scores.append(-1.0)  # Penalize wrong answers
    except (ValueError, ZeroDivisionError):
      # Penalize if float conversion or division fails.
      scores.append(-0.5)

  return scores


# range: [0, 1.5]
def check_numbers(prompts, completions, answer, **kwargs):
  # [FUNCTION DOC]
  # Numeric-match reward. Inputs: prompts (unused), completions, answer (gold answers), kwargs.
  # Logic: uses a regex (multiline + dotall) finding the answer start tag followed (non-greedy) by
  # the first run of one or more digits/dots, captured. Gold answers are stringified, stripped,
  # commas removed. For each pair (zip truncation): no capture gives 0; otherwise both values are
  # parsed to floats and compared: equal gives 1.5, else 0.0; a parse failure (e.g. a capture of
  # just dots) gives 0. Output: list of scores in [0, 1.5]. No exceptions escape; no side effects.
  # KNOWN ISSUE: the captured number cannot include a leading minus sign, so negative answers can
  # never be matched; also the tag in the regex is not escaped (harmless for these tags).
  match_numbers = re.compile(
      rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
  )
  responses = completions
  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  # Clean formatting (e.g. stripping spaces and commas).
  clean_answers = [str(a).strip().replace(",", "") for a in answer]

  scores = []
  for guess, true_answer in zip(extracted_responses, clean_answers):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer_val = float(true_answer.strip().replace(",", ""))
      guess_val = float(guess.strip().replace(",", ""))
      scores.append(1.5 if guess_val == true_answer_val else 0.0)
    except ValueError:
      scores.append(0)
      continue
  return scores
