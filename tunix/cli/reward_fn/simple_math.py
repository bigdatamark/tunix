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
# Simple binary reward functions (format and answer correctness) for math-style GRPO training.
# Module constants: reasoning and answer tag strings, a reward-function signature type alias,
# and a compiled regex (multiline + dotall) requiring, between optional leading whitespace at a
# line start and optional trailing whitespace at a line end: reasoning start tag, non-empty
# reasoning, reasoning end tag, any text, answer start tag, captured non-empty answer, answer end
# tag. Each reward function returns one score per completion. No side effects.

"""simple format and answer binary reward functions."""

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
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

# binary reward function on format
def check_format(prompts, completions, r=0.1, **kwargs):
  # [FUNCTION DOC]
  # Binary format reward. Inputs: prompts (unused), completions (list of strings), r - reward value
  # (default 0.1), kwargs ignored. Output: list with r for each completion where the module format
  # regex finds a match, else integer 0. No exceptions or side effects.
  return [
      0 if match_format.search(response) is None else r
      for response in completions
  ]


# binary reward function on answer correctness
# NB does not use sympy, only checks float equality 
def check_answer(prompts, completions, answer, r=1, **kwargs):
  # [FUNCTION DOC]
  # Binary answer-correctness reward. Inputs: prompts (unused), completions, answer (list of gold
  # answer strings), r - reward for a correct answer (default 1), kwargs ignored.
  # Logic: finds the answer start tag followed (non-greedy) by the first run of digits/dots, captured
  # (or None). For each pair (zip truncation): None gives 0; else gold and guess are stripped and
  # parsed as floats; equal gives r, unequal gives 0.0; a ValueError on parsing gives 0.
  # Output: list of scores. No side effects.
  # KNOWN ISSUE: the comprehension variable shadows the parameter r inside the comprehension only
  # (harmless), negative numbers are never captured, and gold answers must be strings (a non-string
  # gold answer raises AttributeError on strip, which is not caught).
  match_numbers = re.compile(
      rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
  )
  responses = completions
  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      # TODO (atwigg) add sympy for more complex answers
      scores.append(r if guess == true_answer else 0.0)
    except ValueError:
      scores.append(0)
      continue
  return scores
