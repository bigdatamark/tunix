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
# limitations under the License

# [MODULE DOC]
# GSM8K reward scoring compatible with the verl framework's GSM8K scorer.
# Provides answer extraction from a model output (strict mode looks for the "#### <number>"
# convention; flexible mode takes the last number) and a scoring function comparing the
# extracted answer string to the ground truth. Module constant: the number of trailing
# characters (300) that extraction considers, to keep regex matching fast. No side effects.

import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
  # [FUNCTION DOC]
  # Extracts the final numeric answer string from a solution text.
  # Inputs: solution_str - model output; method - "strict" or "flexible" (default "strict").
  # Logic: asserts method is one of the two values (AssertionError otherwise). Truncates the text
  # to its last 300 characters if longer. Strict: finds all occurrences of four hashes, a space,
  # then an optional minus followed by digits/dots/commas; if none returns None, else takes the
  # last occurrence and removes commas and dollar signs. Flexible: finds all runs of optional minus
  # followed by digits/dots/commas; if none returns None; otherwise iterates from the last match
  # backwards and returns the first one that is not empty and not a lone dot.
  # Output: extracted answer string or None. No side effects.
  # KNOWN ISSUE: in flexible mode, if every match is invalid (e.g. only "."), the loop finishes
  # and the last-iterated (invalid) value is returned instead of None; flexible mode also does not
  # strip commas, unlike strict mode.
  assert method in ["strict", "flexible"]

  # Optimization: Regular expression matching on very long strings can be slow.
  # For math problems, the final answer is usually at the end.
  # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
  if len(solution_str) > _SOLUTION_CLIP_CHARS:
    solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

  if method == "strict":
    # this also tests the formatting of the model
    solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if len(solutions) == 0:
      final_answer = None
    else:
      # take the last solution
      final_answer = solutions[-1].replace(",", "").replace("$", "")
  elif method == "flexible":
    answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
    final_answer = None
    if len(answer) == 0:
      # no reward is there is no answer
      pass
    else:
      invalid_str = ["", "."]
      # find the last number that is not '.'
      for final_answer in reversed(answer):
        if final_answer not in invalid_str:
          break
  return final_answer


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.0, score=1.0
):
  # [FUNCTION DOC]
  # Scores a solution against the ground truth for GSM8K.
  # Inputs: solution_str - model output; ground_truth - expected answer string; method - extraction
  # mode passed to the extractor; format_score - reward when an answer is extracted but wrong
  # (default 0.0); score - reward for a correct answer (default 1.0).
  # Logic: extracts the answer; returns integer 0 if nothing was extracted, score if the extracted
  # string equals ground_truth exactly (string equality, no numeric normalization), else format_score.
  # Output: a number. Exceptions: AssertionError for an invalid method. No side effects.
  """The scoring function for GSM8k.

  Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning."
  Proceedings of the 62nd Annual
  Meeting of the Association for Computational Linguistics (Volume 1: Long
  Papers). 2024.

  Args:
      solution_str: the solution text
      ground_truth: the ground truth
      method: the method to extract the solution, choices are 'strict' and
        'flexible'
      format_score: the score for the format
      score: the score for the correct answer
  """
  answer = extract_solution(solution_str=solution_str, method=method)
  if answer is None:
    return 0
  else:
    if answer == ground_truth:
      return score
    else:
      return format_score
