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
# Math reward function for RL training (DeepScaleR-style), registered in the
# agentic reward registry under the name "deepscaler_math".
# Module constants: THOUGHT_DELIMITER_END is the closing reasoning tag
# "</think>"; RewardOutput aliases the agentic reward output type (imported but
# currently unused by the active code path).
# Depends on math_utils for answer extraction and grading.
from typing import Any, Callable, Dict, List

from tunix.utils import math_utils
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.rewards import reward_types


THOUGHT_DELIMITER_END = "</think>"
RewardOutput = reward_types.RewardOutput


@reward.register("deepscaler_math")
def math_reward(prompts: List[str], completions: List[str], answer: List[str], **kwargs):
  # [FUNCTION DOC]
  # Computes a binary correctness reward per completion.
  # Inputs: prompts (unused), completions - list of model response strings,
  #   answer - list (parallel to completions) of ground truths; each entry may
  #   be None, a scalar (str/float/int) or a list of acceptable answers;
  #   extra keyword arguments are ignored.
  # Behavior per completion i:
  #   - None or empty response -> 0.0.
  #   - If the response contains "</think>", take the text after the FIRST
  #     occurrence (the segment between first and second delimiter if several);
  #     else the whole response.
  #   - Extract the final answer (last boxed expression) via
  #     math_utils.extract_answer; None -> 0.0.
  #   - Ground truth None -> 0.0; scalar wrapped into a one-element list.
  #   - Each truth is string-converted; if it contains a boxed marker, its boxed
  #     content is extracted (dropped if extraction fails), otherwise kept as is.
  #     No usable truths -> 0.0.
  #   - Correct if any truth satisfies grade_answer_mathd OR grade_answer_sympy
  #     OR grade_answer_special_handling (evaluated in that order, short-circuit);
  #     reward 1.0 on the first match, else 0.0.
  # Output: list of floats, one per completion.
  # Exceptions: IndexError if answer is shorter than completions. Side effects:
  #   none (grading may run sympy with timeouts inside math_utils).
  # KNOWN ISSUE: the docstring describes task/action arguments and a single
  #   float return, but the function takes prompts/completions/answer lists and
  #   returns a list of floats.
  """
  A reward function for math tasks that implements the RewardFunction protocol.
  Args:
    task: The task dictionary containing data_source, ground_truth and other metadata
    action: The agent's response/solution

  Returns:
    float: The calculated reward value based on math evaluation
  """
  rewards = []
  # Extract information from task_info
  for i, completion in enumerate(completions):
    model_response = completion

    # Handle None or empty response
    if model_response is None or model_response == "":
      rewards.append(0.0)
      continue

    # Extract solution.
    if THOUGHT_DELIMITER_END in model_response:
      model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
    else:
      model_solution = model_response

    model_answer = math_utils.extract_answer(model_solution)
    if model_answer is None:
      # return RewardOutput(0.0, {"is_correct": False})
      rewards.append(0.0)
      continue

    # Process the ground truth(s)
    ground_truths = answer[i]
    if ground_truths is None:
      # return RewardOutput(0.0, {"is_correct": False})
      rewards.append(0.0)
      continue

    # Convert single answer to list for uniform processing
    if isinstance(ground_truths, str | float | int):
      ground_truths = [ground_truths]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
      truth = str(truth)
      if "\\boxed" in truth:
        processed_truth = math_utils.extract_answer(truth)
        if processed_truth is not None:
          processed_ground_truths.append(processed_truth)
      else:
        processed_ground_truths.append(truth)

    if not processed_ground_truths:
      rewards.append(0.0)
      continue

    # Check against all possible correct answers
    found_correct_answer = False
    for ground_truth in processed_ground_truths:
      if found_correct_answer:
        break
      is_correct = (
          math_utils.grade_answer_mathd(model_answer, ground_truth)
          or math_utils.grade_answer_sympy(model_answer, ground_truth)
          or math_utils.grade_answer_special_handling(
              model_answer, ground_truth
          )
      )
      if is_correct:
        found_correct_answer = True
        reward_value: float = 1.0  # Base reward for a correct answer.
        # Apply tool call bonus if applicable and answer is correct
        # if task_info.get("has_toolcall", False):
        #   reward_value += 0.5
        rewards.append(reward_value)
    if not found_correct_answer:
      rewards.append(0.0)
  return rewards
