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
# Global, name-keyed registry of reward functions plus built-in examples.
# A reward function takes (task dict, action string) and returns a RewardOutput.
# Provides register (decorator), unregister, get_reward_fn, combine_rewards
# (weighted sum), and built-ins registered at import time: "exact_match",
# "is_two", "dummy", "calculate". Also a private restricted arithmetic evaluator
# based on the ast module and an operator map (add, sub, mul, true division,
# unary minus/plus, power).
# Side effects: importing the module populates the module-level private
# registry dict with the four built-ins; registering the same name twice (e.g. by
# re-registering a built-in) raises ValueError.
"""Registry and utilities for reward functions used in agentic RL experiments.

This module provides a mechanism to register, retrieve, and combine various
reward functions. Reward functions take a task context and an agent's action
as input and return a `reward_types.RewardOutput` containing a scalar reward and
metadata.
"""

import ast
import logging
import operator
from typing import Any, Callable, Dict

from tunix.rl.agentic.rewards import reward_types

_REGISTRY: Dict[
    str, Callable[[Dict[str, Any], str], reward_types.RewardOutput]
] = {}


def register(name: str):
  # [FUNCTION DOC]
  # Decorator factory registering a reward function under name.
  # Inputs: name string. Outputs: a decorator that, when applied to fn, stores fn
  # in the registry under name and returns fn unchanged.
  # Exceptions: the decorator raises ValueError("Reward <name> already registered.")
  # if the name is taken (checked at decoration time, not at factory call time).
  # Side effects: mutates the global registry.
  """Decorator for registering reward functions into the global registry.

  Enables reward functions to be discovered and instantiated by name,
  supporting configuration-driven reward selection in experimental settings.

  Args:
      name (str): Unique identifier for the reward function

  Returns:
      Callable: The decorated function, registered in the system

  Raises:
      ValueError: If a reward function with the given name already exists
  """

  def _wrap(fn):
    # [FUNCTION DOC]
    # HELPER: Inner decorator; see register.
    if name in _REGISTRY:
      raise ValueError(f"Reward {name} already registered.")
    _REGISTRY[name] = fn
    return fn

  return _wrap


def unregister(name: str) -> bool:
  # [FUNCTION DOC]
  # Remove name from the registry. Returns True if it was present and removed,
  # False otherwise. Side effects: mutates the global registry.
  """Remove a reward function from the registry.

  Enables cleanup of registered functions, particularly useful for
  unit testing to prevent state leakage between test cases.

  Args:
      name (str): Name of the reward function to remove

  Returns:
      bool: True if the function was removed, False if it wasn't registered
  """
  if name in _REGISTRY:
    del _REGISTRY[name]
    return True
  return False


def get_reward_fn(name: str):
  # [FUNCTION DOC]
  # Return the reward function registered under name.
  # Exceptions: KeyError if not registered.
  """Retrieve a registered reward function by name.

  Args:
      name (str): The registered name of the reward function

  Returns:
      Callable: The reward function implementation
  """
  return _REGISTRY[name]


@register("exact_match")
def exact_match(task: Dict[str, Any], action: str) -> reward_types.RewardOutput:
  # [FUNCTION DOC]
  # Registered as "exact_match". Converts task["ground_truth"] (default "") to a
  # string and strips whitespace; compares with the stripped action.
  # Outputs: RewardOutput with reward 1.0 on equality else 0.0, and metadata
  # {"exact_match": score}.
  # Exceptions: AttributeError if action is not a string.
  """Binary reward based on exact string matching with ground truth.

  Returns 1.0 for perfect matches after whitespace normalization,
  0.0 for any deviation. Suitable for deterministic answer tasks.

  Args:
      task (Dict[str, Any]): Task context containing 'ground_truth' field
      action (str): Agent's response to evaluate

  Returns:
      RewardOutput: Binary reward (1.0 or 0.0) with match status
  """
  truth = str(task.get("ground_truth", "")).strip()
  score = 1.0 if action.strip() == truth else 0.0
  return reward_types.RewardOutput(score, {"exact_match": score})


def combine_rewards(
    weights: Dict[str, float],
) -> Callable[[Dict[str, Any], str], reward_types.RewardOutput]:
  # [FUNCTION DOC]
  # Build a composite reward function computing a weighted sum.
  # Inputs: weights, mapping of registered reward names to float weights.
  # Outputs: a function of (task, action) that, for each name in weights order,
  # calls the named reward fn (looked up at call time), adds weight times its
  # reward to a running total starting at 0.0, and merges its metadata into one dict
  # (later entries overwrite same keys). Returns RewardOutput(total, merged).
  # Exceptions: KeyError at call time for unregistered names.
  """Create a composite reward function from multiple registered functions.

  Performs weighted linear combination of multiple reward components,
  enabling complex reward engineering through composition.

  Args:
      weights (Dict[str, float]): Mapping from reward function names to weights

  Returns:
      Callable: Composite reward function that computes weighted sum

  Example:
      composite_fn = combine_rewards({"exact_match": 1.0, "zero": 0.0})
  """

  def _fn(task: Dict[str, Any], action: str):
    # [FUNCTION DOC]
    # HELPER: The composite function returned by combine_rewards; see above.
    total, meta = 0.0, {}
    for name, w in weights.items():
      out = get_reward_fn(name)(task, action)
      total += w * out.reward
      meta.update(out.metadata)
    return reward_types.RewardOutput(total, meta)

  return _fn


# -------- Example Reward Function --------
@register("is_two")
def is_two_reward(
    task: Dict[str, Any], action: str
) -> reward_types.RewardOutput:
  # [FUNCTION DOC]
  # Registered as "is_two". Strips the action and parses it as float; score 1.0 if
  # the value equals 2.0, else 0.0; unparseable strings give 0.0.
  # Outputs: RewardOutput(score, {"is_two": score}). task is ignored.
  """Specialized reward function that checks if action represents the number 2.

  Attempts to parse the action as numeric value and returns 1.0 if it equals
  2.0,
  otherwise returns 0.0. Handles both string and numeric representations.

  Args:
      action (str): Agent's response to evaluate

  Returns:
      RewardOutput: Binary reward with parsing status in metadata
  """
  try:
    value = float(action.strip())
    score = 1.0 if value == 2.0 else 0.0
  except ValueError:
    score = 0.0
  return reward_types.RewardOutput(score, {"is_two": score})


@register("dummy")
def dummy_reward(
    task: Dict[str, Any], action: str
) -> reward_types.RewardOutput:
  # [FUNCTION DOC]
  # Registered as "dummy". Always returns RewardOutput(0.0, {}); inputs ignored.
  """A dummy reward function that always returns zero."""
  return reward_types.RewardOutput(0.0, {})


_OP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Pow: operator.pow,
}


def _safe_eval(expr_str: str) -> float:
  # [FUNCTION DOC]
  # HELPER: Safely evaluate an arithmetic expression string.
  # Inputs: expression string. Behavior: parses it in expression mode and walks the
  # tree recursively: constants return their value; binary and unary operations are
  # allowed only if their operator is in the operator map; anything else raises.
  # Outputs: the result converted to float.
  # Exceptions: SyntaxError for invalid syntax; ValueError for unsupported nodes or
  # operators; ZeroDivisionError; OverflowError / TypeError for some inputs (e.g.
  # string constants are accepted as constants and may fail on float conversion or
  # arithmetic).
  # KNOWN ISSUE: power is allowed without bounds, so expressions such as huge
  # exponents can consume excessive CPU/memory.
  def _eval(node):
    # [FUNCTION DOC]
    # HELPER: Recursive node evaluator for _safe_eval (behavior described there).
    if isinstance(node, ast.Constant):
      return node.value
    elif isinstance(node, ast.BinOp):
      op_type = type(node.op)
      if op_type not in _OP_MAP:
        raise ValueError(f"Unsupported operator: {op_type}")
      return _OP_MAP[op_type](_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp):
      op_type = type(node.op)
      if op_type not in _OP_MAP:
        raise ValueError(f"Unsupported operator: {op_type}")
      return _OP_MAP[op_type](_eval(node.operand))
    raise ValueError(f"Unsupported node: {type(node)}")

  tree = ast.parse(expr_str, mode="eval")
  return float(_eval(tree.body))


@register("calculate")
def calculate_reward(
    task: Dict[str, Any], action: str
) -> reward_types.RewardOutput:
  # [FUNCTION DOC]
  # Registered as "calculate". Reward for arithmetic questions.
  # Inputs: task dict with "question" (e.g. "3 + 4 = ?"); action string.
  # Behavior: expression = question with "= ?" removed, then all "=" removed,
  # stripped. Answer = action with "The answer is " removed, stripped, trailing
  # periods removed, parsed as float. Evaluates the expression with _safe_eval and
  # scores 1.0 if the absolute difference is below 1e-6, else 0.0. Any exception
  # is logged via the standard logging module as a warning and yields 0.0.
  # Outputs: RewardOutput(score, {"calculate_correct": score}).
  """Calculates the reward for a math expression based on answer correctness.

  Safely evaluates the expression using AST parsing.

  Args:
    task: The task context containing the 'question' field.
    action: The model's answer as a string.

  Returns:
    RewardOutput: 1.0 if the model's answer matches the evaluated expression
      within a tolerance, 0.0 otherwise.
  """
  question_str = task.get("question", "")
  expression = question_str.replace("= ?", "").replace("=", "").strip()

  try:
    answer_str = action.replace("The answer is ", "").strip().rstrip(".")
    answer = float(answer_str)
    correct_value = _safe_eval(expression)
    tolerance = 1e-6
    if abs(correct_value - answer) < tolerance:
      score = 1.0
    else:
      score = 0.0

  except Exception as e:
    logging.warning(
        "Failed to calculate reward for task %s and action %s: %s",
        task,
        action,
        e,
    )
    score = 0.0
  return reward_types.RewardOutput(score, {"calculate_correct": score})
