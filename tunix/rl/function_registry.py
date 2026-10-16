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
# Thread-safe named function registry used to plug RL algorithm components in
# by string name. Four categories exist: "policy_loss_fn", "value_loss_fn",
# "advantage_estimator", "reward_manager". A module-level default_registry
# instance is created at import time, and convenience get_* / register_*
# functions wrap it per category (register_* return decorators). Functions
# are registered as a side effect of importing modules such as algo_core and
# reward_manager.

import threading
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional
from absl import logging

_POLICY_LOSS_FN_CATEGORY = "policy_loss_fn"
_VALUE_LOSS_FN_CATEGORY = "value_loss_fn"
_ADVANTAGE_ESTIMATOR_CATEGORY = "advantage_estimator"
_REWARD_MANAGER_CATEGORY = "reward_manager"


class FunctionRegistry:
  # [CLASS DOC]
  # Registry mapping category name to a dict of function name to callable,
  # guarded by a threading.Lock. Class constant DEFAULT_ALLOWED_CATEGORIES is
  # the frozenset of the four category names. Only allowed categories may be
  # registered into.
  """A thread-safe registry for functions, organized by category."""

  DEFAULT_ALLOWED_CATEGORIES: FrozenSet[str] = frozenset({
      _POLICY_LOSS_FN_CATEGORY,
      _VALUE_LOSS_FN_CATEGORY,
      _ADVANTAGE_ESTIMATOR_CATEGORY,
      _REWARD_MANAGER_CATEGORY,
  })

  def __init__(self, allowed_categories: Optional[Iterable[str]] = None):
    # [FUNCTION DOC]
    # Input: optional iterable of allowed category names. If None or empty, uses
    # DEFAULT_ALLOWED_CATEGORIES, else a frozenset of the given names. Raises
    # ValueError if the resulting set is empty (only reachable with an iterable
    # that is truthy but yields nothing, e.g. a generator). Creates an empty
    # function dict per allowed category and a lock.
    """Initializes the registry.

    Args:
        allowed_categories: An iterable of strings representing the only
          category names permitted for registration. If None, defaults to
          DEFAULT_ALLOWED_CATEGORIES.
    """
    if not allowed_categories:
      self._allowed_categories: FrozenSet[str] = self.DEFAULT_ALLOWED_CATEGORIES

    else:
      self._allowed_categories: FrozenSet[str] = frozenset(allowed_categories)

    if not self._allowed_categories:
      raise ValueError(
          "FunctionRegistry initialized with no allowed categories."
      )

    self._registry: Dict[str, Dict[str, Callable[..., Any]]] = {
        cat: {} for cat in self._allowed_categories
    }
    self._lock = threading.Lock()

  def _validate_category(self, category: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Raises ValueError listing the sorted allowed categories if category
    # is not allowed; otherwise returns None.
    """Raises ValueError if the category is not allowed."""
    if category not in self._allowed_categories:
      raise ValueError(
          f"Invalid category: '{category}'. "
          f"Allowed categories are: {sorted(list(self._allowed_categories))}"
      )

  def register(
      self, category: str, name: str
  ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    # [FUNCTION DOC]
    # Validates the category immediately (ValueError if not allowed) and returns a
    # decorator. The decorator, under the lock, creates the category dict if
    # missing, logs a warning if name already exists in the category (then
    # overwrites), stores the function, and returns the function unchanged.
    """Returns a decorator to register a function under a category and name."""
    self._validate_category(category)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
      # [FUNCTION DOC]
      # HELPER: Decorator closure described in register; stores func under the
      # captured category and name and returns func.
      with self._lock:
        if category not in self._registry:
          self._registry[category] = {}

        if name in self._registry[category]:
          logging.warning(
              "Function '%s' is already registered in category '%s'. "
              "Overwriting with new function.",
              name,
              category,
          )
        self._registry[category][name] = func
      return func

    return decorator

  def get(self, category: str, name: str) -> Callable[..., Any]:
    # [FUNCTION DOC]
    # Under the lock, returns the callable registered under category and name.
    # Raises LookupError (with exception chaining suppressed) if the category or
    # the name is unknown.
    """Retrieves a registered function by category and name."""
    with self._lock:
      try:
        category_funcs = self._registry[category]
      except KeyError:
        raise LookupError(f"No such category: '{category}'") from None
      try:
        return category_funcs[name]
      except KeyError:
        raise LookupError(
            f"No function named '{name}' in category '{category}'"
        ) from None

  def list_categories(self) -> List[str]:
    # [FUNCTION DOC]
    # Under the lock, returns a list of category names in the registry.
    """Lists all registered categories."""
    with self._lock:
      return list(self._registry.keys())

  def list_functions(self, category: str) -> List[str]:
    # [FUNCTION DOC]
    # Under the lock, returns the names registered in category; raises
    # LookupError if the category is unknown.
    """Lists all function names within a given category."""
    with self._lock:
      try:
        return list(self._registry[category].keys())
      except KeyError:
        raise LookupError(f"No such category: '{category}'") from None


# module-level registry instance.
default_registry = FunctionRegistry()


def get_policy_loss_fn(name: str) -> Callable[..., Any]:
  # [FUNCTION DOC]
  # Returns the default registry's policy loss function named name; raises
  # LookupError if unknown.
  """Returns the policy loss function by name."""
  return default_registry.get(_POLICY_LOSS_FN_CATEGORY, name)


def register_policy_loss_fn(
    name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  # [FUNCTION DOC]
  # Returns a decorator registering a policy loss function under name in the
  # default registry.
  """Returns a decorator to register a policy loss function by name."""
  return default_registry.register(_POLICY_LOSS_FN_CATEGORY, name)


def get_advantage_estimator(name: str) -> Callable[..., Any]:
  # [FUNCTION DOC]
  # Returns the default registry's advantage estimator named name; LookupError
  # if unknown.
  """Returns the advantage estimator function by name."""
  return default_registry.get(_ADVANTAGE_ESTIMATOR_CATEGORY, name)


def register_advantage_estimator(
    name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  # [FUNCTION DOC]
  # Returns a decorator registering an advantage estimator under name in the
  # default registry.
  """Returns a decorator to register an advantage estimator function by name."""
  return default_registry.register(_ADVANTAGE_ESTIMATOR_CATEGORY, name)


def get_reward_manager(name: str) -> Callable[..., Any]:
  # [FUNCTION DOC]
  # Returns the default registry's reward manager named name (a class or
  # callable); LookupError if unknown.
  """Returns the reward manager function by name."""
  return default_registry.get(_REWARD_MANAGER_CATEGORY, name)


def register_reward_manager(
    name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  # [FUNCTION DOC]
  # Returns a decorator registering a reward manager under name in the default
  # registry.
  """Returns a decorator to register a reward manager function by name."""
  return default_registry.register(_REWARD_MANAGER_CATEGORY, name)


def get_value_loss_fn(name: str) -> Callable[..., Any]:
  # [FUNCTION DOC]
  # Returns the default registry's value loss function named name; LookupError
  # if unknown.
  """Returns the value loss function by name."""
  return default_registry.get(_VALUE_LOSS_FN_CATEGORY, name)


def register_value_loss_fn(
    name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  # [FUNCTION DOC]
  # Returns a decorator registering a value loss function under name in the
  # default registry.
  """Returns a decorator to register a value loss function by name."""
  return default_registry.register(_VALUE_LOSS_FN_CATEGORY, name)
