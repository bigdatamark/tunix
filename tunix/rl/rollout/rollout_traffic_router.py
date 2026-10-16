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
# HELPER: Small interface module for routing rollout traffic across multiple
# rollout engines. Defines a keyword-only dataclass of routing parameters and
# an abstract router type whose single method maps parameters to an engine
# index. Aliases ABC and abstractmethod from the abc module. No side effects.
"""Rollout traffic router interface."""

import abc
import dataclasses

ABC = abc.ABC
abstractmethod = abc.abstractmethod


@dataclasses.dataclass(kw_only=True)
class RolloutTrafficRoutingParams:
  # [CLASS DOC]
  # Keyword-only dataclass carrying inputs used to select a rollout engine.
  # Field trajectory_id (int, required): identifier of the trajectory being
  # generated, allowing sticky per-trajectory routing. No validation.
  """Parameters for rollout traffic routing."""

  # trajectory id, used for per-trajectory routing.
  trajectory_id: int


class RolloutTrafficRouter:
  # [CLASS DOC]
  # Abstract interface for rollout traffic routers. Subclasses implement route.
  # KNOWN ISSUE: the class does not inherit from ABC (nor use ABCMeta), so the
  # abstractmethod decorator is not enforced and the class can be instantiated
  # directly; calling route on it then returns None.

  @abstractmethod
  def route(self, params: RolloutTrafficRoutingParams) -> int:
    # [FUNCTION DOC]
    # Abstract. Input: a RolloutTrafficRoutingParams. Output: integer index of the
    # rollout engine that should serve the request. Base body only holds a
    # docstring and returns None. No side effects expected.
    """returns the rollout engine id."""
