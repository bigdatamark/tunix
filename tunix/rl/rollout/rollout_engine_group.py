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
# Placeholder scaffolding for a group of multiple rollout engines behind one
# rollout-like interface. Defines empty routing / weight-sync config types, a
# frozen group config dataclass, and RolloutEngineGroup whose methods mirror
# BaseRollout but are not implemented yet. No import-time side effects.
"""Rollout worker group."""

import dataclasses
import enum
from typing import Any, Dict, Optional, Tuple

import jax
import jaxtyping
from tunix.rl.rollout import base_rollout


class RolloutTrafficRoutingConfig(enum.Enum):
  # [CLASS DOC]
  # Enum intended to name the traffic routing algorithm. Currently has no
  # members (placeholder); cannot be meaningfully instantiated.
  """Rollout traffic routing algorithm."""

  pass


@dataclasses.dataclass(kw_only=True, frozen=True)
class RolloutWeightSyncConfig:
  # [CLASS DOC]
  # Frozen keyword-only placeholder dataclass for the weight-sync strategy; no
  # fields yet.
  """Configuration for the weight sync strategy."""

  pass


@dataclasses.dataclass(kw_only=True, frozen=True)
class RolloutEngineGroupConfig:
  # [CLASS DOC]
  # Frozen keyword-only dataclass. Fields (all required): num_engines (total
  # engine count), engine_config_template (RolloutConfig applied to every
  # engine), traffic_routing_config (routing algorithm; comment says default is
  # round-robin by trajectory id), weight_sync_config (weight sync strategy).
  # No validation.
  # KNOWN ISSUE: comments mention round-robin defaults but no defaults exist
  # and the routing enum has no members.
  """Configuration for the rollout engine group."""

  # number of total rollout engines
  num_engines: int

  # config template for all engines
  engine_config_template: base_rollout.RolloutConfig

  # specify traffic routing algorithm
  # by default it's round-robin (based on trajectory-id, not per-request)
  traffic_routing_config: RolloutTrafficRoutingConfig

  # specify weight sync algorithm, by default it's round-robin
  weight_sync_config: RolloutWeightSyncConfig


class RolloutEngineGroup:
  # [CLASS DOC]
  # Intended to own several rollout engines and route generation requests to
  # them. Currently a stub: every operation except model raises
  # NotImplementedError. No constructor or state.
  """RolloutEngineGroup manages multiple rollout engines, and route traffic to them based on routing config."""

  def generate(
      self, prompts, rollout_config: base_rollout.RolloutConfig, **kwargs
  ) -> base_rollout.RolloutOutput:
    # [FUNCTION DOC]
    # Stub for batched generation (prompts, rollout_config, extra kwargs).
    # Always raises NotImplementedError.
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
      **kwargs
  ) -> jax.Array:
    # [FUNCTION DOC]
    # Stub for computing per-token log-probs of completions given prompts and an
    # optional completion mask. Always raises NotImplementedError.
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    # [FUNCTION DOC]
    # Stub for pushing new parameters (optionally filtered by variable types)
    # into all engines. Always raises NotImplementedError.
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def pad_id(self) -> int:
    # [FUNCTION DOC]
    # Stub for the pad token id. Always raises NotImplementedError.
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def eos_id(self) -> int:
    # [FUNCTION DOC]
    # Stub for the EOS token id. Always raises NotImplementedError.
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def model(self) -> Any:
    # [FUNCTION DOC]
    # Returns None. Only caller is CPU-offloading logic, which is disabled when an
    # engine group is used, so no single underlying model is exposed.
    # if use RolloutEngineGroup, we disable CPU offloading, which
    # is the only caller of this function.
    return None
