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
# Dr. GRPO ("GRPO done right", arXiv 2503.20783) configuration and learner.
# Defines DrGRPOConfig, a GRPOConfig subclass that pins the algorithm variant
# and advantage estimator to "drgrpo" and defaults loss aggregation to
# "sequence-mean-token-scale", and DrGRPOLearner, an unchanged subclass of the
# GRPO learner parameterized with that config. Re-exports TrainingInputT,
# RewardFn and MetricFn aliases from rl_learner. The jax, algo_core and
# function_registry imports are unused here (algo_core import ensures the
# registered "drgrpo" functions exist). No other side effects.
# KNOWN ISSUE: the module docstring says "Helper functions for GRPO Trainer",
# which does not describe the contents.
"""Helper functions for GRPO Trainer."""

import dataclasses
import jax
from tunix.rl import algo_core
from tunix.rl import function_registry
from tunix.rl import rl_learner
from tunix.rl.grpo import grpo_learner as grpo_learner_lib

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


@dataclasses.dataclass(kw_only=True)
class DrGRPOConfig(grpo_learner_lib.GRPOConfig):
  # [CLASS DOC]
  # Keyword-only dataclass config for Dr. GRPO. Inherits every GRPOConfig field
  # and its validation (num_generations > 1, loss_algo in grpo/gspo-token).
  # Overrides: algo_variant fixed to "drgrpo" and advantage_estimator fixed to
  # "drgrpo" (both excluded from the constructor), loss_agg_mode default
  # "sequence-mean-token-scale" (no per-length normalization).
  # KNOWN ISSUE: "drgrpo" is not in AlgorithmConfig's valid algo_variant or
  # advantage_estimator lists; it only works because GRPOConfig.__post_init__
  # does not call the base validation.
  """Configuration for DrGRPO."""

  # https://arxiv.org/abs/2503.20783
  algo_variant: str = dataclasses.field(default="drgrpo", init=False)
  advantage_estimator: str = dataclasses.field(default="drgrpo", init=False)
  loss_agg_mode: str = dataclasses.field(default="sequence-mean-token-scale")


class DrGRPOLearner(grpo_learner_lib.GrpoLearner[DrGRPOConfig]):
  # [CLASS DOC]
  # Dr. GRPO learner: identical behavior to GRPOLearner; the differences come
  # entirely from DrGRPOConfig (the "drgrpo" advantage estimator, which skips
  # std normalization, and the token-scale loss aggregation). Adds no methods.
  """Dr. GRPO Learner."""
