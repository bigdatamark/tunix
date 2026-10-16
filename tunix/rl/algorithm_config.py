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
# Base configuration dataclass shared by all RL algorithms (GRPO, PPO, DAPO,
# ...). Provides validated string selectors for the algorithm variant, the
# advantage estimator and the policy loss function (looked up by name in
# function_registry), the reward-manager name, and an optional KL clamp.
# Uses absl logging to print the config on construction.
import dataclasses
from absl import logging

@dataclasses.dataclass(slots=True, kw_only=True)
class AlgorithmConfig:
  # [CLASS DOC]
  # Slotted, keyword-only dataclass. Fields: algo_variant (default "grpo"),
  # advantage_estimator (default "grpo"), policy_loss_fn (default "grpo"),
  # reward_manager (default "sequence-level", name of a registered reward
  # manager), kl_clamp_value (float or None; when set, per-token KL is clamped
  # symmetrically to plus/minus this value in common.compute_kl_divergence).
  # Construction validates selectors and logs all fields (see __post_init__).
  """Configuration for RL algorithms.

  Parameters:
    algo_variant: The core algorithm variant to use.
    advantage_estimator: The advantage estimator to use.
    policy_loss_fn: The policy loss function to use.
  """

  algo_variant: str = "grpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
  reward_manager: str = "sequence-level"
  # Optional symmetric clamp applied to per-token KL inside
  # `common.compute_kl_divergence`. `None` (default) disables the clamp and
  # preserves prior behavior bit-for-bit. Set to a positive float (e.g.
  # `10000.0`) to bound rare outliers — useful when the trained policy
  # briefly drifts far from the reference and the `low_var_kl` estimator's
  # `exp(diff)` term saturates bf16 / overflows fp32 and poisons the loss
  # for the rest of the step.
  kl_clamp_value: float | None = None

  def __post_init__(self):
    # [FUNCTION DOC]
    # Validation hook run after dataclass init. Checks algo_variant is one of
    # grpo, gspo-token, ppo, dapo; advantage_estimator is one of grpo, gae;
    # policy_loss_fn is one of grpo, ppo. Raises ValueError naming the allowed
    # values and the received value on the first failing check. Then calls
    # print_config (side effect: info logs). reward_manager and kl_clamp_value are
    # not validated.
    # KNOWN ISSUE: the allowed lists are hardcoded and narrower than what
    # function_registry supports (e.g. "drgrpo", "dapo" estimators); subclasses
    # such as GRPOConfig override __post_init__ without calling this, so the
    # validation and logging are silently skipped there.
    valid_algo_variants = [
        "grpo",
        "gspo-token",
        "ppo",
        "dapo",
    ]
    valid_advantage_estimators = ["grpo", "gae"]
    valid_policy_loss_fns = ["grpo", "ppo"]
    if self.algo_variant not in valid_algo_variants:
      raise ValueError(
          f"algo_variant must be one of {valid_algo_variants}. "
          f"Received: {self.algo_variant!r}"
      )
    if self.advantage_estimator not in valid_advantage_estimators:
      raise ValueError(
          f"advantage_estimator must be one of {valid_advantage_estimators}."
          f" Received: {self.advantage_estimator}"
      )
    if self.policy_loss_fn not in valid_policy_loss_fns:
      raise ValueError(
          f"policy_loss_fn must be one of {valid_policy_loss_fns}."
          f" Received: {self.policy_loss_fn}"
      )

    # Automatically prints configuration upon initialization.
    self.print_config()

  def print_config(self):
    # [FUNCTION DOC]
    # Logs, via absl info logging, a header line with the concrete class name and
    # then one line per dataclass field with its name and current value. Works
    # for subclasses since it iterates the dataclass fields of the instance.
    # Returns None.
    """Prints all configuration fields, working dynamically for child classes."""
    logging.info(f"Initializing {self.__class__.__name__}:")
    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      logging.info(f"  {field.name}: {value}")
