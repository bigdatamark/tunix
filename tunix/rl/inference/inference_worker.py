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
# Inference worker that hosts the non-trained auxiliary models of an RL setup
# (critic, reference, reward) and exposes forward computations on them using
# helpers from tunix.rl.common: scalar scores, reference per-token log-probs
# and critic values. No import-time side effects.
"""Inference worker for RL."""

from flax import nnx
import jax
from tunix.rl import common


class InferenceWorker:
  # [CLASS DOC]
  # Holds a dict of role name to nnx.Module (allowed roles: "critic",
  # "reference", "reward") plus, per role, the (graphdef, state) pair obtained
  # by splitting the module. Only one model per role.
  """Inference worker hosting critic, reference and reward models."""

  def __init__(self, models: dict[str, nnx.Module]):
    # [FUNCTION DOC]
    # Input: models dict mapping role to module. Raises ValueError if any key is
    # not critic, reference or reward. Stores the dict and, for each model, the
    # nnx split result (graph definition and state) in a private mapping.
    # Splitting does not copy weights.
    for k in models.keys():
      if k not in ["critic", "reference", "reward"]:
        raise ValueError(
            f"Model role {k} is not supported. Supported models are critic,"
            " reference and reward."
        )
    self._models = models
    self._model_states = {}
    for k, m in models.items():
      self._model_states[k] = nnx.split(m)
    # TODO(tsbao): support multiple reward models.

  def get_rewards(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    # [FUNCTION DOC]
    # Inputs: prompt token array, completion token array, pad id, eos id. Looks up
    # the "reward" model; raises ValueError if absent. Returns the result of
    # common.compute_score on the reward model and the tokens (per-sequence
    # reward scores as a jax array).
    reward_model = self._models.get("reward")
    if reward_model is None:
      raise ValueError("Reward model is not available.")
    return common.compute_score(
        reward_model, prompt_tokens, completion_tokens, pad_id, eos_id
    )

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      temperature: float = 1.0,
  ) -> jax.Array:
    # [FUNCTION DOC]
    # Inputs: prompt tokens, completion tokens, pad id, eos id, temperature
    # (default 1.0). Uses the stored reference graphdef/state and returns
    # common.compute_per_token_logps with gradients stopped and logits not
    # returned: array of log-probs of each completion token under the reference
    # model, shape batch by completion length.
    # KNOWN ISSUE: when no reference model exists the dict lookup returns None and
    # tuple unpacking raises TypeError before the intended ValueError is reached.
    graphdef, state = self._model_states.get("reference")
    if graphdef is None:
      raise ValueError("Reference model is not available.")
    return common.compute_per_token_logps(
        graphdef,
        state,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        stop_gradient=True,
        return_logits=False,
        temperature=temperature,
    )

  def get_values(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    # [FUNCTION DOC]
    # Inputs: prompt tokens, completion tokens, pad id, eos id. Merges the stored
    # critic graphdef/state back into a module and returns common.compute_score
    # on it (per-token value estimates from the critic head).
    # KNOWN ISSUE: if no critic exists, unpacking None raises TypeError; the
    # None check after nnx.merge can never trigger.
    graphdef, state = self._model_states.get("critic")
    critic_model = nnx.merge(graphdef, state)
    if critic_model is None:
      raise ValueError("Critic model is not available.")
    return common.compute_score(
        critic_model,
        prompt_tokens,
        completion_tokens,
        pad_id,
        eos_id,
    )

  def get_model(self, role: str) -> nnx.Module:
    # [FUNCTION DOC]
    # Returns the original module registered for role; raises ValueError if the
    # role was not provided at construction.
    if role not in self._models:
      raise ValueError(f"Model role {role} is not available.")
    return self._models[role]
