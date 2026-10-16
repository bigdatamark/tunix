# [MODULE DOC]
# R2E-Gym based software-engineering environment for tunix agentic RL. If r2egym is importable,
# its Action, EnvArgs and RepoEnv are used, else they are None placeholders (the env then fails at
# use time). Defines tool command file lists inside the r2egym package: R2EGYM_COMMAND_FILES
# (file_editor, search, execute_bash, finish) and SWEAGENT_COMMAND_FILES (str_replace_editor,
# execute_bash, submit), the helper _unpack_entry, and SWEEnv, which wraps an r2egym RepoEnv (a
# Docker/Kubernetes sandbox for one repository task) behind the BaseTaskEnv interface.
import json
import os
from typing import Any, Optional, cast
import numpy as np

try:
  import r2egym  # pytype: disable=import-error
  from r2egym.agenthub.action import Action  # pytype: disable=import-error
  from r2egym.agenthub.environment.env import EnvArgs, RepoEnv  # pytype: disable=import-error
except ImportError:
  r2egym = cast(Any, None)
  EnvArgs = cast(Any, None)
  RepoEnv = cast(Any, None)
  Action = cast(Any, None)

from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult


if r2egym:
  R2EGYM_PATH = os.path.dirname(r2egym.__file__)
else:
  R2EGYM_PATH = ""
# List of tools to be used in the environment.
R2EGYM_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/file_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/search.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/finish.py"),
]

SWEAGENT_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/str_replace_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/submit.py"),
]


def _unpack_entry(entry: dict) -> dict:
  # [FUNCTION DOC]
  # HELPER: returns a new dict where numpy array values are replaced by .item() (so must be
  # size-1), list values must have length 1 and are replaced by their single element (ValueError
  # otherwise), and other values are copied as is. Used to undo batch collation of a single entry.
  """Utility to clean up and unpack the dataset entry."""
  unpacked_entry = {}
  for k, v in entry.items():
    if isinstance(v, np.ndarray):
      unpacked_entry[k] = v.item()
    elif isinstance(v, list):
      if len(v) != 1:
        raise ValueError(
            f"Can only convert a list of size 1; got size {len(v)}"
        )
      unpacked_entry[k] = v[0]
    else:
      unpacked_entry[k] = v
  return unpacked_entry


class SWEEnv(BaseTaskEnv):
  # [CLASS DOC]
  # BaseTaskEnv for one R2E-Gym task. Attributes: entry (unpacked task dict), step/reward timeouts,
  # total_steps, delete_image, backend, env (RepoEnv or None until first reset), verbose, scaffold
  # ("r2egym" or "sweagent", selects tool set), final_reward_fn (env.compute_reward, set on reset),
  # extra_kwargs with group_id and pair_index.
  """Software Engineering Environment for code-related tasks."""

  def __init__(
      self,
      entry: dict,
      group_id: int | None = None,
      pair_index: int | None = None,
      step_timeout: int = 90,
      reward_timeout: int = 300,
      backend: str = "kubernetes",
      delete_image: bool = False,
      verbose: bool = False,
      scaffold: str = "r2egym",
      max_steps: int = 1,
  ):
    # [FUNCTION DOC]
    # Stores configuration (entry unpacked via _unpack_entry; step_timeout 90 s, reward_timeout 300 s,
    # backend "kubernetes", delete_image False, verbose False, scaffold "r2egym", max_steps 1), asserts
    # scaffold is r2egym or sweagent (AssertionError), calls BaseTaskEnv.__init__(max_steps), and records
    # group_id / pair_index in extra_kwargs. Does not start any container. Raises ValueError from
    # _unpack_entry for multi-element lists.
    # KNOWN ISSUE: docstring says entry may be None and pair_index random, but neither is supported; verbose,
    # scaffold and max_steps are undocumented.
    """Initialize the SWE environment.

    Args:
        entry: Dataset containing the tasks. If None, uses default dataset.
        group_id: ID of the group to which the task belongs.
        pair_index: Index of the pair to use. If None, selects a random pair.
        step_timeout: Timeout for each step in seconds.
        reward_timeout: Timeout for reward computation in seconds.
        backend: Backend to use for the environment.
        delete_image: Whether to delete the Docker image after closing.
    """
    self.entry = _unpack_entry(entry)
    self.step_timeout = step_timeout
    self.reward_timeout = reward_timeout
    self.total_steps = 0
    self.delete_image = delete_image
    self.backend = backend
    self.env = None
    self.verbose = verbose
    self.scaffold = scaffold
    assert scaffold in [
        "r2egym",
        "sweagent",
    ], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
    super().__init__(max_steps=max_steps)

    if not hasattr(self, "extra_kwargs"):
      self.extra_kwargs = {}

    self.extra_kwargs["group_id"] = group_id
    self.extra_kwargs["pair_index"] = pair_index

  def _initial_observation(self) -> Any:
    # [FUNCTION DOC]
    # HELPER: on first call creates RepoEnv(EnvArgs(ds=entry)) with backend, timeouts and verbosity;
    # on later calls resets the existing RepoEnv. Sets final_reward_fn to env.compute_reward, registers the
    # scaffold's command files, resets total_steps to 0, and returns env.get_task_instruction() (the task
    # prompt polled from the runtime). Side effects: starts/resets a container. Raises TypeError if
    # r2egym is not installed (EnvArgs is None).
    if not self.env:
      # Initialize environment if not created yet.
      env_args = EnvArgs(ds=self.entry)
      self.env = RepoEnv(
          env_args,
          backend=self.backend,
          step_timeout=self.step_timeout,
          reward_timeout=self.reward_timeout,
          verbose=self.verbose,
      )
    else:
      self.env.reset()
    self.final_reward_fn = self.env.compute_reward
    if self.scaffold == "r2egym":
      self.env.add_commands(R2EGYM_COMMAND_FILES)
    else:
      self.env.add_commands(SWEAGENT_COMMAND_FILES)
    self.total_steps = 0

    # Polls docker runtime to get task instruction.
    return self.env.get_task_instruction()

  def _step_impl(self, action: Any) -> EnvStepResult:
    # [FUNCTION DOC]
    # Converts a string action via Action.from_string (otherwise uses the object as is). If it has no
    # function_name, returns empty observation, reward 0, done False, empty info without touching the env.
    # Raises ValueError if env is not initialized. Otherwise calls env.step, increments total_steps and
    # returns EnvStepResult(str(obs), reward, done, info); RepoEnv reward is always 0, the real reward
    # comes from final_reward_fn.
    if isinstance(action, str):
      action_obj = Action.from_string(action)
    else:
      action_obj = action

    if not action_obj.function_name:
      return EnvStepResult(observation="", reward=0, done=False, info={})

    # RepoEnv always returns 0 reward, must be evaluated by DockerRuntime.
    if not self.env:
      raise ValueError("Environment not initialized")
    obs, reward, done, info = self.env.step(action_obj)

    self.total_steps += 1

    return EnvStepResult(
        observation=str(obs), reward=reward, done=done, info=info
    )

  def close(self) -> None:
    # [FUNCTION DOC]
    # Closes the RepoEnv if present; then, if delete_image and env exist, runs "docker rmi <image>" via
    # os.system. No output. Side effects: container teardown, shell command execution.
    # KNOWN ISSUE: reads env.runtime.docker_image after env.close(); env is not reset to None, and the
    # image name is interpolated into a shell command unquoted.
    """Close the environment and clean up resources."""
    if self.env is not None:
      self.env.close()

    if self.delete_image and self.env:
      docker_image = self.env.runtime.docker_image
      os.system(f"docker rmi {docker_image}")

  @staticmethod
  def from_dict(extra_info: dict | str) -> "SWEEnv":
    # [FUNCTION DOC]
    # Static factory: JSON-decodes extra_info if it is a string, copies any keys that match SWEEnv.__init__
    # parameter names into kwargs, sets entry to the whole extra_info dict, and constructs SWEEnv.
    # Output: SWEEnv. Raises json errors for invalid JSON and errors from __init__.
    """Create an environment instance from JSON configuration.

    Args:
        extra_info: Dictionary containing configuration parameters. The entire
          dict will be used as 'entry', and any keys matching __init__
          parameters will be extracted and passed.

    Returns:
        Initialized SWEEnv instance
    """
    import inspect

    if isinstance(extra_info, str):
      extra_info = json.loads(extra_info)

    sig = inspect.signature(SWEEnv.__init__)
    init_params = {}
    for param_name, param in sig.parameters.items():
      if param_name == "self":
        continue
      if param_name in extra_info:
        init_params[param_name] = extra_info[param_name]
      # else if param has default value, use the default value
    init_params["entry"] = extra_info
    return SWEEnv(**init_params)
