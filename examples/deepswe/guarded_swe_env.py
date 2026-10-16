"""SWE environment wrapper with lightweight action guarding."""
# [MODULE DOC]
# Defines GuardedSWEEnv, an SWEEnv whose step() first runs the model's action string through an
# ActionGuard (from action_guard) that can block missing or repeated/failing tool calls before they
# reach the Docker sandbox. Imports action_guard and swe_env as top-level modules, so the examples/
# deepswe directory must be on sys.path.

from action_guard import ActionGuard, GuardConfig
from swe_env import SWEEnv
from tunix.rl.agentic.environments.base_environment import EnvStepResult


class GuardedSWEEnv(SWEEnv):
  # [CLASS DOC]
  # SWEEnv subclass holding self.guard (an ActionGuard). Blocked steps return reward 0, done False and
  # info with guard_blocked True and guard_reason; allowed steps delegate to SWEEnv and the guard records
  # the outcome observation.
  """SWEEnv with failure-aware action restrictions applied in step()."""

  def __init__(self, *args, guard_config=None, **kwargs):
    # [FUNCTION DOC]
    # Creates the ActionGuard from guard_config (or a default GuardConfig) before calling SWEEnv.__init__
    # with all other positional and keyword args. No output; raises whatever SWEEnv.__init__ raises.
    self.guard = ActionGuard(guard_config or GuardConfig())
    super().__init__(*args, **kwargs)

  def _initial_observation(self):
    # [FUNCTION DOC]
    # HELPER: resets the guard's per-episode state and returns SWEEnv's initial observation (task
    # instruction), which may (re)create the Docker environment.
    self.guard.reset()
    return super()._initial_observation()

  def _step_impl(self, action):
    # [FUNCTION DOC]
    # For string actions: parses the function name via the guard's private parser; if none, returns a
    # blocked result with a message demanding exactly one XML tool call (reason "missing_function_call").
    # Otherwise asks the guard to evaluate the action; if blocked, returns the guard's message with its
    # reason. Non-blocked or non-string actions are executed by SWEEnv._step_impl; for string actions the
    # guard then records (action, observation string). Output: EnvStepResult.
    if isinstance(action, str):
      func_name, _ = self.guard._parse_action(action)  # pylint: disable=protected-access
      if not func_name:
        return EnvStepResult(
            observation=(
                "[ACTION GUARD] Your previous response did not include a"
                " valid function call. You must output exactly one tool call"
                " in the required XML format."
            ),
            reward=0.0,
            done=False,
            info={
                "guard_blocked": True,
                "guard_reason": "missing_function_call",
            },
        )

      verdict = self.guard.evaluate(action)
      if verdict.blocked:
        return EnvStepResult(
            observation=verdict.message,
            reward=0.0,
            done=False,
            info={
                "guard_blocked": True,
                "guard_reason": verdict.reason,
            },
        )

    result = super()._step_impl(action)
    if isinstance(action, str):
      self.guard.record_outcome(action, str(result.observation))
    return result
