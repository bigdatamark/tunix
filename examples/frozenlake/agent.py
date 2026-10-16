# [MODULE DOC]
# Conversation agent for the FrozenLake example. Defines two system prompts (SYSTEM_PROMPT: rules,
# symbols, valid actions Up/Down/Left/Right, answer inside triple backticks; MULTI_SHOT_SYSTEM_PROMPT:
# the same plus five worked examples) and FrozenLakeAgent, which turns env text observations into
# user messages and parses the model's last triple-backtick block into an env action string "1".."4"
# (Left=1, Down=2, Right=3, Up=4) or "0" (invalid).
import re
from typing import Any, Optional

from examples.frozenlake.env import FrozenLakeEnv
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent

# Prompting format inspired by the RAGEN project: https://github.com/RAGEN-AI/RAGEN
SYSTEM_PROMPT: str = """You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.
You should be aware that frozen tiles can be slippery, but the chance is small and you should not overthink it.

Please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""

MULTI_SHOT_SYSTEM_PROMPT: str = """You are a helpful assistant. You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.

Below are examples for an interaction:
Example1:
User: Current Observation:
P   _   _   _   _
O   _   _   O   _
O   _   O   _   _
O   _   _   G   _
_   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now at the top right corner. It should reach G at the bottom right corner. I should move it closer to it. I can move right or down but there is a hole in down position and I can not move diagonally. There is no hole in my next movement right so I can move to right. Action: ```Right```

Example2:
User: Current Observation:
_   _   _   _
_   _   _   O
_   O   _   P
O   _   _   G
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now at the near G. It should reach G to its bottom. I should move to be on it. There is no hole in my next movement so I can move to down. Action: ```Down```

Example3:
User: Current Observation:
_   _   _   O   _
O   _   P   O   _
O   _   O   _   _
O   _   _   G   _
_   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: G is at the bottom right relative to P. I want to move closer so I should move right or down. But there is a hole at each position and I do not want to fall into holes. Up and left are both valid but left brings me closer. Action: ```Left```

Example4:
User: Current Observation:
_   _   _   _
_   _   _   O
_   O   _   O
O   G   P   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now near G. But game has not finished. P is not at G and I should never output invalid action. I need to recheck my understanding. P is not actually on G yet because they are not overlapping, it needs reach G to its left. Action: ```Left```

Example5:
User: Current Observation:
_   _   _   O   _
O   _   P   _   _
O   _   O   O   O
O   _   O   G   _
O   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: G is at the bottom right corner of P. I can move left, right, or up. Move right will initially bring me closer but I can't reach G that way. Move up and left means I can still reach G. Move up will result in 9 steps in total while left is 7 steps. I need to move left. Action: ```Left```

Now it is your turn, please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""


class FrozenLakeAgent(base_agent.ConversationAgentBase):
  # [CLASS DOC]
  # Subclass of tunix ConversationAgentBase. State: multistep_prompt flag, last_observation (previous
  # formatted observation string, used to detect no-op moves), plus base-class _messages, _trajectory,
  # cur_step and step counter.

  def __init__(
      self,
      system_prompt: Optional[str] = None,
      use_multistep_prompt: bool | None = True,
  ):
    # [FUNCTION DOC]
    # Chooses MULTI_SHOT_SYSTEM_PROMPT when use_multistep_prompt is truthy (default True), else
    # SYSTEM_PROMPT, calls the base constructor with it, and sets last_observation to None.
    # KNOWN ISSUE: the system_prompt argument is ignored (always overwritten).
    self.multistep_prompt = use_multistep_prompt
    system_prompt = (
        SYSTEM_PROMPT
        if not self.multistep_prompt
        else MULTI_SHOT_SYSTEM_PROMPT
    )
    super().__init__(system_prompt=system_prompt)
    self.last_observation = None

  def _init_messages(self, system_prompt: str) -> None:
    # [FUNCTION DOC]
    # HELPER: resets _messages to a single system message with system_prompt (empty string if falsy).
    """Initialize conversation history with a system prompt.

    Subclasses may override this to inject additional content (e.g., tool
    documentation) into the initial system message.

    Args:
      system_prompt: The system prompt to use.
    """
    self._messages = [{"role": "system", "content": system_prompt or ""}]

  def update_from_env(
    self,
    observation: Any,
    reward: float,
    done: bool,
    info: dict[str, Any] | None = None,
    **kwargs,
  ) -> None:
    # [FUNCTION DOC]
    # Formats observation as "Current Observation: \n" plus str(observation); if not done appends a line
    # saying the goal is not achieved yet. If identical to last_observation, appends a warning that the last
    # response was invalid / position did not change. Stores it as last_observation, calls the base
    # update_from_env with the formatted string (which appends the user message via
    # _observation_to_messages), then sets cur_step to a new Step with that observation.
    # Extra kwargs ignored. Returns None.
    new_obs_str = str(observation)
    # Base message for the user
    new_obs_str = "Current Observation: \n" + new_obs_str
    if not done:
      new_obs_str += "\n" + "You have not achieved the goal, P has not reached G yet. Please give the next action."

    # Check if the observation is the same as the previous step's observation
    if self.last_observation and self.last_observation == new_obs_str:
      new_obs_str += "\nYour last response is invalid. Your position didn't change at all. You may need to recheck your thinking process, action outputted, and the format of response. Remember, you should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```."
    self.last_observation = new_obs_str

    super().update_from_env(new_obs_str, reward, done, info)
    self.cur_step = agent_types.Step(observation=new_obs_str)

  def _observation_to_messages(
      self, observation: Any, reward: float, done: bool, info: dict[str, Any]
  ) -> None:
    # [FUNCTION DOC]
    # HELPER: appends {"role": "user", "content": str(observation)} to _messages. Other args unused.
    self._messages.append({"role": "user", "content": str(observation)})

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    # [FUNCTION DOC]
    # Parses a model response. Finds all non-greedy triple-backtick blocks (dotall); if any, takes the
    # last one stripped, sets thought to the response text before its last occurrence, and lowercases it:
    # a direction word maps via left 1, down 2, right 3, up 4; a digit string in 1..4 is used directly;
    # anything else (or no block) leaves the action "0" (INVALID_ACTION). Appends the assistant message,
    # appends cur_step to the trajectory and fills thought, action (string), model_response; increments
    # step. Output: Action wrapping the action string. Side effects: mutates history/trajectory.
    # KNOWN ISSUE: appends self.cur_step without resetting it, so if called twice without an
    # update_from_env the same Step object is appended twice and overwritten.
    DIRECTION_MAP = {"left": 1, "down": 2, "right": 3, "up": 4}

    thought = response
    action_str = str(FrozenLakeEnv.INVALID_ACTION)

    matches = re.findall(r"```(.*?)```", response, re.DOTALL)

    if matches:
      last_match_content = matches[-1].strip()
      last_match_index = response.rfind(f"```{last_match_content}```")
      if last_match_index != -1:
        thought = response[:last_match_index].strip()

      extracted_text = last_match_content.lower()

      if extracted_text in DIRECTION_MAP:
        action_str = str(DIRECTION_MAP[extracted_text])
      elif extracted_text.isdigit() and int(extracted_text) in DIRECTION_MAP.values():
        action_str = str(int(extracted_text))

    # Add assistant's response to conversation history.
    self._messages.append({"role": "assistant", "content": response})

    self._trajectory.steps.append(self.cur_step)
    # Record complete step with conversation context and parsed action.
    cur_step = self._trajectory.steps[-1]
    cur_step.thought = thought
    cur_step.action = action_str
    cur_step.model_response = response

    self.step += 1
    return agent_types.Action(action=cur_step.action)

  def reset(self) -> None:
    # [FUNCTION DOC]
    # Resets base-agent state and clears last_observation.
    super().reset()
    self.last_observation = None
