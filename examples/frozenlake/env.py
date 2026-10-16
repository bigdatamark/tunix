"""DISCLAIMER:

This implementation is based on the Gymnasium FrozenLake environment and the
RAGEN project:
- Gymnasium: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- RAGEN:
https://github.com/RAGEN-AI/RAGEN/blob/main/ragen/env/frozen_lake/env.py

 Some components have been modified or extended for custom use in this project.
"""
# [MODULE DOC]
# FrozenLake grid-world environment for tunix agentic RL, adapted from Gymnasium FrozenLake and RAGEN.
# Provides a valid-random-map generator and FrozenLakeEnv, which combines tunix BaseTaskEnv (step/reset
# protocol with EnvStepResult, max_steps) with Gymnasium's FrozenLakeEnv (transition dynamics). Text
# observations are grids of tokens P (player), _ (frozen), O (hole), G (goal), X (player in hole),
# and a check mark (player on goal), tab-separated per cell, rows separated by newlines.
# Module global MAX_STEPS (default 5) bounds both the DFS path length used to validate maps and the
# env's step budget; it is overwritten by every FrozenLakeEnv construction (global side effect).

import copy
from typing import Any, Dict
from absl import logging
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
from gymnasium.utils import seeding
import numpy as np
from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult

MAX_STEPS: int = 5


# DFS to check that it's a valid path.
def is_valid(board: list[list[str]], max_size: int) -> bool:
  # [FUNCTION DOC]
  # HELPER: depth-first search from the single "S" cell of board (list of rows of single-char strings)
  # over 4-neighbours within a max_size x max_size grid, not entering "H" cells, abandoning paths longer
  # than MAX_STEPS steps. Returns True as soon as a neighbour is "G", else False. Each cell is expanded
  # at most once (first visit wins, so a shorter path found later may be missed).
  # Raises IndexError if there is no "S".
  # KNOWN ISSUE: because visited cells are marked regardless of step count, DFS order can reject maps
  # that do have a path within MAX_STEPS.
  frontier, discovered = [], set()
  # find the start point
  start_r, start_c = np.where(np.array(board) == "S")
  frontier.append((start_r[0], start_c[0], 0))  # row, col steps
  # dfs to check if there is a path from start to goal
  while frontier:
    r, c, steps = frontier.pop()
    if steps > MAX_STEPS:
      continue

    if (r, c) not in discovered:
      discovered.add((r, c))
      directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
      for x, y in directions:
        r_new = r + x
        c_new = c + y
        if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
          continue
        if board[r_new][c_new] == "G":
          return True
        if board[r_new][c_new] != "H":
          frontier.append((r_new, c_new, steps + 1))
  return False


def generate_random_map(
    size: int = 8, p: float = 0.8, seed: int = 0
) -> tuple[list[str], tuple[int, int]]:
  # [FUNCTION DOC]
  # Generates a reproducible valid map. Uses a Gymnasium seeded numpy Generator from seed; loops:
  # p is clamped to at most 1, a size x size board is sampled with "F" with probability p and "H"
  # otherwise, start and goal positions are sampled uniformly until distinct, written as "S" and "G",
  # and is_valid is checked; repeats until valid. Output: (list of row strings, (goal_row, goal_col)).
  # May loop forever if no valid map is attainable (e.g. very low p). Docstring omits the goal tuple.
  """Generates a random valid map (one that has a path from start to goal)

  Args:
    size: size of each side of the grid
    p: probability that a tile is frozen
    seed: seed to ensure the generation of reproducible maps

  Returns:
    A random valid map
  """
  valid = False
  board: list[list[str]] = []  # initialize to make pyright happy

  np_random, _ = seeding.np_random(seed)

  # generate random start and end points
  goal_r, goal_c = -1, -1
  while not valid:
    p = min(1, p)
    board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p]).tolist()
    while True:
      start_r = int(np_random.integers(0, size))
      start_c = int(np_random.integers(0, size))
      goal_r = int(np_random.integers(0, size))
      goal_c = int(np_random.integers(0, size))
      # Ensure start and goal are different positions
      if (start_r, start_c) != (goal_r, goal_c):
        break
    board[start_r][start_c] = "S"
    board[goal_r][goal_c] = "G"
    valid = is_valid(board, size)
  return ["".join(x) for x in board], (goal_r, goal_c)


def get_goal_position(random_map):
  # [FUNCTION DOC]
  # HELPER: given a 2-D bytes array map, returns (row, col) of the first b"G" cell (numpy ints) or None.
  positions = np.argwhere(random_map == b"G")
  if positions.size == 0:
    return None  # G not found
  return tuple(positions[0])  # returns (row, col)


class FrozenLakeEnv(BaseTaskEnv, GymFrozenLakeEnv):
  # [CLASS DOC]
  # FrozenLake env usable by tunix agents. Actions are ints 1..4 (Left, Down, Right, Up) mapped to gym
  # actions 0..3; 0 (or any falsy action) is a no-op "invalid" action with reward 0. Rewards come from
  # Gym: 1 on reaching the goal, else 0. Episode is done on goal or hole (Gym termination) or by
  # BaseTaskEnv max steps. Class constants: MAP_LOOKUP (byte tile to int: P0 F1 H2 G3), GRID_LOOKUP
  # (int to rendered cell, 4 = in hole, 5 = on goal), ACTION_LOOKUP (names), INVALID_ACTION = 0,
  # PENALTY_FOR_INVALID = -1.
  # KNOWN ISSUE: docstring says invalid actions get -0.1 and the action range is {0,3}; code gives 0.0
  # reward for invalid actions, PENALTY_FOR_INVALID is unused, and valid actions are 1..4.
  """Inherits from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv

  ## Description
  The game starts with the player at random location of the frozen lake grid
  world with the
  goal located at another random location for the 4x4 environment.

  ## Action Space
  The action shape is `(1,)` in the range `{0, 3}` indicating
  which direction to move the player.
  NOTE the action space is different from
  gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv, start from 1
  - 0: Still
  - 1: Left
  - 2: Down
  - 3: Right
  - 4: Up

  ## Starting State
  The episode starts with the player at random location

  ## Rewards
  NOTE added -0.1 as penalty for invalid action
  Reward schedule:
  - Reach goal: +1
  - Reach hole: 0
  - Reach frozen: 0

  ## Arguments
  `is_slippery`: if action is left and is_slippery is True, then:
  - P(move left)=1/3
  - P(move up)=1/3
  - P(move down)=1/3

  ## Example
  P   _   _   _
  _   _   _   O
  O   _   O   _
  O   _   _   G
  """

  # Map gym state in integer
  MAP_LOOKUP = {
      b"P": 0,
      b"F": 1,
      b"H": 2,
      b"G": 3,
  }

  # Define rules to transform to rendered text observation of the environment
  GRID_LOOKUP = {
      0: " P \t",  # player
      1: " _ \t",  # frozen
      2: " O \t",  # hole
      3: " G \t",  # goal
      4: " X \t",  # player fall into hole
      5: " √ \t",  # player on goal
  }

  ACTION_LOOKUP = {
      0: "None",
      1: "Left",
      2: "Down",
      3: "Right",
      4: "Up",
  }

  INVALID_ACTION = 0
  PENALTY_FOR_INVALID = -1

  def __init__(
      self,
      entry: dict[str, Any],
      group_id: int | None = None,
      pair_index: int | None = None,
      max_steps: int = 5,
      **kwargs,
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: entry (dict with optional numpy-scalar "seed" (default 42), "size" (8),
    # "p" (0.8), each read via .item()), group_id and pair_index (stored in extra_kwargs), max_steps
    # (default 5; also assigned to the module global MAX_STEPS), kwargs may contain "desc" (explicit map
    # as list of strings) and "is_slippery" (default False); other kwargs are ignored. Without desc, a
    # random valid map and goal are generated; with desc, the map is turned into a bytes char array and the
    # goal located. Stores goal_postion (sic), initializes BaseTaskEnv(max_steps) and Gym FrozenLakeEnv
    # (desc, is_slippery), sets ACTION_SPACE to Discrete(4, start=1), map_kwargs, env_kwargs, action_map
    # (1..4 to 0..3), reward 0, empty _valid_actions, and extra_kwargs group/pair ids.
    # Raises AttributeError if seed/size/p are plain Python numbers (no .item()).
    # KNOWN ISSUE: mutates the module-global MAX_STEPS, affecting all other instances and map validation.
    global MAX_STEPS
    MAX_STEPS = max_steps

    desc = kwargs.pop("desc", None)
    is_slippery = kwargs.pop("is_slippery", False)
    self.seed = entry["seed"].item() if "seed" in entry else 42
    self.size = entry["size"].item() if "size" in entry else 8
    self.p = entry["p"].item() if "p" in entry else 0.8

    if desc is None:
      random_map, goal_position = generate_random_map(
          size=self.size, p=self.p, seed=self.seed
      )
    else:
      random_map = np.asarray(copy.deepcopy(desc), dtype="c")
      goal_position = get_goal_position(random_map)

    self.goal_postion = goal_position

    BaseTaskEnv.__init__(self, max_steps=MAX_STEPS)
    GymFrozenLakeEnv.__init__(self, desc=random_map[:], is_slippery=is_slippery)

    self.ACTION_SPACE = gym.spaces.Discrete(4, start=1)

    self.map_kwargs = {
        "size": self.size,
        "p": self.p,
    }
    self.env_kwargs = {
        "is_slippery": is_slippery,
        "desc": copy.deepcopy(desc),
        "seed": self.seed,
    }
    self.action_map = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
    }

    self.reward = 0
    self._valid_actions = []

    if not hasattr(self, "extra_kwargs"):
      self.extra_kwargs = {}
    self.extra_kwargs["group_id"] = group_id
    self.extra_kwargs["pair_index"] = pair_index

  def _get_player_position(self):
    # [FUNCTION DOC]
    # HELPER: returns (row, col) of the player from Gym's flat state index self.s and column count ncol.
    return (self.s // self.ncol, self.s % self.ncol)

  def _initial_observation(self) -> Any:
    # [FUNCTION DOC]
    # HELPER: resets the Gym env with self.seed, zeroes reward and valid actions, returns the
    # tiny_rgb_array text rendering as the initial observation.
    GymFrozenLakeEnv.reset(self, seed=self.seed)
    self.reward = 0
    self._valid_actions = []
    init_observation = self.render(mode="tiny_rgb_array")
    return init_observation

  def finished(self):
    # [FUNCTION DOC]
    # Returns True if the player's current tile is G or H (episode terminal), else False.
    player_pos = self._get_player_position()
    return self.desc[player_pos] in b"GH"

  def success(self):
    # [FUNCTION DOC]
    # Returns True if the player's current tile is G.
    # KNOWN ISSUE: docstring says goal or hole, but only the goal counts.
    """Check if the agent has reacched the goal (G) or hole (H)"""
    player_pos = self._get_player_position()
    return self.desc[player_pos] in b"G"

  def step(self, action: Any) -> tuple[Any, float, bool, Dict[str, Any]]:  # type: ignore[signature-mismatch]
    # [FUNCTION DOC]
    # Delegates to BaseTaskEnv.step (which handles step counting/max steps and calls _step_impl).
    # Output: whatever BaseTaskEnv.step returns (observation, reward, done, info).
    return BaseTaskEnv.step(self, action)

  def _step_impl(self, action: Any) -> EnvStepResult:
    # [FUNCTION DOC]
    # Performs one step. If already on the goal, returns the current render with reward 1.0, done True,
    # action_is_effective False. Falsy action becomes INVALID_ACTION; action is cast to int (ValueError for
    # non-numeric strings). Invalid action returns render, reward 0.0, done False, not effective. Otherwise
    # maps the action via action_map (KeyError for values outside 1..4), steps Gym, and returns render,
    # float reward, Gym terminated flag as done, and action_is_effective = whether the state index changed.
    """- Map custom action to gymnasium FrozenLakeEnv action and take the step

    - Check if the action is effective (whether player moves in the env).
    """
    if self.success():
      return EnvStepResult(
          observation=self.render(),
          reward=1.0,
          done=True,
          info={"action_is_effective": False},
      )

    if not action:
      action = self.INVALID_ACTION
    action = int(action)

    assert isinstance(action, int), "Action must be an integer"
    assert not self.success(), "Agent has already reached the goal or hole"

    if action == self.INVALID_ACTION:
      return EnvStepResult(
          observation=self.render(),
          reward=0.0,
          done=False,
          info={"action_is_effective": False},
      )

    prev_player_position = int(self.s)

    player_pos, reward, done, _, prob = GymFrozenLakeEnv.step(
        self, int(self.action_map[action])
    )

    obs = self.render()
    return EnvStepResult(
        observation=obs,
        reward=float(reward),
        done=done,
        info={"action_is_effective": prev_player_position != int(player_pos)},
    )

  def render(self, mode="tiny_rgb_array"):
    # [FUNCTION DOC]
    # Renders the env. mode must be tiny_rgb_array (default), list, state, rgb_array or ansi
    # (AssertionError otherwise). rgb_array/ansi temporarily set Gym's render_mode and return Gym's render.
    # Otherwise copies desc, replaces S by F and puts P at the player position. "state" returns an int
    # array via MAP_LOOKUP, with 4 if the player is on a hole and 5 on the goal. "list" returns a list of
    # row strings of stripped cell symbols joined by spaces. "tiny_rgb_array" returns the GRID_LOOKUP cells
    # concatenated per row, rows joined by newlines.
    assert mode in ["tiny_rgb_array", "list", "state", "rgb_array", "ansi"]
    if mode in ["rgb_array", "ansi"]:
      prev_render_mode = self.render_mode
      self.render_mode = mode
      obs = GymFrozenLakeEnv.render(self)
      self.render_mode = prev_render_mode
      return obs
    room_state = copy.deepcopy(self.desc)

    # replace the position of start 'S' with 'F'
    position_S = np.where(room_state == b"S")
    room_state[position_S] = b"F"

    # replace the position of the player with 'P'
    position_P = self._get_player_position()
    room_state[position_P] = b"P"

    if mode == "state":
      # transform 'S', 'F', 'H', 'G' to numpy integer array
      room_state = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room_state)
      # add player in hole or player on goal
      if self.desc[position_P] == b"H":
        room_state[position_P] = 4
      elif self.desc[position_P] == b"G":
        room_state[position_P] = 5
      return room_state

    room_state = self.render(mode="state").tolist()

    if mode == "list":
      lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
      return [" ".join(lookup(cell) for cell in row) for row in room_state]

    if mode == "tiny_rgb_array":
      lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
      result = "\n".join(
          "".join(lookup(cell) for cell in row) for row in room_state
      )
      # result += f"Player Position is at ({position_P[0]}, {position_P[1]}), Goal Position is at ({self.goal_postion[0]}, {self.goal_postion[1]})"
      return result

  def reset(self):  # type: ignore[signature-mismatch]
    # [FUNCTION DOC]
    # Resets BaseTaskEnv state and Gym state with self.seed (same map and start every time) and returns
    # (tiny_rgb_array observation, empty info dict).
    BaseTaskEnv.reset(self)
    GymFrozenLakeEnv.reset(self, seed=self.seed)
    return self.render(mode="tiny_rgb_array"), {}

  @classmethod
  def from_dict(cls, env_info: dict) -> "FrozenLakeEnv":
    # [FUNCTION DOC]
    # Builds an env from a dict: entry=env_info, is_slippery from env_info (default False).
    # KNOWN ISSUE: passes max_turns as a kwarg, which __init__ ignores, so max_steps always stays 5 and the
    # "max_turns" value is lost.
    return cls(
        entry=env_info,
        max_turns=env_info.get("max_turns", MAX_STEPS),
        is_slippery=env_info.get("is_slippery", False),
    )
