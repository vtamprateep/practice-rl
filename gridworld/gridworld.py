import logging
import pathlib
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class InvalidMove(Exception):
    pass


class Action(Enum):
    UP = "UP"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    DOWN = "DOWN"


@dataclass(eq=True, frozen=True)
class State:
    position: tuple[int, int]


class GridWorldSim:
    """Representation of gridworld, loads from a text file where:
    'o': Empty space
    's': Starting position
    'e': Goal / end position

    Assumes grid world is a valid rectangle of sorts."""

    def __init__(self, path_to_world: str):
        self.world_schema = None
        self.world_height = None
        self.world_width = None

        self.start = None
        self.end = None

        # Load world schema, determine height and width
        with open(path_to_world, "r") as fp:
            self.world_schema = eval("".join(fp.readlines()))

        self.world_height = len(self.world_schema)
        self.world_width = len(self.world_schema[0])

        # Determine start and end position
        for i in range(len(self.world_schema)):
            for j in range(len(self.world_schema)):
                if self.world_schema[i][j] == "s":
                    self.start = (i, j)
                if self.world_schema[i][j] == "e":
                    self.end = (i, j)

        self.position = self.start

    def __str__(self):
        world_str = [str(row) for row in self.world_schema]
        return_str = "World schema:\n{}"
        return return_str.format("\n".join(world_str))

    def _is_valid_position(self, position: tuple[int, int]) -> bool:
        """Checks that position is in the grid and return true if so, false otherwise."""
        if not 0 <= position[0] < self.world_height:
            return False
        elif not 0 <= position[1] < self.world_width:
            return False

        return True

    def move(self, action: Action, plan: bool = False) -> tuple[float, State]:
        """Calls plan_move to get reward then updates position. If plan = True, doesn't update
        position."""
        # If in goal state, just return reward of goal state and don't update position
        if self.position == self.end:
            return (10, State(self.position))

        # Calculate new position
        new_position = None
        if action == Action.UP:
            new_position = (self.position[0] - 1, self.position[1])
        elif action == Action.DOWN:
            new_position = (self.position[0] + 1, self.position[1])
        elif action == Action.LEFT:
            new_position = (self.position[0] + 1, self.position[1])
        elif action == Action.RIGHT:
            new_position = (self.position[0] + 1, self.position[1])
        else:
            raise InvalidMove()

        # If move would put position outside grid, return -1 reward, otherwise update position
        # and return reward.
        if not self._is_valid_position(new_position):
            return (-1, State(self.position))

        self.position = new_position
        if self.position == self.end:
            return (10, State(self.position))

    def get_position(self) -> tuple[int, int]:
        return self.position


class GridWorldAgent:

    def __init__(
        self, world: GridWorldSim, discount: float = 0.1, p_explore: float = 0.2
    ):
        self.world_sim = world
        self.discount = discount
        self.p_explore = p_explore

        # State function and state <> action function
        self.v_value_table: dict[State, float] = defaultdict(int)
        self.q_value_table: dict[tuple[State, Action], float] = defaultdict(int)

    def next_move(self) -> None:
        """Make next best move based on policy, log action, state, and reward."""
        cur_pos = State(self.world_sim.get_position())
        possible_moves = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        # Loop over next possible moves
        best_move = random.choice(possible_moves)  # Pick random move to start
        best_est_reward = None
        for move in possible_moves:
            state_action_pair = (cur_pos, move)
            if state_action_pair in self.q_value_table:
                # Update best move if None or if current action is better than previous
                state_action_value = self.q_value_table[state_action_pair]
                if best_est_reward is None or state_action_value > best_est_reward:
                    best_est_reward = state_action_value
                    best_move = move

        # Add explore randomness to go left or right of the desired movement
        alternate_moves = None
        if alternate_moves in (Action.UP, Action.DOWN):
            alternate_moves = random.choice([Action.LEFT, Action.RIGHT])
        elif alternate_moves in (Action.LEFT, Action.RIGHT):
            alternate_moves = random.choice([Action.UP, Action.DOWN])

        if random.random() < self.p_explore:
            reward, new_state = self.world_sim.move(random.choose(alternate_moves))
        else:
            reward, new_state = self.world_sim.move(best_move)

        # Update value functions: q = E[R_t+1 + discount * v_star_(S_t+1) | S_t = s, A = a]
        self.v_value_table[new_state] = reward


if __name__ == "__main__":
    path_to_world = pathlib.Path("./gridworld/world/world_1.txt")
    world_sim = GridWorldSim(path_to_world)
    print(world_sim)
    print(world_sim.get_possible_moves())
