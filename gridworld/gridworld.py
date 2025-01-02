import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

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

    def __init__(self, path_to_world: Union[str, Path]):
        # Load world schema, determine height and width
        with open(path_to_world, "r") as fp:
            self.world_schema = eval("".join(fp.readlines()))

        self.world_height = len(self.world_schema)
        self.world_width = len(self.world_schema[0])

        # Determine start and end position
        for i in range(len(self.world_schema)):
            for j in range(len(self.world_schema[0])):
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
            return (0, State(self.position))

        # Calculate new position
        new_position = None
        if action == Action.UP:
            new_position = (self.position[0] - 1, self.position[1])
        elif action == Action.DOWN:
            new_position = (self.position[0] + 1, self.position[1])
        elif action == Action.LEFT:
            new_position = (self.position[0], self.position[1] - 1)
        elif action == Action.RIGHT:
            new_position = (self.position[0], self.position[1] + 1)
        else:
            LOG.error(f"Invalid move: {action}")
            raise InvalidMove()

        # If move would put position outside grid, return -1 reward, otherwise update position
        # and return reward.
        if not self._is_valid_position(new_position):
            return (-1, State(self.position))

        self.position = new_position
        if self.position == self.end:
            return (10, State(self.position))

        return (0, State(self.position))

    def get_position(self) -> tuple[int, int]:
        return self.position

    def reset(self) -> None:
        self.position = self.start


class GridWorldAgent:

    def __init__(
        self,
        world: GridWorldSim,
        discount: float = 0.9,
        p_explore: float = 0.2,
        steps: int = 200,
        episodes: int = 100,
    ):
        self.world_sim = world
        self.discount = discount
        self.p_explore = p_explore
        self.steps = steps
        self.episodes = episodes

        # State function and state <> action function
        self.v_value_table: dict[State, float] = defaultdict()
        self.q_value_table: dict[tuple[State, Action], float] = defaultdict()

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

        # Add explore randomness to go left or right of the desired movement, execute movement
        if best_move in (Action.UP, Action.DOWN):
            alternate_move = random.choice([Action.LEFT, Action.RIGHT])
        else:
            alternate_move = random.choice([Action.UP, Action.DOWN])

        next_move = best_move
        if random.random() < self.p_explore:
            next_move = alternate_move

        reward, new_state = self.world_sim.move(next_move)

        # Update value functions:
        # q = E[R_t+1 + discount * v_star(S_t+1) | S_t = s, A = a]
        # v = maxarg q_star(state action pairs)
        if self.v_value_table.get(new_state) is not None:
            self.q_value_table[(cur_pos, next_move)] = (
                reward + self.discount * self.v_value_table[new_state]
            )
        else:
            self.q_value_table[(cur_pos, next_move)] = reward

        seen_state_action = []
        for move in possible_moves:
            if self.q_value_table.get((cur_pos, move)) is not None:
                seen_state_action.append(self.q_value_table[(cur_pos, move)])

        self.v_value_table[cur_pos] = max(seen_state_action)

    def run_simulation(self) -> None:
        """Run the number of episodes as defined configured."""
        for i in range(self.episodes):
            LOG.info(f"Episode {i + 1} in progress...")
            self.world_sim.reset()
            for _ in range(self.steps):
                self.next_move()

            LOG.info(f"Episode {i + 1} completed!")

        LOG.info(f"Simulation complete. Ran {self.episodes} episodes.")

    def __str__(self):
        """Print gridworld state table"""
        height = 0
        width = 0
        for state in self.v_value_table.keys():
            height = max(height, state.position[0] + 1)
            width = max(width, state.position[1] + 1)

        print(height, width)
        state_table_array = [[0] * width for _ in range(height)]
        print(state_table_array)
        for state, value in self.v_value_table.items():
            state_table_array[state.position[0]][state.position[1]] = "{:2.2f}".format(
                value
            )

        str_representation = [str(row) for row in state_table_array]
        return_str = "Gridworld State Table:\n{}"
        return return_str.format("\n".join(str_representation))


if __name__ == "__main__":
    path_to_world = Path("./gridworld/world/world_1.txt")
    world_sim = GridWorldSim(path_to_world)
    gridworld_agent = GridWorldAgent(world_sim)
    gridworld_agent.run_simulation()
    print(gridworld_agent)
