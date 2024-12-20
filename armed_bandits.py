import logging
import random
from typing import Iterable, Union

import numpy as np

logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class ArmedBandit:
    """Simulated scenario, generates variable number of arms, each with
    their own normal distribution around a mean. Pulling an arm generates
    a reward relative to the normal distribution. Goal for an agent is to
    pull the arms some number of times and generate the as much reward as
    possible.
    """

    class Arm:
        """Individual arms of the bandit."""

        def __init__(
            self,
            mean: Union[int, float],
            std: Union[int, float],
            rand_seed: Union[int, None],
        ) -> None:
            self.mean = mean
            self.std = std

        def get_mean(self) -> float:
            return self.mean

        def get_std(self) -> float:
            return self.std

        def pull(self) -> float:
            return np.random.normal(self.mean, self.std)

    def __init__(
        self,
        total_arms: int = 10,
        min_reward: Union[int, float] = 0,
        max_reward: Union[int, float] = 10,
        std: Union[int, float] = 1,
        rand_seed: Union[int, None] = None,
    ) -> None:
        self.total_arms = total_arms
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.std = std
        self.rand_seed = rand_seed

        self.arms: dict[int, ArmedBandit.Arm] = dict()
        self._create_arms(
            self.total_arms, self.min_reward, self.max_reward, self.std, self.rand_seed
        )

    def pull_arm(self, arm_index: int) -> float:
        """Simulate a pull of an arm given an 0-indexed arm number."""
        if self.arms == dict():
            LOG.error("Bandit arms not initialized, returning...")
            raise Exception()

        return self.arms[arm_index].pull()

    def get_all_arms(self) -> list[int]:
        """Returns set of all arms a player can pull."""
        return list(self.arms.keys())

    def _create_arms(
        self,
        total_arms: int,
        min_reward: Union[int, float],
        max_reward: Union[int, float],
        std: Union[int, float],
        rand_seed: Union[int, None],
    ) -> None:
        """Populates an entry in self.arms for the number of arms desired. Each entry
        is a function that returns a value of a pull of that arm given some mean and standard
        deviation value.
        """
        for i in range(total_arms):
            mean = random.uniform(min_reward, max_reward)
            LOG.info(f"Creating arm with mean {mean} and std {std}.")
            self.arms[i] = self.Arm(mean, std, rand_seed)


class ActionRewardFunctionTable:
    """Given a reward, returns expected value. Method here is just averaging the rewards seen
    for a state.
    """

    class ExpectedActionReward:
        """Tracks the reward and the updating of the reward given an action."""

        def __init__(self) -> None:
            self.sample_population = 0
            self.expected_reward = 0.0

        def get_reward(self) -> float:
            """Returns current expected reward."""
            return self.expected_reward

        def update_reward(self, observed_reward: float) -> None:
            """Update expected reward based on observed reward."""
            numerator = self.expected_reward * self.sample_population + observed_reward
            denominator = self.sample_population + 1
            self.expected_reward = numerator / denominator
            self.sample_population += 1

    def __init__(self, action_set: Iterable[int]) -> None:
        self.action_reward_table = {
            action: self.ExpectedActionReward() for action in action_set
        }

    def predict_reward(self, action: int) -> float:
        if action not in self.action_reward_table:
            raise KeyError("Key not found in action reward table.")

        return self.action_reward_table[action].get_reward()

    def update_table(self, action: int, observed_reward: float) -> None:
        self.action_reward_table[action].update_reward(observed_reward)


class ArmedBanditAgent:
    """Agent to play the armed bandit game with the goal of maximizing rewards."""

    def __init__(self, total_turns: int, e_explore: float = 0.1):
        self.total_turns = total_turns
        self.e_explore = e_explore
        self.is_setup = False

    def setup_simulation(self, bandit: ArmedBandit) -> None:
        """Configure bandit game and reward table"""
        self.armed_bandit_sim = bandit

        action_states = [
            arm_index for arm_index in self.armed_bandit_sim.get_all_arms()
        ]
        self.action_reward_table = ActionRewardFunctionTable(action_states)
        self.is_setup = True

    def _next_action(self) -> int:
        possible_actions = list(self.armed_bandit_sim.get_all_arms())

        # Determine exploit or explore
        explore_p = random.random()
        if explore_p < self.e_explore:
            return random.choice(possible_actions)

        # Loop over possible actions and choose best
        best_action, best_reward = 0, None
        for action in possible_actions:
            if (
                best_reward is None
                or self.action_reward_table.predict_reward(action) > best_reward
            ):
                best_action = action
                best_reward = self.action_reward_table.predict_reward(action)

        return best_action

    def play(self) -> float:
        """Set agent to play armed bandit. Returns"""
        if not self.is_setup:
            LOG.error("Agent not set-up, please call setup_simulation() first.")
            raise Exception()

        LOG.info(f"Playing armed bandit with {self.total_turns} turns...")

        total_reward = 0.0
        for _ in range(self.total_turns):
            # Use action reward function to figure out best move
            next_move = self._next_action()
            new_reward = self.armed_bandit_sim.pull_arm(next_move)

            # Update reward table and running tally of score
            total_reward += new_reward
            self.action_reward_table.update_table(next_move, new_reward)

        return total_reward


if __name__ == "__main__":
    sim_results = []
    for i in range(11):
        agent = ArmedBanditAgent(1000, i / 10)
        agent.setup_simulation(ArmedBandit(rand_seed=0))
        sim_results.append((i / 10, agent.play()))

    print("Sim results for the following explore probabilities over 1000 :\n")
    for p, r in sim_results:
        print(f"For epsilon {p}, sim earned {r} score.")
