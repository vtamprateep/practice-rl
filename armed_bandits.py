import logging
import random
from typing import List, Union

import numpy as np

logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Arm:
    """Individual arms of the bandit."""

    def __init__(self, mean: Union[int, float], std: Union[int, float]) -> None:
        self.mean = mean
        self.std = std

    def get_mean(self) -> float:
        return self.mean

    def get_std(self) -> float:
        return self.std

    def pull(self) -> float:
        return np.random.normal(self.mean, self.std)


class ArmedBandits:
    """Simulated scenario, generates variable number of arms, each with
    their own normal distribution around a mean. Pulling an arm generates
    a reward relative to the normal distribution. Goal for an agent is to
    pull the arms some number of times and generate the as much reward as
    possible.
    """

    def __init__(
        self,
        total_arms: int = 10,
        min_reward: Union[int, float] = 0,
        max_reward: Union[int, float] = 10,
        std: Union[int, float] = 1,
    ) -> None:
        self.total_arms = total_arms
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.std = std

        self.arms: dict[int, Arm] = dict()

    def initialize(self) -> None:
        """Initialize arms of the bandit."""
        self._create_arms(self.total_arms, self.min_reward, self.max_reward, self.std)

    def pull_arm(self, arm_index: int) -> float:
        """Simulate a pull of an arm given an 0-indexed arm number."""
        if self.arms == dict():
            LOG.error("Bandit arms not initialized, returning...")
            raise Exception()

        return self.arms[arm_index].pull()

    def get_arm(self, arm_index: int = -1) -> Union[List[tuple[int, Arm]], Arm]:
        """Return arm given arm_index. If not specified, return all arms
        as list of tuples where first element is the arm_index and second element
        is the arm object.
        """
        if arm_index >= 0:
            return self.arms[i]

        return [(k, v) for k, v in self.arms.items()]

    def _create_arms(
        self,
        total_arms: int,
        min_reward: Union[int, float],
        max_reward: Union[int, float],
        std: Union[int, float],
    ) -> None:
        """Sets populates an entry in self.arms for the number of arms desired. Each entry
        is a function that returns a value of a pull of that arm given some mean and standard
        deviation value.
        """
        for i in range(total_arms):
            mean = random.uniform(min_reward, max_reward)
            LOG.info(f"Creating arm with mean {mean} and std {std}.")
            self.arms[i] = Arm(mean, std)


if __name__ == "__main__":
    bandit = ArmedBandits()
    bandit.initialize()
    for i in range(10):
        print(bandit.pull_arm(i))
