import logging
import random
from typing import Callable, Union

import numpy as np

logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class ArmedBanditsProblem:
    """Simulated scenario, generates variable number of arms, each with
    their own normal distribution around a mean. Pulling an arm generates
    a reward relative to the normal distribution. Goal for an agent is to
    pull the arms some number of times and generate the as much reward as
    possible.
    """

    def __init__(
        self,
        total_turns: int,
        total_arms: int = 10,
        min_reward: Union[int, float] = 0,
        max_reward: Union[int, float] = 10,
        std: Union[int, float] = 1,
    ) -> None:
        self.total_turns = total_turns
        self.total_arms = total_arms
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.std = std

        self.arms: dict[int, Callable] = dict()

    def initialize(self) -> None:
        self._create_arms(self.total_arms, self.min_reward, self.max_reward, self.std)

    def _create_arms(
        self,
        total_arms: int,
        min_reward: Union[int, float],
        max_reward: Union[int, float],
        std: Union[int, float],
    ) -> None:
        for i in range(total_arms):
            mean = random.uniform(min_reward, max_reward)

            def bandit_arm() -> float:
                return np.random.normal(mean, std)

            self.arms[i] = bandit_arm
