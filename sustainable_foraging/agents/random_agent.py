import random

from sustainable_foraging.agents import BaseAgent


class RandomAgent(BaseAgent):
    name = "Random Agent"

    def step(self, obs):
        return random.choice(obs.actions)
