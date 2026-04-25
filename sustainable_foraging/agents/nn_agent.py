import random

from sustainable_foraging.agents.agent import BaseAgent


class NNAgent(BaseAgent):
    name = "NN Agent"

    def step(self, obs):
        return random.choice(obs.actions)
