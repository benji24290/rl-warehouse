import gym
import random
import sys


class WarehouseEnv(gym.Env):
    def __init__(self, seed=None):
        self.seed = seed
        if seed is None:
            self.seed = random.randint(0, sys.maxsize)
        print('Env initialized seed:', self.seed)

    def step(self):
        print('Step success!')

    def reset(self):
        print('env reset')
