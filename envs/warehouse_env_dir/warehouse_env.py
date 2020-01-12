import gym
import random
import sys


class Article():
    # frequency is the probability that an article of this type is requested
    # float 0-1
    # availability of this article, between 0-1 smaller availability takes longer to get delivered to the warehouse
    # float 0-1
    # name of this article
    # string
    def __init__(self, frequency=0.5, availability=0.5, name="article name"):
        self.frequency = frequency
        self.availability = availability
        self.name = name


class Request():
    def __init__(self):
        # the reward value of this request
        # float 0-1
        self.reward_value = 1
        # the max steps to deliver else the request will be cancelled and added a negative reward
        # int
        self.time = 5
        # the article type gets random article
        # Article
        self.article = None


class WarehouseEnv(gym.Env):
    def __init__(self, seed=None, max_requests=3, max_arrivals=1):
        self.seed = seed
        self.max_requests = max_requests
        self.max_arrivals = max_arrivals
        if seed is None:
            self.seed = random.randint(0, sys.maxsize)
        print('Env initialized seed:', self.seed)

    def step(self):
        self._new_request()
        print('Step success!')

    def reset(self):
        print('env reset')

    def _new_request(self):
        pass
