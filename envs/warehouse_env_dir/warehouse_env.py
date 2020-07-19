try:
    from envs.warehouse_env_dir.article import Article
    from envs.warehouse_env_dir.article_collection import ArticleCollection
    from envs.warehouse_env_dir.storage import Storage
    from envs.warehouse_env_dir.storage_space import StorageSpace
    from envs.warehouse_env_dir.actions import Actions
    from envs.warehouse_env_dir.arrivals import Arrivals
    from envs.warehouse_env_dir.arrival_space import ArrivalSpace
    from envs.warehouse_env_dir.requests import Requests
    from envs.warehouse_env_dir.orders import Orders
    from envs.warehouse_env_dir.rewards import Rewards
    from envs.warehouse_env_dir.logger import log
    from envs.warehouse_env_dir.env_config import EnvConfig
except ModuleNotFoundError:
    from article import Article
    from article_collection import ArticleCollection
    from storage import Storage
    from storage_space import StorageSpace
    from actions import Actions
    from arrivals import Arrivals
    from arrival_space import ArrivalSpace
    from requests import Requests
    from orders import Orders
    from rewards import Rewards
    from logger import log
    from env_config import EnvConfig
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import gym
import itertools
import random
import sys
import math


# TODO Add docstrings


class WarehouseEnv(gym.Env):
    def __init__(self, config):
        self.seed = config.seed
        self.turns = config.turns
        self.rewards = Rewards(self.turns)
        self.storage_spaces = config.storage_spaces
        self.max_requests = config.max_requests
        self.max_arrivals = config.max_arrivals
        self.steps_to_request = config.steps_to_request
        self.simple_state = config.simple_state
        self.turn = 0
        self.game_over = False
        if self.seed is None:
            self.seed = random.randint(0, sys.maxsize)
        random.seed(self.seed)
        print('Env initialized seed:', self.seed)
        self._make_new_instances()
        self._print_env_state()

    def step(self, action=None, article_id=None):
        if self.turn < self.turns:

            self._pre_step()
            self.actions.do_action(action, article_id)
            # self.actions.action_random()
            log.info('Step ', self.turn)

            self._post_step()

            reward = self.rewards.calculate_step_reward()

            # resulting_state = []
            resulting_state = self.get_state()
            self.turn += 1

            # state, reward, gameover, debug info
        else:
            log.info('Finished!')
            # resulting_state = []
            resulting_state = self.get_state()
            reward = 0
            self.game_over = True
            # state, reward, gameover, debug info

        self.rewards.add_state_info(state=self.get_state(
        ), possible_articles=self.possible_articles.get_possible_articles())
        return resulting_state, reward, self.game_over, None

    def _post_step(self):
        pass

    def _pre_step(self):
        # get storage cost with the last state
        # self.action_reward += self.env.storage.get_storage_reward()
        self.rewards.add_reward_loop_storage(
            self.storage.get_storage_reward())

        # update requests (delete undeliverd)
        # self.action_reward += self.env.requests.update_requests()
        self.rewards.add_reward_loop_request_updates(
            self.requests.update_requests())

        # generate requests
        # 1x turn order, 1xturn store, 2xturn arrival
        if(self.turn % self.steps_to_request == 0):
            self.requests.generate_request(self.possible_articles)

        # handle arrivals
        # self.action_reward += self.handle_arrivals(self.orders.update_orders())
        self.rewards.add_reward_loop_arrival(
            self.arrivals.handle_arrivals(self.orders.update_orders()))

    def reset(self):
        # print(self.get_state())
        self.game_over = False
        self.turn = 0
        self._make_new_instances()
        self.rewards.reset_episode()
        log.info('env reset')
        # TODO extended state
        return self.get_state()

    def render(self):
        # TODO render
        pass

    def _print_env_info(self):
        print(self.storage)
        print(self.possible_articles.get_possible_articles())
        print(self.possible_articles)
        print(self.requests)

    def _print_env_state(self):
        print(self.storage.get_storage_state())
        print(self.requests.get_requests_state())
        print(self.arrivals.get_arrivals_state())

    def get_state(self):
        if self.simple_state:
            return tuple(self._get_simple_state())
        return tuple(self._get_extended_state())

    def _get_extended_state(self):
        # return [self.storage.get_storage_state(), self.requests.get_requests_state(), self.arrivals.get_arrivals_state()]
        return self.storage.get_simple_storage_state() + self.requests.get_simple_requests_state() + self.arrivals.get_simple_arrivals_state()+self.orders.get_simple_orders_state()
        # [0, 1, 3][1, 0][2, 0][1, 0](art+1) ^ 9

    def _get_simple_state(self):
        return self.storage.get_simple_storage_state()+self.requests.get_simple_requests_state() + self.arrivals.get_simple_arrivals_state()
        # [0, 1, 3][1, 0][2, 0](art+1) ^ 7

    def get_possible_states(self):
        if self.simple_state:
            return self._get_simple_possible_states()
        return self._get_extendet_possible_states()

    def _get_extendet_possible_states(self):
        state_elements = self.max_requests + self.max_arrivals + \
            self.storage_spaces + self.orders.max_orders
        possible_states = tuple(
            range(len(self.possible_articles.get_possible_articles())+1))
        arrays = []
        for i in range(state_elements):
            arrays.append(possible_states)
        cp = list(product(*arrays))
        print('There are', len(cp), 'possible states')
        return cp

    def _get_simple_possible_states(self):
        state_elements = self.max_requests + self.max_arrivals + self.storage_spaces
        possible_states = tuple(
            range(len(self.possible_articles.get_possible_articles())+1))
        arrays = []
        for i in range(state_elements):
            arrays.append(possible_states)
        cp = list(product(*arrays))
        print('There are', len(cp), 'possible states')
        return cp

    def _make_new_instances(self):
        self.arrivals = Arrivals(self.max_arrivals)
        self.possible_articles = ArticleCollection()
        self.requests = Requests(self, self.max_requests)
        self.storage = Storage(self.storage_spaces)
        self.actions = Actions(self)
        self.orders = Orders()


def test_prob():
    arr = []
    config = EnvConfig(None, 2, 2, 3, 1000)
    env = WarehouseEnv(config)
    for i in range(100):
        arr.append(env.possible_articles.get_random_article_with_prob().id)
    print(arr.count(1))
    print(arr.count(2))
    print(arr.count(3))


# TODO add config seeds to reproduce (with params steps and episodes)
# TODO improve qleraning ---> oracles? & random add oracles
# TODO or try heuristic without oracles
