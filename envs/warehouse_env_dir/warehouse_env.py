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
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import gym
import itertools
import random
import sys


# This reward values are the constant part and will be modified with distance
ORDER_POSITIVE_REWARD = 10
ORDER_NEGATIVE_REWARD = -50
STORE_POSITIVE_REWARD = 10
STORE_NEGATIVE_REWARD = -10
DELIVER_POSITIVE_REWARD = 100
DELIVER_NEGATIVE_REWARD = -100
REQUEST_EXPIRED_REWARD = -50
INVENTORY_COST_FACTOR = 1

# TODO Add docstrings


class WarehouseEnv(gym.Env):
    def __init__(self, seed=None, max_requests=2, max_arrivals=1, storage_spaces=3, max_turns=50, simple_state=True):
        self.seed = seed
        self.rewards = Rewards(max_turns)
        self.storage_spaces = storage_spaces
        self.max_requests = max_requests
        self.max_arrivals = max_arrivals
        self.max_turns = max_turns
        self.simple_state = simple_state
        self.turn = 0
        self.game_over = False
        if seed is None:
            self.seed = random.randint(0, sys.maxsize)
        random.seed(self.seed)
        print('Env initialized seed:', self.seed)
        self._make_new_instances()
        self._add_test_values()
        self._print_env_state()

    def step(self, action=None, article_id=None):
        if self.turn < self.max_turns:

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
        # TODO add counter in request to only gen %2 turn or with prob.
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
        return self.storage.get_storage_state() + self.requests.get_requests_state() + self.arrivals.get_arrivals_state()

    def _get_simple_state(self):
        # return [self.storage.get_storage_state(), self.requests.get_requests_state(), self.arrivals.get_arrivals_state()]
        return self.storage.get_simple_storage_state()+self.requests.get_simple_requests_state() + self.arrivals.get_simple_arrivals_state()
        #[0, 1, 3][1, 0][2, 0](art+1) ^ 7

    def get_possible_states(self):
        if self.simple_state:
            return self._get_simple_possible_states()
        return self._get_extendet_possible_states()

    def _get_extendet_possible_states(self):
        # TODO generate possible states
        pass
        # self.storage.get_possible_storage_states() + self.requests.get_possible_requests_states() + \
        #   self.arrivals.get_possible_arrivals_states()

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

    def _add_test_values(self):
        # Storage article
        self.storage.store(Article(3, 0.5,  "Normal"), 2)
        # New request
        self.requests.generate_request(self.possible_articles)
        self.requests.generate_request(self.possible_articles)
        # New element in arrival
        self.arrivals.add_article_to_arrival(Article(2, 0.5,  "Normal"))


# Random
def random_actions(count=100, steps=1000):
    # env = WarehouseEnv(None, 2, 2, 3, 50)
    env = WarehouseEnv(None, 2, 2, 3, steps)

    num_ep = count

    for i in range(num_ep):
        # Print all n episodes
        if i % 1000 == 0:
            print('starting episode', i)
        done = False
        # observation = env.reset()
        env.reset()
        while not done:
            # TODO seed?
            # rand = np.random.random()
            # action = max_action(Q, observation, env.possible_actions)
            [state, reward, game_over, debug] = env.step()
            if env.game_over:
                break
        if i % 1000 == 0:
            print('Episode ', i, ' reward is:',
                  env.rewards.get_total_episode_reward())

    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)

    # env.rewards.plot_loot_storage(1)
    # env.rewards.plot_loot_request_updates(1)
    # env.rewards.plot_loot_arrival(1)
    # env.rewards.plot_action_deliver(1)
    # env.rewards.plot_action_order(1)
    # env.rewards.plot_action_store(1)
    return env.rewards.all_episode_rewards_per_step


# Q Function
def q_function(count=50000, steps=1000, alpha=0.1, gamma=1.0, eps=1.0):

    def max_action(Q, state, actions):
        values = np.array([Q[state, a] for a in actions])
        action = np.argmax(values)
        # print('best action is', action, actions[action])
        return actions[action]

    env = WarehouseEnv(None, 2, 2, 3, steps)

    ALPHA = alpha  # learningrate
    GAMMA = gamma
    EPS = eps  # eps greedy action selection

    # TODO init q here
    Q = {}
    for state in env.get_possible_states():
        # TODO add actions with articles
        for action in env.actions.actions:
            Q[state, action] = 0

    num_ep = count
    total_rewards = np.zeros(num_ep)

    for i in range(num_ep):
        # Print all n episodes
        if i % 1000 == 0:
            print('starting episode', i)
        done = False
        observation = env.reset()
        while not done:
            # TODO seed?
            rand = random.random()
            action = max_action(Q, observation, env.actions.actions) if rand < (1-EPS) \
                else env.actions.get_random_action()
            observation_, reward, game_over, debug = env.step(action)

            action_ = max_action(Q, observation_, env.actions.actions)
            # Update Q table
            Q[observation, action] = Q[observation, action] + ALPHA*(reward +
                                                                     GAMMA*Q[observation_, action_] - Q[observation, action])
            observation = observation_

            # linear decrease of epsilon
            if EPS - 2 / num_ep > 0:
                EPS -= 2 / num_ep
            else:
                EPS = 0

            if env.game_over:
                break
        if i % 1000 == 0:
            print('Episode ', i, ' reward is:',
                  env.rewards.get_total_episode_reward())

    # print(Q)
    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)
    # env.rewards.plot_episode_rewards(num_ep-1)
    return env.rewards.all_episode_rewards_per_step


# DQN
# TODO add dqn


# Heuristic


def heuristic(count=100, check=True, steps=1000):
    # env = WarehouseEnv(None, 2, 2, 3, 50)
    env = WarehouseEnv(None, 2, 2, 3, steps)
    check_for_orders = check
    num_ep = count

    for i in range(num_ep):
        # Print all n episodes
        if i % 1000 == 0:
            print('starting episode', i)
        done = False
        # observation = env.reset()
        env.reset()
        while not done:
            action = None
            a_id = None
            # prio 1: fullfill requests if article is in store
            for req in env.requests.requests:
                for space in env.storage.storage_spaces:
                    if req.article == space.article and action == None:
                        #print('deliver:', req.article.name, space.article.name)
                        action = 'DELIVER'
                        a_id = req.article.id
            # prio 2: store articles from arrival
            if(action == None):
                position = env.actions.store_oracle()
                if position:
                    for arr_space in env.arrivals.arrivals:
                        if arr_space.article:
                            action = 'STORE'
                            a_id = arr_space.article.id

            # prio 3: order article from wich the least are stored
            if(action == None):
                # TODO least stored
                for possible in env.possible_articles.articles:
                    if possible.id not in env.storage.get_simple_storage_state():
                        if(check_for_orders):
                            if len(env.orders.orders) < 1:
                                action = 'ORDER'
                                a_id = possible.id
                            else:
                                for order in env.orders.orders:
                                    if(order.article.id is not possible.id):
                                        # print(possible.id, 'is not in:',
                                        #      env.storage.get_simple_storage_state())
                                        action = 'ORDER'
                                        a_id = possible.id
                                    else:
                                        # print(possible.id, 'was already ordered',
                                        #      order.article.id)
                                        pass
                        else:
                            action = 'ORDER'
                            a_id = possible.id

            # TODO maybe delete idle?
            # prio 4: do nothing
            if(action == None):
                action = 'IDLE'
            [state, reward, game_over, debug] = env.step(action, a_id)
            if env.game_over:
                break
        if i % 1000 == 0:
            print('Episode ', i, ' reward is:',
                  env.rewards.get_total_episode_reward())

    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)

    return env.rewards.all_episode_rewards_per_step


def test_prob():
    arr = []
    env = WarehouseEnv(None, 2, 2, 3, 1000)
    for i in range(100):
        arr.append(env.possible_articles.get_random_article_with_prob().id)
    print(arr.count(1))
    print(arr.count(2))
    print(arr.count(3))


if __name__ == '__main__':
    def plot():
        # Test 1 heur vs q vs rand
        #plt.plot(random_actions(5000), label='random')
        #plt.plot(q_function(5000, 1000, 0.1), label='Q5k-1k-g1')
        #plt.plot(q_function(5000, 1000, 0.1, 0.5), label='Q5k-1k-g0.5')
        #plt.plot(heuristic(5000), label='heur-true')
        #plt.plot(heuristic(5000, False), label='heur-false')

        # Test 2 Q different step count / different episodes
        #plt.plot(q_function(5000, 1000), label='Q5k-1k')
        #plt.plot(q_function(10000), label='Q10k-1k')
        #plt.plot(q_function(5000, 2000), label='Q5k-2k')

        # Test 3 Alphas
        #plt.plot(q_function(5000, 1000, 0.1), label='Q5k-1k-a0.1')
        #plt.plot(q_function(5000, 1000, 0.25), label='Q5k-1k-a0.25')
        #plt.plot(q_function(5000, 1000, 0.5), label='Q5k-1k-a0.5')

        # Test 4 Gammas
        #plt.plot(q_function(5000, 1000, 0.1, 1), label='Q5k-1k-g1')
        #plt.plot(q_function(5000, 1000, 0.1, 0.75), label='Q5k-1k-g0.75')
        #plt.plot(q_function(5000, 1000, 0.1, 0.5), label='Q5k-1k-g0.5')

        # Test 5 Epsilon
        #plt.plot(q_function(5000, 1000, 0.1, 1, 1), label='Q5k-1k-e1')
        #plt.plot(q_function(5000, 1000, 0.1, 1, 0.75), label='Q5k-1k-e0.75')
        #plt.plot(q_function(5000, 1000, 0.1, 1, 0.5), label='Q5k-1k-e0.5')

        # Test 6 Gamma with more steps/episodes

        # Test 7 Epsilon with more steps/episodes

        # plt.legend()
        # plt.show()

        test_prob()

    plot()
    # random_actions()
    # q_function(50000)
    # heuristic()


# TODO add config seeds to reproduce (with params steps and episodes)
# TODO improve qleraning ---> oracles? & random add oracles
# TODO or try heuristic without oracles
