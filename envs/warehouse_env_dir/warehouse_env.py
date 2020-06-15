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
import math


# TODO Add docstrings


class WarehouseEnv(gym.Env):
    def __init__(self, seed=None, max_requests=2, max_arrivals=1, storage_spaces=3, max_turns=50, steps_to_request=3, simple_state=True):
        self.seed = seed
        self.rewards = Rewards(max_turns)
        self.storage_spaces = storage_spaces
        self.max_requests = max_requests
        self.max_arrivals = max_arrivals
        self.max_turns = max_turns
        self.steps_to_request = steps_to_request
        self.simple_state = simple_state
        self.turn = 0
        self.game_over = False
        if seed is None:
            self.seed = random.randint(0, sys.maxsize)
        random.seed(self.seed)
        print('Env initialized seed:', self.seed)
        self._make_new_instances()
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


# Random
def random_actions(count=100, steps=1000, seed=None):
    # env = WarehouseEnv(None, 2, 2, 3, 50)
    env = WarehouseEnv(seed, 2, 2, 3, steps)

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
    return env.rewards


# Q Function
def q_function(count=50000, steps=1000, alpha=0.1, gamma=1.0, eps=1.0, seed=None):

    def max_action(Q, state, actions):
        values = np.array([Q[state, a] for a in actions])
        action = np.argmax(values)
        # print('best action is', action, actions[action])
        return actions[action]

    env = WarehouseEnv(seed, 2, 2, 3, steps)

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
    env.rewards.set_q(Q)
    # print_q_matrix(Q)

    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)
    # env.rewards.plot_episode_rewards(num_ep-1)
    return env.rewards


def q_function_extended_order(count=50000, steps=1000, alpha=0.1, gamma=1.0, eps=1.0, seed=None, steps_to_request=3, simple_state=True):

    def max_action(Q, state, actions):
        values = np.array([Q[state, a] for a in actions])
        action = np.argmax(values)
        # print('best action is', action, actions[action])
        return actions[action]

    env = WarehouseEnv(seed, 2, 2, 3, steps,
                       steps_to_request=steps_to_request, simple_state=simple_state)

    ALPHA = alpha  # learningrate
    GAMMA = gamma
    EPS = eps  # eps greedy action selection

    # TODO init q here
    Q = {}
    for state in env.get_possible_states():
        # TODO add actions with articles
        for action in env.actions.actions_extended:
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
            action = max_action(Q, observation, env.actions.actions_extended) if rand < (1-EPS) \
                else env.actions.get_random_action_extended()
            observation_, reward, game_over, debug = env.step(action)

            action_ = max_action(Q, observation_, env.actions.actions_extended)
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

    env.rewards.set_q(Q)
    # print(Q)
    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)
    # env.rewards.plot_episode_rewards(num_ep-1)
    return env.rewards


def q_function_with_idle(count=50000, steps=1000, alpha=0.1, gamma=1.0, eps=1.0, seed=None, steps_to_request=3):

    def max_action(Q, state, actions):
        values = np.array([Q[state, a] for a in actions])
        action = np.argmax(values)
        # print('best action is', action, actions[action])
        return actions[action]

    env = WarehouseEnv(seed, 2, 2, 3, steps,
                       steps_to_request=steps_to_request)

    ALPHA = alpha  # learningrate
    GAMMA = gamma
    EPS = eps  # eps greedy action selection

    # TODO init q here
    Q = {}
    for state in env.get_possible_states():
        # TODO add actions with articles
        for action in env.actions.actions_with_idle:
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
            action = max_action(Q, observation, env.actions.actions_with_idle) if rand < (1-EPS) \
                else env.actions.get_random_action_with_idle()
            observation_, reward, game_over, debug = env.step(action)

            action_ = max_action(
                Q, observation_, env.actions.actions_with_idle)
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

    env.rewards.set_q(Q)
    # print_q_matrix(Q)
    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)
    # env.rewards.plot_episode_rewards(num_ep-1)
    # return env.rewards.all_episode_rewards_per_step
    return env.rewards


# DQN
# TODO add dqn


# Heuristic


def heuristic(count=100, steps=1000, version='v1', seed=None, steps_to_request=3, simple_state=True):
    # TODO Remove simple state, useless
    # env = WarehouseEnv(None, 2, 2, 3, 50)
    if(version == 'v1'):
        check_for_orders = False
        can_idle = False
    elif(version == 'v2'):
        check_for_orders = True
        can_idle = False
    elif(version == 'v3'):
        check_for_orders = False
        can_idle = True
    elif(version == 'v4'):
        check_for_orders = True
        can_idle = True
    else:
        raise Exception(version, 'is not a valid version')

    env = WarehouseEnv(seed, 2, 2, 3, steps,
                       steps_to_request=steps_to_request, simple_state=simple_state)
    num_ep = count

    Q = {}
    for state in env.get_possible_states():
        # TODO add actions with articles
        for action in env.actions.actions:
            Q[state, action] = 0

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
                        # print('deliver:', req.article.name, space.article.name)
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
                    empty_spaces_storage_arrival = env.storage.get_simple_storage_state().count(
                        0)+env.arrivals.get_simple_arrivals_state().count(0)
                    if (possible.id not in env.storage.get_simple_storage_state() and empty_spaces_storage_arrival > 0):

                        if(check_for_orders):
                            if(empty_spaces_storage_arrival-len(env.orders.orders) > 0):
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
                                # print('empty=', empty_spaces_storage_arrival,
                                #     'open orders=', len(env.orders.orders))
                                pass
                        else:
                            action = 'ORDER'
                            a_id = possible.id
            # prio 4: do nothing
            if(can_idle):
                if(action == None):
                    action = 'IDLE'
            else:
                action = random.choice(env.actions.actions)
            Q[env.get_state(), action] = 100
            [state, reward, game_over, debug] = env.step(action, a_id)
            if env.game_over:
                break
        if i % 1000 == 0:
            print('Episode ', i, ' reward is:',
                  env.rewards.get_total_episode_reward())

    env.rewards.set_q(Q)
    # print_q_matrix(Q)
    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)
    # env.rewards.plot_episode_rewards(num_ep-1)
    return env.rewards


def test_prob():
    arr = []
    env = WarehouseEnv(None, 2, 2, 3, 1000)
    for i in range(100):
        arr.append(env.possible_articles.get_random_article_with_prob().id)
    print(arr.count(1))
    print(arr.count(2))
    print(arr.count(3))


if __name__ == '__main__':
    def tests():
        # Test 1 heur vs q vs rand
        # plt.plot(random_actions(5000).all_episode_rewards_per_step, label='random')
        # plt.plot(q_function(5000, 1000, 0.1).all_episode_rewards_per_step, label='Q5k-1k-g1')
        # plt.plot(q_function(5000, 1000, 0.1, 0.5).all_episode_rewards_per_step, label='Q5k-1k-g0.5')
        # plt.plot(heuristic(5000).all_episode_rewards_per_step, label='heur-true')
        # plt.plot(heuristic(5000, False).all_episode_rewards_per_step, label='heur-false')

        # Test 2 Q different step count / different episodes
        # plt.plot(q_function(5000, 1000).all_episode_rewards_per_step, label='Q5k-1k')
        # plt.plot(q_function(10000).all_episode_rewards_per_step, label='Q10k-1k')
        # plt.plot(q_function(5000, 2000).all_episode_rewards_per_step, label='Q5k-2k')

        # Test 3 Alphas
        # plt.plot(q_function(5000, 1000, 0.1).all_episode_rewards_per_step, label='Q5k-1k-a0.1')
        # plt.plot(q_function(5000, 1000, 0.25).all_episode_rewards_per_step, label='Q5k-1k-a0.25')
        # plt.plot(q_function(5000, 1000, 0.5).all_episode_rewards_per_step, label='Q5k-1k-a0.5')

        # Test 4 Gammas
        # plt.plot(q_function(5000, 1000, 0.1, 1).all_episode_rewards_per_step, label='Q5k-1k-g1')
        # plt.plot(q_function(5000, 1000, 0.1, 0.75).all_episode_rewards_per_step, label='Q5k-1k-g0.75')
        # plt.plot(q_function(5000, 1000, 0.1, 0.5).all_episode_rewards_per_step, label='Q5k-1k-g0.5')

        # Test 5 Epsilon
        # plt.plot(q_function(5000, 1000, 0.1, 1, 1).all_episode_rewards_per_step, label='Q5k-1k-e1')
        # plt.plot(q_function(5000, 1000, 0.1, 1, 0.75).all_episode_rewards_per_step, label='Q5k-1k-e0.75')
        # plt.plot(q_function(5000, 1000, 0.1, 1, 0.5).all_episode_rewards_per_step, label='Q5k-1k-e0.5')

        # Test 6 Gamma with more steps/episodes

        # Test 7 Epsilon with more steps/episodes

        # Test 8 normal q vs q with extended
        # plt.plot(q_function(15000, 2000).all_episode_rewards_per_step, label='Q5k-1k-g1')
        # plt.plot(q_function_extended_order(15000, 2000).all_episode_rewards_per_step,
        # label='Q5k-1k-g1-extended-order')

        # Test 9
        # q_function(5000)
        # q_function_extended_order(5000)
        # heuristic(5000, False)

        # plt.plot(smoothList(random_actions(5000).all_episode_rewards_per_step,
        # degree = 400), label='random')
        # plt.plot(smoothList(q_function(5000).all_episode_rewards_per_step, degree=400), label='Q5k-1k-g1')

        # Test all 10
        if(False):
            plt.plot(
                heuristic(5000, version='v1').all_episode_rewards_per_step, label='heur-v1')
            plt.plot(
                heuristic(5000, version='v2').all_episode_rewards_per_step, label='heur-v2')
            plt.plot(
                heuristic(5000, version='v3').all_episode_rewards_per_step, label='heur-v3')
            plt.plot(
                heuristic(5000, version='v4').all_episode_rewards_per_step, label='heur-v4')

            plt.plot(q_function(
                5000).all_episode_rewards_per_step, label='q')
            # plt.plot(smoothList(q_function_extended_order(5000),
            #                    degree=400), label='q-extended')
            plt.plot(q_function_with_idle(
                5000).all_episode_rewards_per_step, label='q-idle')

        # Test 11 only one ART!
        if(False):

            rew_q_w_i_a0_1 = q_function_with_idle(10000, 1000, 0.1,  seed=1234)
            rew_q_e = q_function_extended_order(10000, 1000, 0.1,  seed=1234)
            rew_h_v1 = heuristic(10000, version='v1', seed=1234)
            rew_h_v2 = heuristic(10000, version='v2', seed=1234)
            rew_h_v3 = heuristic(10000, version='v3', seed=1234)
            rew_h_v4 = heuristic(10000, version='v4', seed=1234)

            plt.plot(rew_q_w_i_a0_1.all_episode_rewards_per_step,
                     label='q-idle-a0.1')
            plt.plot(rew_q_e.all_episode_rewards_per_step,
                     label='q-ext')
            plt.plot(
                rew_h_v1.all_episode_rewards_per_step, label='h-v1')
            plt.plot(
                rew_h_v2.all_episode_rewards_per_step, label='h-v2')
            plt.plot(
                rew_h_v3.all_episode_rewards_per_step, label='h-v3')
            plt.plot(
                rew_h_v4.all_episode_rewards_per_step, label='h-v4')

            plt.legend()
            plt.show()

            rew_q_w_i_a0_1.plot_pos_neg_rewards(name='q-idle-a0.1')
            rew_q_e.plot_pos_neg_rewards(name='q-idle-a0.1')
            rew_q_w_i_a0_1.plot_episode_rewards(999)
            rew_q_e.plot_episode_rewards(999)
            rew_h_v1.plot_episode_rewards(999)
            rew_h_v2.plot_episode_rewards(999)
            rew_h_v3.plot_episode_rewards(999)
            rew_h_v4.plot_episode_rewards(999)

        # Test 11-2 only one ART!
        if(False):

            rew_q_w_i_a0_1 = q_function_with_idle(
                10000, 1000, 0.1,  seed=1234)
            rew_q_e = q_function_extended_order(10000, 1000, 0.1,  seed=1234)
            rew_h_v4 = heuristic(10000, version='v4', seed=1234)
            rew_h_v3 = heuristic(10000, version='v3', seed=1234)

            plt.plot(rew_q_w_i_a0_1.all_episode_rewards_per_step,
                     label='q-idle-a0.1')
            plt.plot(rew_q_e.all_episode_rewards_per_step,
                     label='q-ext')
            plt.plot(
                rew_h_v3.all_episode_rewards_per_step, label='h-v3')
            plt.plot(
                rew_h_v4.all_episode_rewards_per_step, label='h-v4')

            plt.legend()
            plt.show()

            rew_q_w_i_a0_1.plot_pos_neg_rewards(name='q-idle-a0.1')
            rew_q_e.plot_pos_neg_rewards(name='q-idle-a0.1')
            rew_q_w_i_a0_1.plot_episode_rewards(999)
            rew_q_e.plot_episode_rewards(999)
            rew_h_v3.plot_episode_rewards(999)
            rew_h_v4.plot_episode_rewards(999)

        # Test Summary 1 One-Art---3Steps to request
        if(False):
            rew_q_e = q_function_extended_order(
                10000, 100, 0.1,  seed=1234, steps_to_request=3)
            rew_q_e_order = q_function_extended_order(
                10000, 100, 0.1,  seed=1234, simple_state=False, steps_to_request=3)
            rew_h_v4 = heuristic(10000, 100, version='v4',
                                 seed=1234,  steps_to_request=3)
            plt.xlabel('Epochen')
            plt.ylabel('∅-Reward pro Step')
            plt.title('Bestellung alle 4 Steps')
            plt.plot(rew_q_e.get_smooth_all_episode_rewards_per_step(),
                     label='q-func')
            plt.plot(rew_q_e_order.get_smooth_all_episode_rewards_per_step(),
                     label='q-func-v2')
            plt.plot(
                rew_h_v4.get_smooth_all_episode_rewards_per_step(), label='heur')
            plt.legend()
            plt.show()

            rew_q_e.plot_pos_neg_rewards(name='q-ext-a0.1')
            rew_q_e_order.plot_pos_neg_rewards(name='q-ext-a0.1-order')
            rew_q_e.plot_episode_rewards(9800, name='Q-Function')
            rew_q_e_order.plot_episode_rewards(9800, name='Q-Function V2')
            rew_h_v4.plot_episode_rewards(9800, name='Heuristik')

        # Test Summary 1 One-Art---4Steps to request
        if(True):
            # TODO comment in in requests
            rew_q_e = q_function_extended_order(
                10000, 100, 0.1,  seed=1234, steps_to_request=4)
            rew_q_e_order = q_function_extended_order(
                10000, 100, 0.1,  seed=1234, simple_state=False, steps_to_request=4)
            rew_h_v4 = heuristic(1000, 100, version='v4',
                                 seed=1234,  steps_to_request=4)
            plt.xlabel('Epochen')
            plt.ylabel('∅-Reward pro Step')
            plt.title('Bestellung alle 4 Steps')
            plt.plot(rew_q_e.get_smooth_all_episode_rewards_per_step(),
                     label='q-func')
            plt.plot(rew_q_e_order.get_smooth_all_episode_rewards_per_step(),
                     label='q-func-v2')
            plt.plot(
                rew_h_v4.get_smooth_all_episode_rewards_per_step(), label='heur')
            plt.legend()
            plt.show()

            rew_q_e.print_q_matrix()
            # rew_q_e_order.print_q_matrix()

            rew_q_e.plot_pos_neg_rewards(name='q-ext-a0.1')
            rew_q_e_order.plot_pos_neg_rewards(name='q-ext-a0.1-order')
            rew_q_e.plot_episode_rewards(9800, name='Q-Function')
            rew_q_e_order.plot_episode_rewards(9800, name='Q-Function V2')
            rew_h_v4.plot_episode_rewards(980, name='Heuristik')

        # rew_q_w_i_a0_1 = q_function_with_idle(100, 1000, 0.1,  seed=1234)
        # rew_h_v4 = heuristic(100, version='v4', seed=1234)

        # Test 12 one art get best alpha -> validate with 100 seeds
        if(False):
            '''
            plt.plot(q_function_with_idle(100, 1000, 1, seed=1234).all_episode_rewards_per_step,
                     label='q-a1')
            plt.plot(q_function_with_idle(100, 1000, 0.9, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.9')
            plt.plot(q_function_with_idle(100, 1000, 0.8, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.8')
            plt.plot(q_function_with_idle(100, 1000, 0.7, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.7')
            plt.plot(q_function_with_idle(100, 1000, 0.6, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.6')
            plt.plot(q_function_with_idle(100, 1000, 0.5, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.5')
            plt.plot(q_function_with_idle(100, 1000, 0.4, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.4')
            plt.plot(q_function_with_idle(100, 1000, 0.3, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.3')
            plt.plot(q_function_with_idle(100, 1000, 0.2, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.2')
            plt.plot(q_function_with_idle(100, 1000, 0.1, seed=1234).all_episode_rewards_per_step,
                     label='q-a0.1')
                     '''
            a1 = []
            a2 = []
            a3 = []
            a4 = []
            a5 = []
            a6 = []
            a7 = []
            a8 = []
            a9 = []
            a10 = []

            for i in range(100):
                a1.append(np.mean(q_function_with_idle(
                    100, 1000, 0.1, seed=i**2).all_episode_rewards_per_step))
                a2.append(np.mean(q_function_with_idle(
                    100, 1000, 0.2, seed=i**2).all_episode_rewards_per_step))
                a3.append(np.mean(q_function_with_idle(
                    100, 1000, 0.3, seed=i**2).all_episode_rewards_per_step))
                a4.append(np.mean(q_function_with_idle(
                    100, 1000, 0.4, seed=i**2).all_episode_rewards_per_step))
                a5.append(np.mean(q_function_with_idle(
                    100, 1000, 0.5, seed=i**2).all_episode_rewards_per_step))
                a6.append(np.mean(q_function_with_idle(
                    100, 1000, 0.6, seed=i**2).all_episode_rewards_per_step))
                a7.append(np.mean(q_function_with_idle(
                    100, 1000, 0.7, seed=i**2).all_episode_rewards_per_step))
                a8.append(np.mean(q_function_with_idle(
                    100, 1000, 0.8, seed=i**2).all_episode_rewards_per_step))
                a9.append(np.mean(q_function_with_idle(
                    100, 1000, 0.9, seed=i**2).all_episode_rewards_per_step))
                a10.append(np.mean(q_function_with_idle(
                    100, 1000, 1, seed=i**2).all_episode_rewards_per_step))

            plt.plot(a1, label='q-a1')
            plt.plot(a2, label='q-a2')
            plt.plot(a3, label='q-a3')
            plt.plot(a4, label='q-a4')
            plt.plot(a5, label='q-a5')
            plt.plot(a6, label='q-a6')
            plt.plot(a7, label='q-a7')
            plt.plot(a8, label='q-a8')
            plt.plot(a9, label='q-a9')
            plt.plot(a10, label='q-a10')
            print('Mean a1:', np.mean(a1))
            print('Mean a2:', np.mean(a2))
            print('Mean a3:', np.mean(a3))
            print('Mean a4:', np.mean(a4))
            print('Mean a5:', np.mean(a5))
            print('Mean a6:', np.mean(a6))
            print('Mean a7:', np.mean(a7))
            print('Mean a8:', np.mean(a8))
            print('Mean a8:', np.mean(a9))
            print('Mean a10:', np.mean(a10))
            plt.legend()
            plt.show()
        # test_prob()
        # q_function(5000)
        # heuristic(5000, False)

    tests()
    # random_actions()
    # q_function(50000)
    # heuristic()


# TODO add config seeds to reproduce (with params steps and episodes)
# TODO improve qleraning ---> oracles? & random add oracles
# TODO or try heuristic without oracles
