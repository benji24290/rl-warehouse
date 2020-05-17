
import random
import sys
import json
import itertools
import gym
from itertools import product
import numpy as np
import matplotlib.pyplot as plt


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


class Article():
    # frequency is the probability that an article of this type is requested
    # float 0-1
    # name of this article
    # string
    # delivery time of this article, between 0-1 smaller availability takes longer to get delivered to the warehouse
    # int
    def __init__(self, id, frequency=0.5, name="article name", delivery_time=2):
        self.id = id
        self.frequency = frequency
        self.name = name
        self.delivery_time = delivery_time

    def get_name(self):
        return self.name

    def get_id(self):
        return self.id

    def __str__(self):
        return json.dumps(self.__dict__)


class ArticleCollection():
    def __init__(self):
        # TODO dynamic count
        self.articles = []
        self._generate_articles()

    def _generate_articles(self):
        self.articles.append(
            Article(1, 0.2, "Selten"))
        self.articles.append(
            Article(2, 0.8,  "Häufig"))
        self.articles.append(
            Article(3, 0.5,  "Normal"))
        log.info("Possible articles added")

    def get_possible_articles(self):
        possible = []
        for a in self.articles:
            possible.append(a.get_id())
        return possible

    def get_random_article(self):
        return random.choice(self.articles)

    def get_article_by_id(self, id):
        for a in self.articles:
            if(a.get_id() == id):
                return a
        return None

    def __str__(self):
        possible = []
        for a in self.articles:
            possible.append(a)
        return ', '.join(map(str, possible))


class StorageSpace():
    def __init__(self, distance, article=None):
        # the distance
        # int [1,2,3]
        self.distance = distance
        # the stored article
        # Article
        self.article = article

    def store_article(self, article):
        self.article = article

    def retrieve_article(self):
        self.article = None

    def get_storage_space_state(self):
        if self.article == None:
            return 0
        return [self.article.get_id(), self.distance]

    def get_simple_storage_space_state(self):
        if self.article == None:
            return 0
        return self.article.get_id()

    def __str__(self):
        if self.article == None:
            return "{}"
        return self.article.__str__()


class Storage():
    def __init__(self, count):
        self.storage_spaces = []
        self._init_spaces(count)
        log.info('Initialized storage', self.get_storage_state())

    def _init_spaces(self, count):
        # generates all the storage spaces, the distance is equal to n
        # TODO maybe diastance  %3
        for i in range(count):
            self.storage_spaces.append(StorageSpace(i))

    def get_possible_space(self):
        # TODO return random possible space
        pass

    def get_possible_spaces(self):
        '''Returns the index of all the possible spaces'''
        spaces = []
        for i in range(len(self.storage_spaces)):
            if(self.storage_spaces[i].article == None):
                spaces.append(i)
        return spaces

    def get_used_spaces_count(self):
        '''Returns the the count of used spaces

        Returns
        ------
        int: count of used spaces
        '''
        count = 0
        for i in range(len(self.storage_spaces)):
            if(self.storage_spaces[i].article is not None):
                count += 1
        return count

    def get_storage_state(self):
        state = []
        for s in self.storage_spaces:
            state.append(s.get_storage_space_state())
        return state

    def get_simple_storage_state(self):
        state = []
        for s in self.storage_spaces:
            state.append(s.get_simple_storage_space_state())
        return state

    def store(self, article, pos):
        if self._is_space_empty(pos):
            self.storage_spaces[pos].store_article(article)
            return True
        return False

    def get_storage_reward(self):
        '''Calculates the storage cost: returns int'''
        return self.get_used_spaces_count()*-1*INVENTORY_COST_FACTOR

    def _is_space_empty(self, pos):
        try:
            return self.storage_spaces[pos].get_simple_storage_space_state() == 0
        except IndexError:
            return False

    def __str__(self):
        spaces = []
        for s in self.storage_spaces:
            spaces.append(s)
        return ', '.join(map(str, spaces))


class Arrivals():
    def __init__(self, count):
        self.max_arrivals = count
        self.arrivals = []
        self._initialize_spaces(self.max_arrivals)
        log.info('Initialized arrivals', self.get_arrivals_state())

    def _initialize_spaces(self, count):
        for _ in itertools.repeat(None, count):
            self.arrivals.append(ArrivalSpace())

    def add_article_to_arrival(self, article):
        for space in self.arrivals:
            if space.store_article(article):
                return True
        return False

    def remove_article_from_arrival(self, article):
        for a in self.arrivals:
            if a.article == article:
                return a.retrieve_article()
        return None

    def get_arrivals_state(self):
        state = []
        for a in self.arrivals:
            state.append(a.get_arrival_space_state())
        return state

    def get_simple_arrivals_state(self):
        state = []
        for a in self.arrivals:
            state.append(a.get_simple_arrival_space_state())
        return state

    def handle_arrivals(self, arrived_articles):
        arrival_reward = 0
        for a in arrived_articles:
            if self.add_article_to_arrival(a):
                log.debug('handle_arrival:', 'Stored article',
                          self.get_arrivals_state())
                arrival_reward += ORDER_POSITIVE_REWARD
            else:
                arrival_reward += ORDER_NEGATIVE_REWARD
                log.warn('handle_arrival:', 'full')
        log.debug('handle_arrival:', 'arrival reward is ', arrival_reward)
        return arrival_reward


class ArrivalSpace():
    def __init__(self, article=None):
        # the stored article
        # Article
        self.article = article

    def store_article(self, article):
        if self.article == None:
            self.article = article
            return True
        return False

    def retrieve_article(self):
        a = self.article
        self.article = None
        return a

    def get_arrival_space_state(self):
        # TODO return more arrival space info
        if self.article == None:
            return 0
        return self.article.get_id()

    def get_simple_arrival_space_state(self):
        if self.article == None:
            return 0
        return self.article.get_id()


class Request():
    def __init__(self, article):
        # the reward value of this request
        # float 0-1
        self.reward_value = 1
        # the max steps to deliver else the request will be cancelled and added a negative reward
        # int
        self.time = 5
        # the article type gets random article
        # Article
        self.article = article

    def __str__(self):
        return "{} in {} with reward: {}".format(
            self.article.name, self.time, self.reward_value)

    def get_request_state(self):
        return [self.article.get_id(), self.time, self.reward_value]

    def get_simple_request_state(self):
        return self.article.get_id()


class Requests():
    def __init__(self, env, count):
        self.env = env
        self.max_requests = count
        self.requests = []

    def deliver_article(self, article):
        # TODO every request has same
        if article is not None:
            for space in self.env.storage.storage_spaces:
                if space.article == article:
                    for r in self.requests:
                        if r.article == article:
                            space.retrieve_article()
                            self.requests.remove(r)
                            return True
        return False

    def generate_request(self, possible_articles):
        # TODO generate a request with prob.
        if(len(self.requests) < self.max_requests):
            self.requests.append(
                Request(random.choice(possible_articles.articles)))
            log.info('generate_request:', 'added new request')
        else:
            log.warn('generate_request:', 'max requests reached')

    def update_requests(self):
        reward = 0
        for r in self.requests:
            if(r.time > 1):
                r.time = r.time-1
            else:
                self.requests.remove(r)
                reward = reward + REQUEST_EXPIRED_REWARD
                log.warn('update_request:', 'time expired, reward:', reward)
        return reward

    def get_requests_state(self):
        """Returns all pending requests as arrays"""
        state = []
        for r in self.requests:
            state.append(r.get_request_state())
        return state

    def get_simple_requests_state(self):
        """Returs all pending request in one 1d array"""
        state = []
        for i in range(self.max_requests):
            try:
                state.append(self.requests[i].get_simple_request_state())
            except:
                state.append(0)
        return state

    def __str__(self):
        reqs = []
        for r in self.requests:
            reqs.append(r)
        return ', '.join(map(str, reqs))

    def _print_requests(self):
        print("------- Requests -------")
        for obj in self.requests:
            print(obj)


class Order():
    def __init__(self, article):
        self.article = article
        self.time_to_deliver = article.delivery_time

    def has_arrived(self):
        if self.time_to_deliver > 0:
            return False
        return True

    def decrease_time(self):
        self.time_to_deliver -= 1

    def __str__(self):
        return self.article, ' in ', self.time_to_deliver


class Orders():
    def __init__(self):
        self.orders = []

    def new_order(self, article):
        self.orders.append(Order(article))

    def update_orders(self):
        arrived_articles = []
        for o in self.orders:
            o.decrease_time()
            if o.has_arrived():
                arrived_articles.append(o.article)
                self.orders.remove(o)
        return arrived_articles


class Actions():
    def __init__(self, env):
        self.actions = ['STORE', 'DELIVER', 'ORDER']
        self.env = env
        self.action_reward = 0

    def store(self, article_id, storage_pos=None):
        if storage_pos is None:
            storage_pos = self.store_oracle()
        if(storage_pos is not None and self.env.storage.storage_spaces[storage_pos].distance is not None):
            article = self.env.arrivals.remove_article_from_arrival(
                self.env.possible_articles.get_article_by_id(article_id))
            if article is not None:
                if(self.env.storage.store(article, storage_pos)):
                    log.info('store: ' + article.get_name() +
                             'in storage pos: ', storage_pos)
                    return 10 * self.env.storage.storage_spaces[storage_pos].distance
                log.info('store: space is already used...')
                return -10 * self.env.storage.storage_spaces[storage_pos].distance
            log.info('store: article not found')
            return -10 * self.env.storage.storage_spaces[storage_pos].distance
        log.info('store: storage space not found')
        return 0  # storage space

    def deliver(self, article_id):
        # TODO deliver with oracle
        article = self.env.possible_articles.get_article_by_id(article_id)
        if self.env.requests.deliver_article(article):
            log.info('deliver:', article, 'was delivered')
            return 100
        log.warn('deliver:', article, 'could not be delivered')
        return -100

    def order(self, article_id):
        self.env.orders.new_order(
            self.env.possible_articles.get_article_by_id(article_id))
        log.info('order:', 'order, now there ',
                 len(self.env.orders.orders), ' orders')
        return 0

    def do_action(self, action, article_id=None):
        '''Performs specified action, if article_id is None a random id will be generated. Does not return anything, the reward will be added to the env.reward'''
        # TODO use parameter to pick a article
        if action is None:
            # print('random because:', action)
            return self.action_random()

        if article_id is None:
            article_id = self.env.possible_articles.get_random_article().get_id()

        log.debug('do_action:', 'article', article_id)

        if action == 'STORE':
            self.env.rewards.add_reward_action_store(self.store(article_id))
        elif action == 'DELIVER':
            self.env.rewards.add_reward_action_deliver(
                self.deliver(article_id))
        elif action == 'ORDER':
            self.env.rewards.add_reward_action_order(self.order(article_id))
        elif action == 'IDLE':
            print('idle this step....')

    def action_random(self):
        return self.do_action(random.choice(self.actions))

    def get_random_action(self):
        return random.choice(self.actions)

    def store_oracle(self):
        # TODO make store oracle inteligent
        possible = self.env.storage.get_possible_spaces()
        if(len(possible) > 0):
            return random.choice(possible)
        return None

#    def arrival_oracle(self):
#        # random choice of possible?
#        possible = self.env.arrivals.get_possible_spaces()
#        if(len(possible) > 0):
#            return random.choice(self.actions)
#        return None

#    def deliver_oracle(self):
#        return 0


class Rewards():
    def __init__(self):
        self.all_episode_rewards = []
        self.all_rewards_loop_storage = []
        self.all_rewards_loop_request_updates = []
        self.all_rewards_loop_arrival = []
        self.all_rewards_action_deliver = []
        self.all_rewards_action_store = []
        self.all_rewards_action_order = []
        self.step = 0
        self.step_reward = 0
        self.total_episode_reward = 0
        self.rewards_loop_storage = []
        self.rewards_loop_request_updates = []
        self.rewards_loop_arrival = []
        self.rewards_action_deliver = []
        self.rewards_action_store = []
        self.rewards_action_order = []

    def reset_episode(self):
        if self.step > 0:
            self.all_episode_rewards.append(self.total_episode_reward)
            self.all_rewards_loop_storage.append(self.rewards_loop_storage)
            self.all_rewards_loop_request_updates.append(
                self.rewards_loop_request_updates)
            self.all_rewards_loop_arrival.append(self.rewards_loop_arrival)
            self.all_rewards_action_deliver.append(self.rewards_action_deliver)
            self.all_rewards_action_store.append(self.rewards_action_store)
            self.all_rewards_action_order.append(self.rewards_action_order)
            self.step = 0
            self.step_reward = 0
            self.total_episode_reward = 0
            self.rewards_loop_storage = []
            self.rewards_loop_request_updates = []
            self.rewards_loop_arrival = []
            self.rewards_action_deliver = []
            self.rewards_action_store = []
            self.rewards_action_order = []

    def add_reward_loop_storage(self, reward):
        self.rewards_loop_storage.append(reward)

    def add_reward_loop_request_updates(self, reward):
        self.rewards_loop_request_updates.append(reward)

    def add_reward_loop_arrival(self, reward):
        self.rewards_loop_arrival.append(reward)

    def add_reward_action_deliver(self, reward):
        self.rewards_action_deliver.append(reward)

    def add_reward_action_store(self, reward):
        self.rewards_action_store.append(reward)

    def add_reward_action_order(self, reward):
        self.rewards_action_order.append(reward)

    def calculate_step_reward(self):
        self.step_reward = 0
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_loop_storage)
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_loop_request_updates)
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_loop_arrival)
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_action_deliver)
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_action_store)
        self.step_reward += self.get_step_reward_of_array(
            self.rewards_action_order)
        # self.episode_rewards.append(self.step_reward)
        self.total_episode_reward += self.step_reward
        self.step += 1
        return self.step_reward

    def get_total_episode_reward(self):
        return self.total_episode_reward

    def print_final_reward_infos(self):
        self.reset_episode()
        print('________________')
        print('Mean:', np.mean(self.all_episode_rewards))
        print('First episode rew:',
              self.all_episode_rewards[0])
        print('Last episode rew:',
              self.all_episode_rewards[len(self.all_episode_rewards)-1])

    def get_step_reward_of_array(self, array):
        try:
            step_reward = array[self.step]
            return step_reward
        except IndexError:
            array.append(0)
            log.debug('get_step_reward_of_array:', 'no index')
            return 0

    def plot_total_episode_rewards(self):
        '''Plots the total reward for each episode'''
        self.reset_episode()
        plt.plot(self.all_episode_rewards)
        plt.show()

    def plot_episode_rewards(self, episode):
        '''Plots the all rewards for a given episode'''
        self.reset_episode()
        try:
            plt.plot(
                self.all_rewards_loop_storage[episode], label='l_storage')
            plt.plot(
                self.all_rewards_loop_request_updates[episode], label='l_request')
            plt.plot(self.all_rewards_loop_arrival[episode], label='l_arrival')
            plt.plot(
                self.all_rewards_action_deliver[episode], label='a_deliver')
            plt.plot(self.all_rewards_action_store[episode], label='a_store')
            plt.plot(self.all_rewards_action_order[episode], label='a_order')
        except:
            log.error('plot_episode_rewards:', 'no valid episode', episode)
        plt.legend()
        plt.show()

    def plot_loot_storage(self, episode=None):
        '''Plots the loot_storage reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_loop_storage[episode])
        except:
            for episode in self.all_rewards_loop_storage:
                plt.plot(episode)
        plt.show()

    def plot_loot_request_updates(self, episode=None):
        '''Plots the loop_request_updates reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_loop_request_updates[episode])
        except:
            for episode in self.all_rewards_loop_request_updates:
                plt.plot(episode)
        plt.show()

    def plot_loot_arrival(self, episode=None):
        '''Plots the loot_arrival reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_loop_arrival[episode])
        except:
            for episode in self.all_rewards_loop_arrival:
                plt.plot(episode)
        plt.show()

    def plot_action_deliver(self, episode=None):
        '''Plots the action_deliver reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_action_deliver[episode])
        except:
            for episode in self.all_rewards_action_deliver:
                plt.plot(episode)
        plt.show()

    def plot_action_store(self, episode=None):
        '''Plots the action_store reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_action_store[episode])
        except:
            for episode in self.all_rewards_action_store:
                plt.plot(episode)
        plt.show()

    def plot_action_order(self, episode=None):
        '''Plots the action_order reward foreach step in each episode.
        if episode is defined only given episode will be plotted'''
        self.reset_episode()
        try:
            plt.plot(self.all_rewards_action_order[episode])
        except:
            for episode in self.all_rewards_action_order:
                plt.plot(episode)
        plt.show()


class Logger():
    def __init__(self):
        self.show_debug = False
        self.show_info = False
        self.show_warn = False
        self.show_error = True
        self.show_type = True
        self.filter = ""  # "handle_arrival:"

    def info(self, *arg):
        if self.show_info and self._show(arg[0]):
            if self.show_type:
                print("INFO:", *arg)
            else:
                print(*arg)

    def debug(self, *arg):
        if self.show_debug and self._show(arg[0]):
            if self.show_type:
                print("DEBUG:", *arg)
            else:
                print(*arg)

    def warn(self, *arg):
        if self.show_warn and self._show(arg[0]):
            if self.show_type:
                print("WARN:", *arg)
            else:
                print(*arg)

    def error(self, *arg):
        if self.show_error and self._show(arg[0]):
            if self.show_type:
                print("ERROR:", *arg)
            else:
                print(*arg)

    def _show(self, prefix):
        if(self.filter == "" or prefix == self.filter):
            return True
        return False


log = Logger()


class WarehouseEnv(gym.Env):
    def __init__(self, seed=None, max_requests=2, max_arrivals=1, storage_spaces=3, max_turns=50, simple_state=True):
        self.seed = seed
        self.rewards = Rewards()
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
def random_actions(count=100):
    # env = WarehouseEnv(None, 2, 2, 3, 50)
    env = WarehouseEnv(None, 2, 2, 3, 1000)

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
    return env.rewards.all_episode_rewards


# Q Function
def q_function(count=50000):

    def max_action(Q, state, actions):
        values = np.array([Q[state, a] for a in actions])
        action = np.argmax(values)
        # print('best action is', action, actions[action])
        return actions[action]

    env = WarehouseEnv(None, 2, 2, 3, 1000)

    ALPHA = 0.1  # learningrate
    GAMMA = 1.0
    EPS = 1.0  # eps greedy action selection

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

    print(Q)
    env.rewards.print_final_reward_infos()
    # env.rewards.plot_total_episode_rewards()
    # env.rewards.plot_episode_rewards(1)
    # env.rewards.plot_episode_rewards(num_ep-1)
    return env.rewards.all_episode_rewards


# DQN
# TODO add dqn


# Heuristic


def heuristic(count=100):
    # env = WarehouseEnv(None, 2, 2, 3, 50)
    env = WarehouseEnv(None, 2, 2, 3, 1000)

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

    return env.rewards.all_episode_rewards


if __name__ == '__main__':
    def plot():
        plt.plot(random_actions(5000), label='random')
        plt.plot(q_function(5000), label='Q')
        plt.plot(heuristic(5000), label='heur')
        plt.legend()
        plt.show()

    plot()

    # random_actions()
    # q_function(50000)
    # heuristic()


# TODO add config seeds to reproduce (with params steps and episodes)
# TODO improve qleraning ---> oracles? & random add oracles
# TODO or try heuristic without oracles
