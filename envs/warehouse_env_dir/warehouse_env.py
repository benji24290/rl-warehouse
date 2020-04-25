
import random
import sys
import json
import itertools
import gym
import numpy as np
import matplotlib.pyplot as plt


# This reward values are the constant part and will be modified with distance
ORDER_POSITIVE_REWARD = 10
ORDER_NEGATIVE_REWARD = -50
STORE_POSITIVE_REWARD = 10
STORE_NEGATIVE_REWARD = -10
DELIVER_POSITIVE_REWARD = 100
DELIVER_NEGATIVE_REWARD = -100
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
            Article(0, 0.2, "Selten"))
        self.articles.append(
            Article(1, 0.8,  "Häufig"))
        self.articles.append(
            Article(2, 0.5,  "Normal"))
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
        # TODO return storage space info
        if self.article == None:
            return 0
        return self.article.get_id()

    def __str__(self):
        if self.article == None:
            return "{}"
        return self.article.__str__()


class Storage():
    def __init__(self, count=3):
        self.storage_spaces = self._init_spaces(count)
        log.info('Initialized storage', self.get_storage_state())

    def _init_spaces(self, count):
        # TODO dynamic size see articles
        return [StorageSpace(1), StorageSpace(2), StorageSpace(3)]

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

    def get_storage_state(self):
        # TODO return all states as....
        state = []
        for s in self.storage_spaces:
            state.append(s.get_storage_space_state())
        return state

    def store(self, article, pos):
        if self._is_space_empty(pos):
            self.storage_spaces[pos].store_article(article)
            return True
        return False

    def _is_space_empty(self, pos):
        try:
            return self.storage_spaces[pos].get_storage_space_state() == 0
        except IndexError:
            return False

    def __str__(self):
        spaces = []
        for s in self.storage_spaces:
            spaces.append(s)
        return ', '.join(map(str, spaces))


class Arrivals():
    def __init__(self, count=1):
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
        # TODO return storage space info
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


class Requests():
    def __init__(self, count=2):
        self.max_requests = count
        self.requests = []

    def deliver_article(self, article):
        # TODO every request has same
        if article is not None:
            for r in self.requests:
                if r.article == article:
                    self.requests.remove(r)
                    return True
        return False

    def generate_request(self, possible_articles):
        # TODO generate a request with prob.
        # TODO check if max req not reched
        if(len(self.requests) < self.max_requests):
            self.requests.append(
                Request(random.choice(possible_articles.articles)))
            log.info('generate_request:', 'added new request')
        else:
            log.warn('generate_request:', 'max requests reached')

    def get_requests_state(self):
        """Docstring"""
        # TODO return all states as....
        state = []
        for r in self.requests:
            state.append(r.get_request_state())
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
        # TODO make custom class for orders
        self.orders = Orders()
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
        # TODO deliver
        # TODO deliver with oracle
        article = self.env.possible_articles.get_article_by_id(article_id)
        if self.env.requests.deliver_article(article):
            log.info('deliver:', article, 'was delivered')
            return 100
        log.warn('deliver:', article, 'could not be delivered')
        return -100

    def order(self, article_id):
        self.orders.new_order(
            self.env.possible_articles.get_article_by_id(article_id))
        log.info('order:', 'order, now there ',
                 len(self.orders.orders), ' orders')
        return 0

    def do_action(self, action, article_id=None):
        self.action_reward = 0
        # TODO use parameter to pick a article
        if article_id is None:
            article_id = self.env.possible_articles.get_random_article().get_id()

        log.debug('do_action:', 'article', article_id)
        # print('arrived ', len(self.orders.update_orders()), ' orders')
        self.action_reward += self.handle_arrivals(self.orders.update_orders())
        if action == 'STORE':
            return self.store(article_id)+self.action_reward
        elif action == 'DELIVER':
            return self.deliver(article_id)+self.action_reward
        elif action == 'ORDER':
            return self.order(article_id)+self.action_reward

    def handle_arrivals(self, arrived_articles):
        arrival_reward = 0
        for a in arrived_articles:
            if self.env.arrivals.add_article_to_arrival(a):
                log.debug('handle_arrival:', 'Stored article',
                          self.env.arrivals.get_arrivals_state())
                arrival_reward += ORDER_POSITIVE_REWARD
            else:
                arrival_reward += ORDER_NEGATIVE_REWARD
                log.warn('handle_arrival:', 'full')
        log.debug('handle_arrival:', 'arrival reward is ', arrival_reward)
        return arrival_reward

    def action_random(self):
        return self.do_action(random.choice(self.actions))

    def calc_reward(self):
        # TODO
        return 0

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


class Logger():
    def __init__(self):
        self.show_debug = False
        self.show_info = False
        self.show_warn = True
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
        if(prefix != self.filter):
            return True
        return False


log = Logger()


class WarehouseEnv(gym.Env):
    def __init__(self, seed=None, max_requests=2, max_arrivals=1, max_turns=50):
        self.seed = seed
        self.max_requests = max_requests
        self.max_arrivals = max_arrivals
        self.max_turns = max_turns
        self.turn = 0
        self.game_over = False
        if seed is None:
            self.seed = random.randint(0, sys.maxsize)
        print('Env initialized seed:', self.seed)
        self._make_new_instances()
        self._add_test_values()
        self._print_env_state()

    def step(self):
        if self.turn < self.max_turns:
            self.requests.generate_request(self.possible_articles)
            reward = self.actions.action_random()
            log.info('Step ', self.turn)
            # TODO calc state and reward
            resulting_state = []
            self.turn += 1

            # state, reward, gameover, debug info
        else:
            log.info('Finished!')
            # TODO calc state and reward
            resulting_state = []
            reward = 0
            self.game_over = True
            # state, reward, gameover, debug info
        return resulting_state, reward, self.game_over, None

    def reset(self):
        self.game_over = False
        self.turn = 0
        self._make_new_instances()
        log.info('env reset')

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

    def getState(self):
        # TODO return all states including turn?
        return [self.storage.get_storage_state(), self.requests.get_requests_state(), self.arrivals.get_arrivals_state()]

    def _make_new_instances(self):
        self.arrivals = Arrivals(self.max_arrivals)
        self.possible_articles = ArticleCollection()
        self.requests = Requests(self.max_requests)
        self.storage = Storage(3)
        self.actions = Actions(self)

    def _add_test_values(self):
        # Storage article
        self.storage.store(Article(3, 0.5,  "Normal"), 2)
        # New request
        self.requests.generate_request(self.possible_articles)
        self.requests.generate_request(self.possible_articles)
        # New element in arrival
        self.arrivals.add_article_to_arrival(Article(2, 0.5,  "Normal"))


# a = WarehouseEnv(None, 2, 3, 50)

# Q Function
if __name__ == '__main__':
    env = WarehouseEnv(None, 2, 3, 50)

    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0
    # TODO init q here
    # Q = {}
    # for state in env.stateSpacePlus:
    #    for action in env.possible_articles:
    #        Q[state, action] = 0

    # TODO rename to episodes
    num_ep = 100
    total_rewards = np.zeros(num_ep)

    for i in range(num_ep):
        # Print all n episodes
        if i % 10 == 0:
            print('starting episode', i)
        done = False
        ep_rewards = 0
        # TODO Return something in env.reset
        # observation = env.reset()
        env.reset()
        while not done:
            # TODO seed?
            rand = np.random.random()
            # action = max_action(Q, observation, env.possible_actions)
            [state, reward, game_over, debug] = env.step()
            ep_rewards = ep_rewards + reward
            if env.game_over:
                break
        print('Episode ', i, ' reward is:', ep_rewards)

# DQN
