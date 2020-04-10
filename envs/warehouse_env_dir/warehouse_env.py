
import random
import sys
import json
import itertools
import gym
import numpy as np
import matplotlib.pyplot as plt


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

    def getName(self):
        return self.name

    def getId(self):
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
        print("Possible articles added")

    def get_possible_articles(self):
        possible = []
        for a in self.articles:
            possible.append(a.getId())
        return possible

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
        return self.article.getId()

    def __str__(self):
        if self.article == None:
            return "{}"
        return self.article.__str__()


class Storage():
    def __init__(self, count=3):
        self.storage_spaces = self._init_spaces(count)
        print('Initialized storage', self.get_storage_state())

    def _init_spaces(self, count):
        # TODO dynamic size see articles
        return [StorageSpace(1), StorageSpace(2), StorageSpace(3)]

    def get_possible_space(self):
        # TODO return random possible space
        pass

    def get_storage_state(self):
        # TODO return all states as....
        state = []
        for s in self.storage_spaces:
            state.append(s.get_storage_space_state())
        return state

    def store(self, article, pos):
        if self._is_space_empty(pos):
            self.storage_spaces[pos].store_article(article)
        # TODO throw error

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
        print('Initialized arrivals', self.get_arrivals_state())

    def _initialize_spaces(self, count):
        for _ in itertools.repeat(None, count):
            self.arrivals.append(ArrivalSpace())

    def add_article_to_arrival(self, article):
        pass

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
        self.article = article

    def retrieve_article(self):
        self.article = None

    def get_arrival_space_state(self):
        # TODO return storage space info
        if self.article == None:
            return 0
        return self.article.getId()


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
        return [self.article.getId(), self.time, self.reward_value]


class Requests():
    def __init__(self, count=2):
        self.requests = []
        pass

    def generate_request(self, possible_articles):
        # TODO generate a request with prob.
        # TODO check if max req not reched
        self.requests.append(
            Request(random.choice(possible_articles.articles)))

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
            print('Step ', self.turn)
            # TODO calc state and reward
            resulting_state = []
            self.turn += 1
            reward = 0
            # state, reward, gameover, debug info
        else:
            print('Finished!')
            # TODO calc state and reward
            resulting_state = []
            reward = 0
            self.game_over = True
            # state, reward, gameover, debug info
        return resulting_state, reward, self.game_over, None

    def reset(self):
        self.game_over = False
        self.turn = 0
        self._make_new_instances
        print('env reset')

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

    def _add_test_values(self):
        # Storage article
        self.storage.store(Article(3, 0.5,  "Normal"), 2)
        # New request
        self.requests.generate_request(self.possible_articles)
        self.requests.generate_request(self.possible_articles)
        # New element in arrival
        self.arrivals.add_article_to_arrival(Article(2, 0.5,  "Normal"))


#a = WarehouseEnv(None, 2, 3, 50)

# Q Function
if __name__ == '__main__':
    env = WarehouseEnv(None, 2, 3, 50)

    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0
    # TODO init q here
    #Q = {}
    # for state in env.stateSpacePlus:
    #    for action in env.possible_articles:
    #        Q[state, action] = 0

    # TODO rename to episodes
    num_games = 100
    total_rewards = np.zeros(num_games)

    for i in range(num_games):
        # Print all n games
        if i % 10 == 0:
            print('starting episode', i)
        done = False
        ep_rewards = 0
        # TODO Return something in env.reset
        #observation = env.reset()
        env.reset()
        while not done:
            # TODO seed?
            rand = np.random.random()
            print(rand)
            #action = max_action(Q, observation, env.possible_actions)
            env.step()
            if env.game_over:
                break

# DQN
