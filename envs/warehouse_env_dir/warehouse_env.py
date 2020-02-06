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


class WarehouseEnv(gym.Env):
    def __init__(self, seed=None, max_requests=3, max_arrivals=1):
        self.seed = seed
        self.max_requests = max_requests
        self.max_arrivals = max_arrivals
        if seed is None:
            self.seed = random.randint(0, sys.maxsize)
        self.possible_articles = []
        self.requests = []
        self.stock = []
        print('Env initialized seed:', self.seed)
        self._add_articles()

    def step(self):
        self._new_request()
        print('Step success!')

    def reset(self):
        print('env reset')

    # generates a new
    def _new_request(self):
        # TODO pick random but with frequency and check for max_request
        self.requests.append(Request(random.choice(self.possible_articles)))

    def _print_env_info(self):
        self._print_requests()

    def _print_requests(self):
        print("------- Requests -------")
        for obj in self.requests:
            print(obj)

    def _add_articles(self):
        self.possible_articles.append(
            Article(0.2, 0.2, "Selten und unverfügbar"))
        self.possible_articles.append(
            Article(0.8, 0.2, "Häufig und unverfügbar"))
        self.possible_articles.append(
            Article(0.2, 0.8, "Selten und verfügbar"))
        self.possible_articles.append(
            Article(0.8, 0.8, "Häufig und verfügbar"))
        print("Articles added...")
