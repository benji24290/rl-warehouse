import gym
import random
import sys


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


class ArticleCollection():
    def __init__(self):
        self.articles = []

    def _generate_articles(self):
        pass

    def get_possible_articles(self):
        return self.articles


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


class Storage():
    def __init__(self, count=3):
        self.storage_spaces = self._init_spaces(count)
        print('Initialized Storage', self.get_storage_state())
        print(self._is_space_empty(-2))

    def _init_spaces(self, count):
        # TODO dynamic size
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


class Arrivals():
    def __init__(self, count=1):
        self.max_arrivals = count
        self.arrivals = []
        pass

    def add_article_to_arrival(self, article):
        # TODO
        print(self.arrivals, 3, self.max_arrivals)


class ArrivalSpace():
    def __init__(self, article=None):
        # the stored article
        # Article
        self.article = article

    def store_article(self, article):
        self.article = article

    def retrieve_article(self):
        self.article = None


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


class Requests():
    def __init__(self, count=2):
        self.requests = []
        pass

    def generate_request(self):
        # TODO generate a request with prob.
        pass

    def get_requests_state(self):
        # TODO return all states as....
        pass


class WarehouseEnv(gym.Env):
    def __init__(self, seed=None, max_requests=3, max_arrivals=1):
        self.seed = seed
        self.max_requests = max_requests
        self.max_arrivals = max_arrivals
        if seed is None:
            self.seed = random.randint(0, sys.maxsize)
        self.arrivals = Arrivals(self.max_arrivals)
        self.possible_articles = []
        self.requests = []
        self.storage = Storage(3)
        print('Env initialized seed:', self.seed)
        self._add_articles()
        #self.storage.store(Article(3, 0.5,  "Normal"), 1)
        # print(self.storage.get_storage_state())

    def step(self):
        self._new_request()
        print('Step success!')

    def reset(self):
        print('env reset')

    # generates a new request
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
            Article(1, 0.2, "Selten"))
        self.possible_articles.append(
            Article(2, 0.8,  "Häufig"))
        self.possible_articles.append(
            Article(3, 0.5,  "Normal"))
        print("Articles added...")
    # def _add_storage_space(self):
    #   self.stock.append(StorageSpace(1))
    #   self.stock.append(StorageSpace(2))
    #   self.stock.append(StorageSpace(3))


a = WarehouseEnv(None, 3, 1)
