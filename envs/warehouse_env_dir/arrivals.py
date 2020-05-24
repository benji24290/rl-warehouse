import itertools
from envs.warehouse_env_dir.arrival_space import ArrivalSpace
from envs.warehouse_env_dir.logger import log
from envs.warehouse_env_dir.consts import ORDER_POSITIVE_REWARD, ORDER_NEGATIVE_REWARD


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
