from envs.warehouse_env_dir.request import Request
from envs.warehouse_env_dir.logger import log
from envs.warehouse_env_dir.consts import REQUEST_EXPIRED_REWARD


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
        if(len(self.requests) < self.max_requests):
            self.requests.append(
                Request(possible_articles.get_random_article_with_prob()))
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
