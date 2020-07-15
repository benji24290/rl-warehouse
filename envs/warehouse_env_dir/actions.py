try:
    from envs.warehouse_env_dir.logger import log
except ModuleNotFoundError:
    from logger import log
import random


class Actions():
    def __init__(self, env):
        self.env = env
        self.actions = ['STORE', 'DELIVER', 'ORDER']
        self.actions_extended = ['STORE', 'DELIVER',  'IDLE']
        action_count = len(self.env.possible_articles.articles)
        for i in range(action_count):
            self.actions_extended.append('ORDER_'+str(i+1))
        self.actions_with_idle = ['STORE', 'DELIVER',
                                  'ORDER',  'IDLE']
        self.action_reward = 0

    def store(self, article_id, storage_pos=None):
        if storage_pos is None:
            storage_pos = self.store_oracle()
        if(storage_pos is not None and self.env.storage.storage_spaces[storage_pos] is not None):
            article = self.env.arrivals.remove_article_from_arrival(
                self.env.possible_articles.get_article_by_id(article_id))
            if article is not None:
                if(self.env.storage.store(article, storage_pos)):
                    log.info('store: ' + article.get_name() +
                             'in storage pos: ', storage_pos)
                    return 0
                log.info('store: space is already used...')
                return -10
            log.info('store: article not found')
            return -10
        log.info('store: storage space not found')
        return 0  # storage space

    def deliver(self, article_id):
        # TODO deliver with oracle
        article = self.env.possible_articles.get_article_by_id(article_id)
        if self.env.requests.deliver_article(article):
            log.info('deliver:', article, 'was delivered')
            return 100
        log.warn('deliver:', article, 'could not be delivered')
        print(self.env.requests.get_simple_requests_state(),
              self.env.storage.get_simple_storage_state())
        return -100

    def order(self, article_id):
        rew = self.env.orders.new_order(
            self.env.possible_articles.get_article_by_id(article_id))
        log.info('order:', 'order, now there ',
                 len(self.env.orders.orders), ' orders')
        return rew

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
                self.deliver(self.deliver_oracle()))
        elif action == 'ORDER':
            self.env.rewards.add_reward_action_order(self.order(article_id))
            # TODO Remove and make dynamic
        elif action == 'ORDER_1':
            self.env.rewards.add_reward_action_order(self.order(1))
        elif action == 'ORDER_2':
            self.env.rewards.add_reward_action_order(self.order(2))
        elif action == 'ORDER_3':
            self.env.rewards.add_reward_action_order(self.order(3))
        elif action == 'IDLE':
            pass
            # print('idle this step....')

    def action_random(self):
        return self.do_action(random.choice(self.actions))

    def get_random_action(self):
        return random.choice(self.actions)

    def get_random_action_extended(self):
        return random.choice(self.actions_extended)

    def get_random_action_with_idle(self):
        return random.choice(self.actions_with_idle)

    def store_oracle(self):
        possible = self.env.storage.get_possible_spaces()
        if(len(possible) > 0):
            return random.choice(possible)
        return None

    def deliver_oracle(self):
        a_id = None
        for req in self.env.requests.requests:
            if(a_id != None):
                break
            for space in self.env.storage.storage_spaces:
                if(a_id != None):
                    break
                if req.article == space.article:
                    a_id = req.article.id
        return a_id

#    def arrival_oracle(self):
#        # random choice of possible?
#        possible = self.env.arrivals.get_possible_spaces()
#        if(len(possible) > 0):
#            return random.choice(self.actions)
#        return None

#    def deliver_oracle(self):
#        return 0
