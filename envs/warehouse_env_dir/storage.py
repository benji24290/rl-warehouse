try:
    from envs.warehouse_env_dir.storage_space import StorageSpace
    from envs.warehouse_env_dir.consts import INVENTORY_COST_FACTOR
    from envs.warehouse_env_dir.logger import log
except ModuleNotFoundError:
    from storage_space import StorageSpace
    from consts import INVENTORY_COST_FACTOR
    from logger import log


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
