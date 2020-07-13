class EnvConfig():
    def __init__(self, seed=None, max_requests=2, max_arrivals=2, storage_spaces=3, turns=100, steps_to_request=4, simple_state=False):
        self.seed = seed
        self.max_requests = max_requests
        self.max_arrivals = max_arrivals
        self.storage_spaces = storage_spaces
        self.turns = turns
        self.steps_to_request = steps_to_request
        self.simple_state = simple_state
