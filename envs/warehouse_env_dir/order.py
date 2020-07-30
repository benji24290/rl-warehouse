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

    def get_simple_order_state(self):
        return self.article.get_id()
