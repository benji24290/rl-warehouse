from envs.warehouse_env_dir.order import Order


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
