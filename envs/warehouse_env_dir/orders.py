try:
    from envs.warehouse_env_dir.order import Order
    from envs.warehouse_env_dir.consts import ORDER_POSITIVE_REWARD, ORDER_NEGATIVE_REWARD
except ModuleNotFoundError:
    from order import Order
    from consts import ORDER_POSITIVE_REWARD, ORDER_NEGATIVE_REWARD


class Orders():
    def __init__(self):
        self.orders = []
        self.max_orders = 2

    def new_order(self, article):
        if(len(self.orders) < 2):
            self.orders.append(Order(article))
            return ORDER_POSITIVE_REWARD
        else:
            return ORDER_NEGATIVE_REWARD

    def update_orders(self):
        arrived_articles = []
        for o in self.orders:
            o.decrease_time()
            if o.has_arrived():
                arrived_articles.append(o.article)
                self.orders.remove(o)
        return arrived_articles

    def get_simple_orders_state(self):
        """Returs all open orders in one 1d array"""
        state = []
        for i in range(self.max_orders):
            try:
                state.append(self.orders[i].get_simple_order_state())
            except:
                state.append(0)
        return state
