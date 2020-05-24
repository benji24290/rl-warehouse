
class ArrivalSpace():
    def __init__(self, article=None):
        # the stored article
        # Article
        self.article = article

    def store_article(self, article):
        if self.article == None:
            self.article = article
            return True
        return False

    def retrieve_article(self):
        a = self.article
        self.article = None
        return a

    def get_arrival_space_state(self):
        # TODO return more arrival space info
        if self.article == None:
            return 0
        return self.article.get_id()

    def get_simple_arrival_space_state(self):
        if self.article == None:
            return 0
        return self.article.get_id()
