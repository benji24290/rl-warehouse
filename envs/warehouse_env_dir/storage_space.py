class StorageSpace():
    def __init__(self, article=None):
        # the stored article
        # Article
        self.article = article

    def store_article(self, article):
        self.article = article

    def retrieve_article(self):
        self.article = None

    def get_storage_space_state(self):
        if self.article == None:
            return 0
        return [self.article.get_id()]

    def get_simple_storage_space_state(self):
        if self.article == None:
            return 0
        return self.article.get_id()

    def __str__(self):
        if self.article == None:
            return "{}"
        return self.article.__str__()
