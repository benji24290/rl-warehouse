try:
    from envs.warehouse_env_dir.article import Article
    from envs.warehouse_env_dir.logger import log
except ModuleNotFoundError:
    from article import Article
    from logger import log
import random


class ArticleCollection():
    def __init__(self):
        # TODO dynamic count
        self.articles = []
        self._generate_articles()

    def _generate_articles(self):
        self.articles.append(
            Article(1, 0.2, "Selten"))
        self.articles.append(
            Article(2, 0.8,  "Häufig"))
        # self.articles.append(
        #    Article(3, 0.1,  "Normal"))
        log.info("Possible articles added")

    def get_possible_articles(self):
        possible = []
        for a in self.articles:
            possible.append(a.get_id())
        return possible

    def get_random_article(self):
        return random.choice(self.articles)

    def get_random_article_with_prob(self):
        arr = []
        for art in self.articles:
            arr = arr+([art]*(int)(100*art.frequency))
        return random.choice(arr)

    def get_article_by_id(self, id):
        for a in self.articles:
            if(a.get_id() == id):
                return a
        return None

    def __str__(self):
        possible = []
        for a in self.articles:
            possible.append(a)
        return ', '.join(map(str, possible))
