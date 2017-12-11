import json

import newspaper
import requests
from newspaper import Article
from nytimesarticle import articleAPI


class NewsMiner:
    def __init__(self):
        self.client = newspaper
        self.sources = {}
        self.article_api = articleAPI('60425d8974b1484692c368f8c52e4c1f')
        self.nyi_api_key = "60425d8974b1484692c368f8c52e4c1f"

    def setup(self):
        self.get_articles()
        # self.add_source()

    def add_source(self):
        self.sources['cnn'] = self.client.build('http://cnn.com', memoize_articles=False)

    def get_articles(self, search_term, begin_date, end_date):
        articles = []
        url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        for page in range(100):
            queries = {
                'api-key': self.nyi_api_key,
                'q': search_term,
                'begin_date': begin_date,
                'end_date': end_date,
                'page': page
            }
            req_t = requests.get(url, params=queries)
            data = json.loads(req_t.text)
            if 'response' in data:
                articles.extend(data['response']['docs'])
        return articles

    def get_text(self, articles):
        for id,a in articles.items():
            try:
                article = Article(a['web_url'])
                article.download()
                article.parse()
                article.nlp()
                a['summary'] = article.summary
                a['keywords_1'] = article.keywords
                a['text'] = article.text
            except:
                pass
