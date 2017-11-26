import requests
from bs4 import BeautifulSoup


class GoogleMiner:

    def get_search_summary(self,search_query):
        search_result_limit = 100
        search_query = search_query.replace(" ", "+")
        query = "https://www.google.com/search?q=" + search_query + "&num=" + str(search_result_limit)
        r = requests.get(query)
        html_doc = r.text
        soup = BeautifulSoup(html_doc, 'html.parser')
        return soup.find_all(attrs={'class': 'st'})