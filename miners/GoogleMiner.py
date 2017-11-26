import requests
from bs4 import BeautifulSoup


class GoogleMiner:

    def get_search_summary(self,search_query):
        search_query = search_query.replace(" ", "+")
        query = "https://www.google.com/search?q=" + search_query
        r = requests.get(query)
        html_doc = r.text
        soup = BeautifulSoup(html_doc, 'html.parser')
        return soup.find_all(attrs={'class': 'st'})