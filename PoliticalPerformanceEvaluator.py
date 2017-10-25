import pprint

from mining.TwitterMiner import TwitterMiner


def get_tweets():
    return TwitterMiner().login().search.tweets(q="#trump", count=100)


if __name__ == "__main__":
    pprint.pprint(get_tweets())
