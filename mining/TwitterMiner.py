import os

import twitter
from twitter import read_token_file, oauth_dance, write_token_file


class TwitterMiner:
    def __init__(self):
        ""

    def login(self):
        APP_NAME = 'PPE'
        CONSUMER_KEY = 'JK9vbq72nrL8BwBBUhOKASosL'
        CONSUMER_SECRET = 'tQvz6zTa3nMbLMISPRMjFW6UDIpUBoDjMoFqg3zCI9SuoQrEHT'
        TOKEN_FILE = 'out/twitter.oauth'

        try:
            (oauth_token, oauth_token_secret) = read_token_file(TOKEN_FILE)
        except IOError, e:
            (oauth_token, oauth_token_secret) = oauth_dance(APP_NAME, CONSUMER_KEY,
                                                            CONSUMER_SECRET)

            if not os.path.isdir('out'):
                os.mkdir('out')

            write_token_file(TOKEN_FILE, oauth_token, oauth_token_secret)

        return twitter.Twitter(domain='api.twitter.com', api_version='1.1',
                               auth=twitter.oauth.OAuth(oauth_token, oauth_token_secret,
                                                        CONSUMER_KEY, CONSUMER_SECRET))
