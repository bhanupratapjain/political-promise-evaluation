import json
import string

import nltk
import requests
import operator
import numpy as np
import numpy.linalg as LA

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from miners.GoogleMiner import GoogleMiner
from miners.NewsMiner import NewsMiner
from miners.TwitterMiner import TwitterMiner

from textblob import TextBlob
from bs4 import BeautifulSoup


def get_tweets():
    return TwitterMiner().login().search.tweets(q="#trump", count=100)


def get_articles():
    nm = NewsMiner()
    articles = nm.get_articles("Donald Trump", "20170120", "20170830")
    nm.get_text(articles)
    with open('out/nyt_articles.json', 'w') as fout:
        json.dump(articles, fout)
    print(articles)

    # cnn = NewsMiner().sources['cnn']
    # for i in range(10):
    #     print (cnn.size())


def get_tokens():
    corpus = ""
    with open("out/nyt_articles.json") as data_file:
        data = json.load(data_file)
    for article in data:
        corpus += article['text']
    lowers = corpus.lower()
    no_punctuation = lowers.translate(str.maketrans("", "", string.punctuation))
    toker = nltk.RegexpTokenizer(r'\w+')
    tokens = toker.tokenize(no_punctuation)
    count = nltk.Counter(tokens)
    print(count.most_common(10))
    return tokens


def remove_stop_words(tokens):
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    count = nltk.Counter(filtered)
    print(count.most_common(10))
    return filtered


def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        # stemmed.append(nltk.PorterStemmer().stem(item))
        stemmed.append(nltk.WordNetLemmatizer().lemmatize(item))
    count = nltk.Counter(stemmed)
    # print(count.most_common(10))
    return stemmed


def get_article_token():
    article_token_dict = {}
    toker = nltk.RegexpTokenizer(r'\w+')
    with open("out/nyt_articles.json") as data_file:
        data = json.load(data_file)
    for article in data:
        text = article['text']
        lowers = text.lower()
        no_punctuation = lowers.translate(str.maketrans("", "", string.punctuation))
        tokens = toker.tokenize(no_punctuation)
        article_token_dict[article['_id']] = " ".join(tokens)
    return article_token_dict


def tokenize(text):
    toker = nltk.RegexpTokenizer(r'\w+')
    lowers = text.lower()
    no_punctuation = lowers.translate(str.maketrans("", "", string.punctuation))
    tokens = toker.tokenize(no_punctuation)
    stems = stem_tokens(tokens)
    return stems


def get_promise_token():
    promise_token_dict = {}
    token = nltk.RegexpTokenizer(r'\w+')
    with open("out/promises2.json") as data_file:
        data = json.load(data_file)
    for promise in data:
        if promise['promise_description']:
            text = promise['promise_description']
        else:
            text = promise['promise_title']
        lowers = text.lower()
        no_punctuation = lowers.translate(str.maketrans("", "", string.punctuation))
        tokens = token.tokenize(no_punctuation)
        promise_token_dict[promise['promise_title']] = " ".join(tokens)
    return promise_token_dict


def google_search(search_query):
    search_sentiment_result = []
    gm = GoogleMiner()
    for s in gm.get_search_summary(search_query):
        search_sentiment_result.append(sentiment_analysis(s.text))
    my_list = {i: search_sentiment_result.count(i) for i in search_sentiment_result}
    print(max(my_list.items(), key=operator.itemgetter(1))[0])


def sentiment_analysis(c):
    blob = TextBlob(c)
    for sentence in blob.sentences:
        polarity = sentence.sentiment.polarity
        if -1.0 <= polarity <= -0.5:
            return "Negative response"
        elif -0.5 < polarity <= 0:
            return "Slightly Negative response"
        elif 0 <= polarity < 0.5:
            return "Slightly Positive Response"
        else:
            return "Positive Response"


def match_articles(promise):
    cosine_sim = cosine_similarity(tfidf_matrix[promise: promise + 1], tfidf_matrix)
    single_array = np.array(cosine_sim[0])
    article_array = single_array.argsort()[-6:][::-1]
    only_articles = [s for s in article_array if s > 1]
    promise_statement = all_tokens[promise]
    print("The promise made by the candidate was: \n", promise_statement)
    count = 1
    for x in only_articles:
        print(count, all_tokens[x] + "\n")
        print(sentiment_analysis(all_tokens[x]))
        count += 1


if __name__ == "__main__":
    # pprint.pprint(get_tweets())
    # get_articles()
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    # stemmed_tokens = stem_tokens(remove_stop_words(get_tokens()))
    # print(GoogleMiner().get_search_summary("Donald Trump"))
    # exit(0)
    #
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    article_tokens = get_article_token()
    promise_tokens = get_promise_token()
    all_tokens = []
    all_tokens.extend(list(promise_tokens.values()))
    all_tokens.extend(list(article_tokens.values()))
    # print(all_tokens, '\n')
    # print(list(promise_tokens.values()))
    # train = tfidf.fit_transform(list(promise_tokens.values()))
    # test = tfidf.transform(list(promise_tokens.values()))
    # print(train.toarray()[0].tolist())
    tfidf_matrix = tfidf.fit_transform(all_tokens)
    match_articles(0)
    print("Promise sentiment according to Google Search is:")
    google_search(all_tokens[0])

    match_articles(1)
    print("Promise sentiment according to Google Search is:")
    google_search(all_tokens[1])
    # print(tfidf_matrix)

    # feature_names = tfidf.get_feature_names()
    # for col in test.nonzero()[1]:
    #     print (feature_names[col], ' - ', test[0, col])

    # skl_tfidf_comparisons = []
    # for count_0, doc_0 in enumerate(tfidf_matrix.toarray()):
    #     for count_1, doc_1 in enumerate(tfidf_matrix.toarray()):
    #         if count_0==count_1:
    #             continue
    #         skl_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

    # for count_0, doc_0 in enumerate(tfidf_matrix.toarray()):
    #     skl_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))
    # print(linear_kernel(tfidf_matrix[0: 1], tfidf_matrix).flatten())
    # print(linear_kernel(tfidf_matrix[1: 2], tfidf_matrix).flatten())
    # print(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix))
    #
    # print(cosine_similarity(tfidf_matrix[1:2], tfidf_matrix))
    #
    # feature_names = tfidf.get_feature_names()
    #
    # print(feature_names)
