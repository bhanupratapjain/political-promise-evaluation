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


def get_article_token(portion):
    article_token_dict = {}
    toker = nltk.RegexpTokenizer(r'\w+')
    with open("out/nyt_articles.json") as data_file:
        data = json.load(data_file)
    for article in data:
        text = article[portion]
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


def label_polarity(polarity):
    if -1.0 <= polarity <= -0.5:
        return "Negative response"
    elif -0.5 < polarity <= 0:
        return "Slightly Negative response"
    elif 0 <= polarity < 0.5:
        return "Slightly Positive Response"
    else:
        return "Positive Response"


def sentiment_analysis(text, nb=False):
    if nb:
        return TextBlob(text, analyzer=NaiveBayesAnalyzer()).sentiment
    return TextBlob(text).sentiment


def get_tfidf_matrix(articles, promises):
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    documents = []
    documents.extend(list(promises.values()))
    documents.extend(list(articles.values()))
    return tfidf.fit_transform(documents)


def get_article_promise_progress(articles, promises, tfidf_matrix, nb=False):
    progress = {}
    for i, promise in enumerate(promises):
        cosine_sim = cosine_similarity(tfidf_matrix[i: i + 1], tfidf_matrix)
        single_array = np.array(cosine_sim[0])
        article_array = single_array.argsort()[-6:][::-1]
        matched_articles = [s for s in article_array if s > 1]
        article_sentiment = defaultdict(lambda: [])
        for x in matched_articles:
            article_sentiment[list(articles.keys())[x - 2]].append(sentiment_analysis(
                articles[list(articles.keys())[x - 2]], nb))
        progress[promise] = article_sentiment
    return progress


if __name__ == "__main__":
    articles = get_article_token()
    promises = get_promise_token()
    tfidf_matrix = get_tfidf_matrix(articles, promises)
    results = {'article_nb': get_article_promise_progress(articles, promises, tfidf_matrix, nb=True),
               'article_pattern': get_article_promise_progress(articles, promises, tfidf_matrix, nb=False)}
    print(json.dumps(results))

    # match_articles(0)
    # print("Promise sentiment according to Google Search is:")
    # google_search(all_tokens[0])

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
