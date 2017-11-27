import json
import string
from collections import defaultdict

import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from textblob.en.sentiments import NaiveBayesAnalyzer

from miners.GoogleMiner import GoogleMiner
from miners.NewsMiner import NewsMiner
from miners.TwitterMiner import TwitterMiner


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
        sa = TextBlob(text, analyzer=NaiveBayesAnalyzer()).sentiment
        return {"class": sa.classification, "p_pos": sa.p_pos, "p_neg": sa.p_neg}
    sa = TextBlob(text).sentiment
    sa_class = "pos" if sa.polarity > 0 else "neg"
    return {"class": sa_class, "polarity": sa.polarity, "subjectivity": sa.subjectivity}


def get_tfidf_matrix(articles, promises):
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    documents = []
    documents.extend(list(promises.values()))
    documents.extend(list(articles.values()))
    return tfidf.fit_transform(documents)


def get_article_promise_progress(articles, promises, tfidf_matrix, nb=False):
    progress = []
    for i, promise in enumerate(promises):
        cosine_sim = cosine_similarity(tfidf_matrix[i: i + 1], tfidf_matrix)
        single_array = np.array(cosine_sim[0])
        article_array = single_array.argsort()[-6:][::-1]
        matched_articles = [s for s in article_array if s > 1]
        article_sentiment = []
        for x in matched_articles:
            article_sentiment.append({
                "text": articles[list(articles.keys())[x - 2]],
                "sentiment": sentiment_analysis(articles[list(articles.keys())[x - 2]], nb)
            })
        progress.append({
            "promise": promise,
            "result": article_sentiment})
    return progress


def get_google_results(num=10):
    data = defaultdict(lambda: [])
    gm = GoogleMiner()
    for i, promise in enumerate(promises):
        for s in gm.get_search_summary(promise, num):
            text = s.text
            tokens = tokenize(text)
            data[promise].append(" ".join(remove_stop_words(tokens)))
    return data


def get_google_promise_progress(google_sum, nb):
    progress = []
    for promise, g_sum in google_sum.items():
        search_sentiment = []
        for s in g_sum:
            search_sentiment.append({
                "text": s,
                "sentiment": sentiment_analysis(s, nb)
            })
        progress.append({
            "promise": promise,
            "result": search_sentiment})
    return progress


if __name__ == "__main__":
    promises = get_promise_token()
    articles = get_article_token("text")
    articles_sum = get_article_token("summary")
    tfidf_matrix = get_tfidf_matrix(articles, promises)
    google_sum = get_google_results(10)

    results = {'article_text_nb': get_article_promise_progress(articles, promises, tfidf_matrix, nb=True),
               'article_text_pattern': get_article_promise_progress(articles, promises, tfidf_matrix, nb=False),
               'google_nb': get_google_promise_progress(google_sum, nb=True),
               'google_pattern': get_google_promise_progress(google_sum, nb=False)}

    tfidf_matrix_sum = get_tfidf_matrix(articles_sum, promises)
    results["article_summary_pattern"] = get_article_promise_progress(articles_sum, promises, tfidf_matrix_sum,
                                                                      nb=False)
    results["article_summary_nb"] = get_article_promise_progress(articles_sum, promises, tfidf_matrix_sum, nb=True)

    with open('out/results.json', 'w') as fout:
        json.dump(results, fout)

    # with open('out/results.json', 'r') as df:
    #     res = json.load(df)
    #
    # polarity = []
    # for r in res['article_summary_nb']:
    #     polarity.append(r['polarity'])
