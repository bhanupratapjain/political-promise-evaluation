import csv
import glob
import itertools
import json
import operator
import pprint
import random
import string
import sys
from collections import defaultdict, Counter

import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from nltk.corpus import stopwords, wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from pygments.util import xrange
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from textblob.en.sentiments import NaiveBayesAnalyzer
from wordcloud import WordCloud

from miners.GoogleMiner import GoogleMiner
from miners.NewsMiner import NewsMiner
from miners.TwitterMiner import TwitterMiner

plt.style.use('ggplot')


def get_tweets():
    return TwitterMiner().login().search.tweets(q="#trump", count=100)


def get_articles():
    nm = NewsMiner()
    articles_dict = {}
    articles = nm.get_articles("Donald Trump", "20170120", "20171130")
    articles.extend(nm.get_articles("mexico wall", "20170120", "20171130"))
    articles.extend(nm.get_articles("obamacare", "20170120", "20171130"))
    for a in articles:
        articles_dict[a['_id']] = a
    nm.get_text(articles_dict)
    with open('out/nyt_articles_latest.json', 'w') as fout:
        json.dump(articles_dict, fout)
    print(articles)
    # return articles

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
    with open("out/nyt_articles_latest.json") as data_file:
        data = json.load(data_file)
    for article in data:
        text = article[portion]
        lowers = text.lower()
        no_punctuation = lowers.translate(str.maketrans("", "", string.punctuation))
        tokens = toker.tokenize(no_punctuation)
        article_token_dict[article['_id']] = " ".join(tokens)
    return article_token_dict


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


def get_article_token(portion):
    article_token_dict = {}
    toker = nltk.RegexpTokenizer(r'\w+')
    with open("out/nyt_articles_latest.json") as data_file:
        data = json.load(data_file)
    for id, article in data.items():
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
    sa_class = "pos" if sa.polarity >= 0 else "neg"
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


def save_google_results(promises, num=10):
    data = defaultdict(lambda: [])
    gm = GoogleMiner()
    for i, promise in enumerate(promises):
        for s in gm.get_search_summary(promise, num):
            text = s.text
            tokens = tokenize(text)
            data[promise].append(" ".join(remove_stop_words(tokens)))
    with open('out/google_test_data.json', 'w') as fout:
        json.dump(data, fout)
    return data


def get_google_results(promises, num=10):
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


def experiment_1():
    promises = get_promise_token()
    articles = get_article_token("text")
    articles_sum = get_article_token("summary")
    tfidf_matrix = get_tfidf_matrix(articles, promises)
    google_sum = get_google_results(promises, 10)
    results = {'article_text_nb': get_article_promise_progress(articles, promises, tfidf_matrix, nb=True),
               'article_text_pattern': get_article_promise_progress(articles, promises, tfidf_matrix, nb=False),
               'google_nb': get_google_promise_progress(google_sum, nb=True),
               'google_pattern': get_google_promise_progress(google_sum, nb=False)}
    tfidf_matrix_sum = get_tfidf_matrix(articles_sum, promises)
    results["article_summary_pattern"] = get_article_promise_progress(articles_sum, promises, tfidf_matrix_sum,
                                                                      nb=False)
    results["article_summary_nb"] = get_article_promise_progress(articles_sum, promises, tfidf_matrix_sum, nb=True)
    with open('out/results.json', 'w') as fout:
        json.dump(results, fout, indent=4)


def get_combined_articles():
    articles = []
    csv.field_size_limit(sys.maxsize)
    toker = nltk.RegexpTokenizer(r'\w+')
    for filename in glob.glob("data/all-the-news/*.csv"):
        with open(filename) as data_file:
            news_reader = csv.reader(data_file, delimiter=',', quotechar='"')
            next(news_reader, None)
            for row in news_reader:
                text = row[9]
                lowers = text.lower()
                no_punctuation = lowers.translate(str.maketrans("", "", string.punctuation))
                tokens = toker.tokenize(no_punctuation)
                articles.append({
                    "id": row[1],
                    "publication": row[3],
                    "text": " ".join(tokens)
                })
    return articles


def save_articles_with_sentiment():
    articles = get_combined_articles()
    sid = SentimentIntensityAnalyzer()
    for i, article in enumerate(articles):
        print(i)
        ss = sid.polarity_scores(article['text'])
        article["polarity"] = ss
    with open('out/articles_train_data_with_sentiment.json', 'w') as fout:
        json.dump(articles, fout)


def get_progress_words():
    progressive_words = ['progress', 'advance', 'breakthrough', 'development', 'evolution', 'growth', 'headway',
                         'improvement', 'increase', 'momentum', 'movement', 'pace', 'process', 'rise', 'stride',
                         'mature', 'change', 'enhancement', 'gain', 'upgrade', 'rise']
    synonyms = set()
    antonyms = set()
    for w in progressive_words:
        for syn in wordnet.synsets(w):
            for l in syn.lemmas():
                if "_" not in l.name():
                    synonyms.add(l.name())
                if l.antonyms():
                    antonyms.add(l.antonyms()[0].name())
    return synonyms, antonyms


def get_progressive_counts(text, progress_syn, progress_ant):
    tokens = Counter(text.split())
    syn_count = 0
    ant_count = 0
    total_count = 0
    for token, count in tokens.items():
        total_count += count
        if token in progress_syn:
            syn_count += count
        if token in progress_ant:
            ant_count += count
    return syn_count, ant_count, total_count


def tanh(x, derivative=False):
    if derivative:
        return 1 - (x ** 2)
    return np.tanh(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_articles_train_data():
    progress_syn, progress_ant = get_progress_words()
    print(progress_syn, progress_ant)
    pos_label = 0
    neg_label = 0
    with open('out/articles_train_data_with_sentiment.json', 'r') as fin:
        data = json.load(fin)
        for i, rec in enumerate(data):
            syn_count, ant_count, total_count = get_progressive_counts(rec['text'], progress_syn, progress_ant)
            if total_count <= 0:
                continue
            polarity = rec['polarity']['compound']
            # print(rec['text'])
            pp = syn_count / total_count
            pn = ant_count / total_count
            ppn = tanh(pp) - tanh(pn)
            progress = 1 if ppn >= 0.01 else 0
            label = 1 if polarity * progress > 0 else 0
            if label == 1:
                pos_label += 1
            else:
                neg_label += 1
            # print(
            #     "row=%s,polarity=%s,syn_count=%s, ant_count=%s, total_count=%s, pp=%s, pn=%s, ppn=%s, progress=%s label=%s" % (i,polarity, syn_count, ant_count, total_count, pp, pn, ppn, progress, label))
            rec['label'] = label
            rec['progress'] = progress
            rec['pos_progress'] = pp
            rec['neg_progress'] = pn
            rec['norm_progress'] = ppn
        with open('out/articles_train_data.json', 'w') as fout:
            json.dump(data, fout)
        print("pos_label=%s; neg_label=%s" % (pos_label, neg_label))


def get_train_articles(balanced=False):
    articles = []
    labels = []
    with open('out/articles_train_data.json', 'r') as fin:
        data = json.load(fin)
        for i, rec in enumerate(data):
            if 'text' in rec and 'label' in rec:
                articles.append(rec['text'])
                labels.append(rec['label'])
    return articles, labels


def nb_train_experiment_2():
    # nltk.download('vader_lexicon')
    articles, labels = get_train_articles()

    docs_train, docs_test, y_train, y_test = train_test_split(
        articles, labels, test_size=0.25, random_state=None)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', MultinomialNB()),
    ])

    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(docs_train, y_train)

    # TASK: print the mean and std for each candidate along with the parameter
    # settings for all the candidates explored by grid search.
    n_candidates = len(grid_search.cv_results_['params'])
    for i in range(n_candidates):
        print(i, 'params - %s; mean - %0.2f; std - %0.2f'
              % (grid_search.cv_results_['params'][i],
                 grid_search.cv_results_['mean_test_score'][i],
                 grid_search.cv_results_['std_test_score'][i]))

    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = grid_search.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=["No Progress", "Progress"]))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)


def svm_train_experiment_5():
    articles, labels = get_train_articles()
    docs_train, docs_test, y_train, y_test = train_test_split(
        articles, labels, test_size=0.25, random_state=None)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', SGDClassifier())
    ])
    clf = pipeline.fit(docs_train, y_train)
    y_predicted = clf.predict(docs_test)
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=["No Progress", "Progress"]))
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)


def rf_train_experiment_7():
    articles, labels = get_train_articles()
    docs_train, docs_test, y_train, y_test = train_test_split(
        articles, labels, test_size=0.25, random_state=None)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', RandomForestClassifier())
    ])
    clf = pipeline.fit(docs_train, y_train)
    y_predicted = clf.predict(docs_test)
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=["No Progress", "Progress"]))
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)


def get_test_articles(promises):
    articles =defaultdict(lambda :[])
    test_articles = get_article_token("text")
    tfidf_matrix = get_tfidf_matrix(test_articles, promises)
    for i, promise in enumerate(promises):
        cosine_sim = cosine_similarity(tfidf_matrix[i: i + 1], tfidf_matrix)
        single_array = np.array(cosine_sim[0])
        article_array = single_array.argsort()[-11:][::-1]
        matched_articles = [s for s in article_array if s > 2]
        for x in matched_articles:
            articles[promise].append(test_articles[list(test_articles.keys())[x - 3]])
    return articles


def nb_test_experiment_3():
    promises = get_promise_token()
    label_1_keywords = []
    label_0_keywords = []
    # test_articles = get_test_articles(promises)
    articles = get_article_token("text")
    test_articles = list(articles.values())
    train_articles, train_labels = get_train_articles()
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', MultinomialNB()),
    ])
    pipeline.fit(train_articles, train_labels)
    y_predicted = pipeline.predict(test_articles)
    with open("out/nyt_articles_latest.json") as data_file:
        a_data = json.load(data_file)
        for i, a_id in enumerate(list(articles.keys())):
            if y_predicted[i] == 0:
                label_0_keywords.extend(a_data[a_id]['keywords_1'])
            else:
                label_1_keywords.extend(a_data[a_id]['keywords_1'])
            print("Summary: ", a_data[a_id]['summary'])
            print("Keywords: ", a_data[a_id]['keywords_1'])
            print("Prediction: ", y_predicted[i])
            print("\n")
    plt.figure(figsize=(15, 8))
    wordcloud_0 = WordCloud(width=1000, height=500, background_color="white").generate(' '.join(label_0_keywords))
    wordcloud_1 = WordCloud(width=1000, height=500, background_color="white").generate(' '.join(label_1_keywords))

    # plt.subplot(121)
    plt.imshow(wordcloud_0)
    plt.axis("off")
    plt.savefig('plots/experiment-3-label_0.png')

    # plt.subplot(122)
    plt.imshow(wordcloud_1)
    plt.axis("off")
    plt.savefig('plots/experiment-3-label_1.png')
    # plt.show()

    print("Label 0 Keywords Distribution")
    pprint.pprint(nltk.FreqDist(label_0_keywords))
    print(nltk.FreqDist(label_0_keywords).most_common(10))
    print("Label 1 Keywords Distribution")
    pprint.pprint(Counter(label_1_keywords))
    print(nltk.FreqDist(label_1_keywords))
    print(nltk.FreqDist(label_1_keywords).most_common(10))


def nb_matched_test_experiment_8():
    promises = get_promise_token()
    test_articles = get_test_articles(promises)
    train_articles, train_labels = get_train_articles()
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', MultinomialNB()),
    ])
    pipeline.fit(train_articles, train_labels)
    for promise,articles in test_articles.items():
        print(promise)
        print(pipeline.predict(articles))

def svm_test_experiment_6():
    promises = get_promise_token()
    label_1_keywords = []
    label_0_keywords = []
    # test_articles = get_test_articles(promises)
    articles = get_article_token("text")
    test_articles = list(articles.values())
    train_articles, train_labels = get_train_articles()
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', MultinomialNB()),
    ])
    pipeline.fit(train_articles, train_labels)
    y_predicted = pipeline.predict(test_articles)
    with open("out/nyt_articles_latest.json") as data_file:
        a_data = json.load(data_file)
        for i, a_id in enumerate(list(articles.keys())):
            if y_predicted[i] == 0:
                label_0_keywords.extend(a_data[a_id]['keywords_1'])
            else:
                label_1_keywords.extend(a_data[a_id]['keywords_1'])
            print("Summary: ", a_data[a_id]['summary'])
            print("Keywords: ", a_data[a_id]['keywords_1'])
            print("Prediction: ", y_predicted[i])
            print("\n")
    plt.figure(figsize=(15, 8))
    wordcloud_0 = WordCloud(width=1000, height=500, background_color="white").generate(' '.join(label_0_keywords))
    wordcloud_1 = WordCloud(width=1000, height=500, background_color="white").generate(' '.join(label_1_keywords))

    # plt.subplot(121)
    plt.imshow(wordcloud_0)
    plt.axis("off")
    plt.savefig('plots/experiment-3-label_0.png')

    # plt.subplot(122)
    plt.imshow(wordcloud_1)
    plt.axis("off")
    plt.savefig('plots/experiment-3-label_1.png')
    # plt.show()

    print("Label 0 Keywords Distribution")
    pprint.pprint(nltk.FreqDist(label_0_keywords))
    print(nltk.FreqDist(label_0_keywords).most_common(10))
    print("Label 1 Keywords Distribution")
    pprint.pprint(Counter(label_1_keywords))
    print(nltk.FreqDist(label_1_keywords))
    print(nltk.FreqDist(label_1_keywords).most_common(10))


def experiment_4():
    promises = get_promise_token()
    test_data = list(itertools.chain.from_iterable(get_google_results(promises, 10).values()))
    train_articles, train_labels = get_train_articles()
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', MultinomialNB()),
    ])
    pipeline.fit(train_articles, train_labels)
    y_predicted = pipeline.predict(test_data)
    for i, a in enumerate(test_data):
        print(a)
        print(y_predicted[i])


def get_colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def set_facecolor(rects):
    colors = get_colors()
    ncolor = len(colors)
    for index, rect in enumerate(rects):
        color = colors[index % ncolor]
        rect.set_facecolor(color)


def experiment_train_data_distribution():
    train_data, train_labels = get_train_articles()
    label_counts = Counter(train_labels)
    publications = []
    with open('out/articles_train_data.json', 'r') as fin:
        data = json.load(fin)
        for i, rec in enumerate(data):
            publications.append(rec['publication'])
    pub_count = sorted(Counter(publications).items(), key=lambda x: x[1], reverse=True)

    pub_name, pub_val = [], []
    for p in pub_count:
        pub_name.append(p[0])
        pub_val.append(p[1])

    print("Total Test Data Size :", len(train_labels))
    print(label_counts)
    print(pub_count)

    # Plot Label Distribution
    plt.bar(["0", "1"], [label_counts[0], label_counts[1]], color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/train-label-distribution.png")
    plt.close()

    # Plot Publication Distribution
    pub_x = np.arange(len(pub_name))
    print(pub_x)
    set_facecolor(plt.bar(pub_x, pub_val, color=plt.rcParams['axes.prop_cycle'].by_key()['color']))
    plt.xlabel("Publications")
    plt.xticks(pub_x, pub_name, rotation='vertical')
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/train-publication-distribution.png")
    plt.close()

    # Plot Test Data Distribution
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer())
    ])
    random_samples = sorted(random.sample(xrange(len(train_data)), 1000))
    data = pipeline.fit_transform([train_data[i] for i in random_samples]).todense()
    pca = PCA(n_components=3).fit(data)
    X = pca.transform(data)
    fig = pyplot.figure()
    ax = Axes3D(fig)
    # ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    # pyplot.show()

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=['tomato' if train_labels[i] == 1 else 'teal' for i in random_samples])
    # plt.legend([a.collections[0], b.collections[0]], ["Label 0", "Label 1"],
    #            loc="upper right")
    scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c='tomato',marker = 'o')
    scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", c='teal',marker = 'o')
    ax.legend([scatter1_proxy, scatter2_proxy], ['label_1', 'label_0'], numpoints=1)
    plt.savefig("plots/train-data-distribution.png")
    plt.close()


if __name__ == "__main__":
    experiment_1()
    # generate_articles_test_data()
    # experiment_train_data_distribution()
    # nb_train_experiment_2()
    # nb_test_experiment_3()
    # experiment_4()
    # svm_train_experiment_5()
    # svm_test_experiment_6()
    # rf_train_experiment_7()
    # nb_matched_test_experiment_8()
