import nltk
import json
import string


def promise_evaluation(m):
    with open("../out/results.json") as data_file:
        data = json.load(data_file)

    with open("../out/promises2.json") as repo_file:
        promise_data = json.load(repo_file)

    for i in m:
        for k in range(0, 2):
            methodology_used = i
            promise = data[i][k]['promise']
            predicted_promise_status = data[i][k]['result'][0]['sentiment']['class']
            actual_promise_status = promise_data[k]['promise_status']

            if predicted_promise_status == 'pos' and (actual_promise_status == 'In progress' or actual_promise_status == 'Achieved'):
                prediction_status = "Accurate Positive Prediction"
            elif predicted_promise_status == 'neg' and (actual_promise_status == 'Broken'):
                prediction_status = "Accurate Negative Prediction"
            else:
                prediction_status = "Inaccurate Prediction"

            print(methodology_used)
            print(promise)
            print(predicted_promise_status)
            print(actual_promise_status)
            print(prediction_status)


def get_promise_token():
    promise_token_dict = {}
    token = nltk.RegexpTokenizer(r'\w+')
    with open("../out/promises2.json") as data_file:
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


if __name__ == "__main__":
    promises = get_promise_token()
    methodologies = ['article_text_pattern',
                     'google_pattern',
                     'article_text_nb',
                     'article_summary_nb',
                     'google_nb',
                     'article_summary_pattern'
                     ]

    promise_evaluation(methodologies)
