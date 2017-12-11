# political-promise-evaluation
PPE provides a unified view of promises made by Political Candidates while campaign- ing and the promises fulfilled during their tenure. PPE aims to scour Political Manifesto’s and summarize political promises. We will extract relevant Tweets on Twitter and then evaluate the the fulfillment status of the promises


# Experiments

## Experiment 2
Naive Bayes with Train  (unbalanced)
http://scikit-learn.org/stable/modules/naive_bayes.html
```
    0 params - {'vect__ngram_range': (1, 1)}; mean - 0.82; std - 0.00
    1 params - {'vect__ngram_range': (1, 2)}; mean - 0.81; std - 0.00
                 precision    recall  f1-score   support
    
    No Progress       0.83      0.99      0.90     28942
       Progress       0.65      0.10      0.18      6677
    
    avg / total       0.79      0.82      0.76     35619
    
    Confusion Matrix
        
    [[28575   367]
     [ 5994   683]]
```
 
## Experiment 3
NB with actual test data

Total Articles = 624
Predicted Label 1 = 24
Predicted Label 0 = 600
Top KeyWords in Label 1 Predictions =  results/experiment3.txt
Top Keywords in Label 0 Predictions =  results/experiment3.txt


## Experiment 4 
NB with Google Search results
Identified all summaries to zero.
`results/experiment4.txt`
    

## Experiment 5
SVM with train (unbalanced)

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
             precision    recall  f1-score   support

No Progress       0.84      0.99      0.91     28788
   Progress       0.88      0.18      0.29      6831

avg / total       0.84      0.84      0.79     35619

[[28620   168]
 [ 5633  1198]]



# Experiment 6 
Random Forest with Train (unbalanced)

http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

             precision    recall  f1-score   support

No Progress       0.82      0.99      0.90     28947
   Progress       0.64      0.08      0.15      6672

avg / total       0.79      0.82      0.76     35619

Confusion Matrix
[[28627   320]
 [ 6111   561]]
 

# Report

## Data

### Promises
- Trump Tracker API 

### Promise Evaluation
- Twitter 
- NYT
    - search terms 'Donald Trump','mexico wall','obamacare'
    - date between  "20170120", "20171130"
    - Total articles = 624
- Google Search
-  All News Dataset: https://www.kaggle.com/snapcrack/all-the-news
    - Unlabelled, had to label.
        ```
        Total Test Data Size : 142473
        Counter({0: 115576, 1: 26897})
        Counter({'Breitbart': 23781, 'New York Post': 17493, 'NPR': 11992, 'CNN': 11488, 'Washington Post': 11114, 'Reuters': 10710, 'Guardian': 8681, 'New York Times': 7803, 'Atlantic': 7179, 'Business Insider': 6757, 'National Review': 6203, 'Talking Points Memo': 5214, 'Vox': 4947, 'Buzzfeed News': 4854, 'Fox News': 4354})
        ```     
  

## Methodology

### Document Matching 
- Bag of Words
- Tf-Idf

### Progress Tracking
- Naive Bayes (NTLK - Pre trained) (Pure Sentiment)
- Pattern (Lexical) (Pure Sentiment)
- Custom Trained Classifier (Vader Sentiment + Bag of words for Progressive Words )
    - Train Data Labelling
        - Label = 0 (Broken or in Progress)
        - Label = 1 (Completed)
        - Label = Binary(Binary(progress_coefficient) * polarity)
            - Polarity = vader polarity
            - progress_coefficient, 
                ```
                    progress_synonyms = syn_count / total_count
                    progress_antonyms = ant_count / total_count
                    progress_coefficient  = tanh(pp) - tanh(pn)
                ```
    - Naive Bayes -> Experiment 2
    - Random Forest (TODO)
    - SVM (TODO)
    