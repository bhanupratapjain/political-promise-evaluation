# political-promise-evaluation
PPE provides a unified view of promises made by Political Candidates while campaign- ing and the promises fulfilled during their tenure. PPE aims to scour Political Manifesto’s and summarize political promises. We will extract relevant Tweets on Twitter and then evaluate the the fulfillment status of the promises


# Experiments

## Experiment 2

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
Total Articles = 
Predicted Label 1 = 24
Predicted Label 0 = 600

Top KeyWords in Label 1 Predictions =  
Top Keywords in Label 0 Predictions = 


## Experiment 4

Identified all summaries to zero.

`results/experiment4.txt`

# Report


## Data

### Promises
- Trump Tracker API 

### Promise Evaluation
- Twiiter 
- NYT
- Google Search

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
    
    
    
## Getting more articles 
- search terms 'Donald Trump','mexico wall','obamacare'
- date between  "20170120", "20171130"
- Total articles = 624
