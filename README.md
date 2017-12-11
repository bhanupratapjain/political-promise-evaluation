# Political Promise Evaluation
PPE provides a unified view of promises made by Political Candidates while campaigning and the promises fulfilled during their tenure. PPE aims to scour Political Manifesto’s and summarize political promises. We will extract relevant Tweets on Twitter and then evaluate the the fulfillment status of the promises

## Running Instructions
```
- Setup Python Version >= 3
- pip install -r requirements.txt
- Get data from https://www.kaggle.com/snapcrack/all-the-news. Extract under directory "data"
- python PPE.py
- Check the output on console and under dir "out" and "plots" (Exsiting data will be overwriten)
``` 
## Code Structure
### Data Miners
    - miners/GoogleMiner.py: Google Summary Miner
    - miners/NewsMiner.py : News Articles Miner
    - miners/TwitterMiner.py : Twitter Miner 
    - promise_extraction/extract.py : Promises
### Model Generation And Evaluation
    - PPE.py

## Experiments 
### 1) Lexicon Based
#### Lexical Pattern on Article Text

**Promise:** Build a wall. Trump's campaign began with a promise to build a wall across the United States' southern border and deport the country's 11 million undocumented immigrants.

    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
**Promise:** Repeal and Replace Obamacare Act

    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
**Promise:** Middle Class Tax Relief And Simplification Act

    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
#### Naive Bayes on Google Search Results
**Promise:** Build a wall. Trump's campaign began with a promise to build a wall across the United States' southern border and deport the country's 11 million undocumented immigrants.

    [1, 1, 1, 1, 0, 1, 1, 0, 0, 1]
    
**Promise:** Repeal and Replace Obamacare Act

    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1]
    
**Promise:** Middle Class Tax Relief And Simplification Act

    [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
    
#### Lexical Pattern on Google Search Results
**Promise:** Build a wall. Trump's campaign began with a promise to build a wall across the United States' southern border and deport the country's 11 million undocumented immigrants.

    [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    
**Promise:** Repeal and Replace Obamacare Act

    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    
**Promise:** Middle Class Tax Relief And Simplification Act

    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    
#### Naive Bayes on Article Summary
**Promise:** Build a wall. Trump's campaign began with a promise to build a wall across the United States' southern border and deport the country's 11 million undocumented immigrants.

    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
**Promise:** Repeal and Replace Obamacare Act

    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    
**Promise:** Middle Class Tax Relief And Simplification Act

    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    
#### Naive Bayes on Article Text
**Promise:** Build a wall. Trump's campaign began with a promise to build a wall across the United States' southern border and deport the country's 11 million undocumented immigrants.

    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
**Promise:** Repeal and Replace Obamacare Act

    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
**Promise:** Middle Class Tax Relief And Simplification Act

    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
#### Lexical Pattern on Article Summary

**Promise:** Build a wall. Trump's campaign began with a promise to build a wall across the United States' southern border and deport the country's 11 million undocumented immigrants.

    [1, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    
**Promise:** Repeal and Replace Obamacare Act

    [1, 1, 0, 1, 0, 0, 1, 1, 1, 1]
    
**Promise:** Middle Class Tax Relief And Simplification Act

    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]

### 2) Naive Bayes with Train
http://scikit-learn.org/stable/modules/naive_bayes.html=

    0 params - {'vect__ngram_range': (1, 1)}; mean - 0.82; std - 0.00
    1 params - {'vect__ngram_range': (1, 2)}; mean - 0.81; std - 0.00
                 precision    recall  f1-score   support
    
    No Progress       0.83      0.99      0.90     28942
       Progress       0.65      0.10      0.18      6677
    
    avg / total       0.79      0.82      0.76     35619
    
**Confusion Matrix**
        
    [[28575   367]
     [ 5994   683]]
 
### 3) NB with actual test data

    Total Articles = 624
    Predicted Label 1 = 24
    Predicted Label 0 = 600
    Top KeyWords in Label 1 Predictions =  results/experiment3.txt
    Top Keywords in Label 0 Predictions =  results/experiment3.txt


### 4) NB with Google Search results
Identified all summaries to zero.
`results/experiment4.txt`
    

### 5) SVM with train

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
             precision    recall  f1-score   support

    No Progress       0.84      0.99      0.91     28788
    Progress          0.88      0.18      0.29      6831
    avg / total       0.84      0.84      0.79     35619
    
**Confusion Matrix:**

    [[28620   168]
    [ 5633  1198]]



### 6) Random Forest with Train

http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

                    precision    recall  f1-score   support

    No Progress       0.82      0.99      0.90     28947
    Progress          0.64      0.08      0.15      6672
    avg / total       0.79      0.82      0.76     35619

**Confusion Matrix**

    [[28627   320]
    [ 6111   561]]
 
### 7) Classifying matched articles to promises using Naive Bayes

**Promise:** Build a wall. Trump's campaign began with a promise to build a wall across the United States' southern border and deport the country's 11 million undocumented immigrants.

    [0 0 0 0 0 0 0 0 0 0]
    
**Promise:** Repeal and Replace Obamacare Act

    [0 0 0 0 0 0 0 0 1 0]
    
**Promise:** Middle Class Tax Relief And Simplification Act

    [1 1 1 0 1 1 0 0 0 1]