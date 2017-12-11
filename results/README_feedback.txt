### Work completed before project presentation

Before our project presentation, a lot of our time was spent in extraction and collation of data for the promises. Our
promise tracking data was being pulled from:
* Twitter
* New York Times
* Google Search

And we were implementing a Naive Bayes model trained on the movie review dataset. Because the classifier wasn't trained
for our specific needs so the results weren't as accurate as were hoping to achieve. The solution we reached at this
point was that we surely needed to create our own classifier as a first step.

Also, since our project focuses on a political situation so it was entirely possible that even a newspaper article
about the fulfillment of a promise can have an overall negative sentiment.

### Feedback received after presentation:

1) The sentiment classifier may not be enough, so need to discuss further plans or results.

#  This was similar to our intuition that sentiment classifier was just not good enough to give us the results we were
looking for and the Professors comments supported that.

    # Steps taken to rectify:
    -> We added another dataset to increase the size of our training data.
    -> We created a custom classifier from the newly extracted training data set of 147,000+ articles and labelled them
    as articles related to Broken (0 label) or Completed (1 label) promises.
    -> We created a sequence of equations to balance out the sentiment and give us useful data.
    -> Then we ran Naive Bayes, SVM and Random Forest on the processed data to achieve our results.

2) Also need to describe rules or patterns used in detail in the report.

# We added all the rules and patterns that we used during the experimentation and final development phase in detail in
the report.