System Details : Ubuntu 16.04 64 bit, latest Spark 2.0 prebuilt with latest hadoop.
Used Pyspark and pandas

Steps Performed :

0] Combined "Title*weight + summary" to description. I haven't handled unicode chars.
1] Tokenized
2] Stopword Removal
3] CountVectorizer to get frequencies of terms.
4] Tf-idf
5] LDA to get topics
Click on the "clusters-generated.html" to visualize the topic clusters with terms and weights.
Set "noTopicsLDA" and "maxTermsperTopics" parameters to low values to get a nice image.



6] Distribution of document over topics : Topics terms are combined with frequencies to arrive at
   score which indicates how important that topic is to the document.

The word 'lidar' which appears > 900 times doesn't appear in any topics identified
because it is so common that tf-idf gives it low scores.


7] K-Means Clustering performed on the above distribution to get Topic clusters.


8] Documents classified according to the above clusters.
Doc-Clusters shows may documents belong to one cluster topic. Lidar related patents
are obviously many in the dataset.

9] I have used the topic distribution of documents as a feature vector for classification.

I have used LogisticRegressionWithLBFGS and SVMWithSGD classifier.
SVM performs better than Logistic, sometimes. By increasing "noTopicsLDA" and "maxTermsperTopics"
parameters, we get better classification.
The metrics measured are "Area under PR" and "Area under ROC".

Other features could have been added. like publicationDate, place of application filers, etc.
Rest don't appear to have significant effect.


Parameters :

There are few model tuning parameters in the beginning of the script.

    #wieght for the title : title is added to the summary 5 times
    titleWeight = 5

    #docfrequencies for CountVectorizer
    mindocFrequencies = 4.0

    #How many clusters for K-means clustering for topic distrubutions of documents

    noofKmeanClusters = 5

    #How many words in the vocabulary to take
    vocabSize = 600

    #How many topics to extra form the LDAModel
    noTopicsLDA = 60

    #How many terms per describing the Topic do we want
    maxTermsperTopics = 40






