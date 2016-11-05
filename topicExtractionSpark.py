from __future__ import print_function

import sys
from operator import add

import operator

from operator import itemgetter
import json
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import CountVectorizerModel

from pyspark import SparkContext
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors, Vector
from pyspark.mllib.feature import IDF
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import SparseVector, VectorUDT, DenseVector
from pyspark.ml.feature import Tokenizer, RegexTokenizer
import re
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import broadcast
import pandas as pd
from pyspark.mllib.clustering import KMeans, KMeansModel
import random

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, SVMWithSGD
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.regression import LabeledPoint

if __name__ == "__main__":

    spark = SparkSession.builder.appName("PythonTopicModelling").getOrCreate()
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)
    jsonfile = open("uspto.json")
    lines = jsonfile.readlines()

    #wieght for the title : title is added to the summary 5 times
    titleWeight = 5

    #docfrequencies for CountVectorizer
    mindocFrequencies = 4.0

    #How many clusters for K-means clustering for topic distrubutions of documents

    noofKmeanClusters = 5

    #How many words in the vocabulary to take
    vocabSize = 600

    #How many topics to extra form the LDAModel
    noTopicsLDA = 50

    #How many terms per describing the Topic do we want
    maxTermsperTopics = 30

    count = 0
    docId = 0
    textcorpus = []
    labelstatus = {}

    impCount = 0

    for line in lines :
        obj = json.loads(line.strip())

        """
        if count > 100 :
            break

        count +=1
        """

        origtitle = obj["object"]["title"]
        title = ""
        for r in range(titleWeight):
            title += origtitle

        #remove words upto size 4 from the summary through regular expression

        strval = ( title + " " +obj["object"]["summary"]).encode('ascii', 'ignore')

        #remove all characters that are not alphanumeric and underscore
        strval = re.sub(r'[^\w]', ' ', strval)

        #remove all words with length <=4
        strval = re.sub(r'\b\w{1,4}\b', '', strval)

        #  collapse all white spaces to single space
        strval = re.sub("\s\s+", " ", strval)
        status = 0

        if (obj["object"]["status"] == u'Patented Case'):

            status = 1


        tempval = (docId, strval, status)
        labelstatus[docId] = status

        textcorpus.append(tempval)

        docId += 1




    sentenceDataFrame = spark.createDataFrame(textcorpus, ["label", "sentence","status"])



    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsDataFrame = tokenizer.transform(sentenceDataFrame)
    for words_label in wordsDataFrame.select("words", "label").take(3):
        #print(words_label)
        pass

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    wordsDataFrame = remover.transform(wordsDataFrame)

    for words_label in wordsDataFrame.select("filtered", "label").take(3):
        #print(words_label)
        pass


    cv = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=vocabSize, minDF=mindocFrequencies)

    cvmodel = cv.fit(wordsDataFrame)
    result = cvmodel.transform(wordsDataFrame).select("label","features")
    vocablist = (cvmodel.vocabulary)
    #print (vocablist)
    #result.select("features").show(truncate=False)



    countVectors = result.select("features")



    frequencyVectors = countVectors.rdd.map(lambda vector: DenseVector(vector[0].toArray()))


    idf = IDF().fit(frequencyVectors)
    tfidf = idf.transform(frequencyVectors)

    frequencyVectors = frequencyVectors.zipWithIndex()

    #resultMod = result.rdd.map(lambda vector: DenseVector(vector[1].toArray()))

    # resultMod = resultMod.toDF()

    resultMod = sqlContext.createDataFrame(frequencyVectors, ["features","documentId"])

    # prepare corpus for LDA
    corpus = tfidf.map(lambda x: [1, x]).cache()




    # train LDA
    # optimizer parameter "em" or "online"
    ldaModel = LDA.train(corpus, k=noTopicsLDA, maxIterations=40, optimizer="online", docConcentration=-1.0,
                         topicConcentration=-1.0)


    topicIndices = ldaModel.describeTopics(maxTermsPerTopic=maxTermsperTopics)




    topicsRDD = sc.parallelize(topicIndices)

    """
    for row in topicsRDD.collect():
        # print (Vectors.sparse(row[0][0],row[0][1],row[0][2]) )
        print(row)
        break

    exit()
    """

    # zipping the word with its probability distribution for that topic
    termsRDD = topicsRDD.map(lambda topic: (zip(itemgetter(*topic[0])(vocablist), topic[1])))

    zippedRDD = topicsRDD.map(lambda topic: (zip(topic[0], topic[1])))

    # for Every topic, sparse vector of distribution over words
    docCalcs = zippedRDD.map(lambda topic:  DenseVector((SparseVector(vocabSize,topic)).toArray()))

    #schema = StructType([StructField("topicwordDistribution", Vector(), False)])

    #df = sqlContext.applySchema(docCalcs, schema)

    #docCalcs = docCalcs.map(lambda l: Row(l))

    docCalcs = docCalcs.zipWithIndex()

    #docCalcs = docCalcs.collect()

    docCalcs = sqlContext.createDataFrame(docCalcs, ['topicwordDistribution','topicId'])



    #print(type(docCalcs))

    for words_label in docCalcs.select('topicwordDistribution','topicId').take(3):
        #print(words_label)
        pass





    finalDF = resultMod.join(broadcast(docCalcs)) #.select("label", "topicId", (vector_udf(result.features, docCalcs.topicwordDistribution)).alias("score"))

    #finalDF.withColumn("score", (df("score")))

    #finalDF = finalDF.map(lambda x: computeDot(x[0], x[1][1]).cache())

    #finalDF = finalDF.withColumn()


    for words_label in finalDF.take(3):
        #print(words_label)
        pass

    #finalDF.show(truncate=False)

    pandaFrame = finalDF.toPandas()

    #print (pandaFrame)
    #pandaFrame["Vale"] = pandaFrame.features.to_frame()

    #print (pandaFrame.features)
    #print ("-------------------")
    #print (pandaFrame.topicwordDistribution)

    #print (pandaFrame.features  .dot(pandaFrame.topicwordDistribution))

    ans = [row.features.dot(row.topicwordDistribution) for idx, row in pandaFrame.iterrows()]

    pandaFrame["score"] = pd.DataFrame(ans)

    #print (ans)

    #print (pandaFrame)

    #pandaFrame['score'] = (pandaFrame.features).dot(pandaFrame.topicwordDistribution)

    #print(pandaFrame)
    pivotTable = pandaFrame.pivot(index='documentId', columns='topicId', values='score')



    #print (pivotTable)

    #scoreDist = pivotTable.apply(lambda x: x.tolist(), axis=0)

    scoreDist = pivotTable.values.tolist()





    scoreDistRDD = sc.parallelize(scoreDist)



    """
    for row in scoreDist.collect():
        print(row)

    exit()
    """

    #spDF = sqlContext.createDataFrame(pivotTable)
    #spDF.show(truncate=False)



    clusters = KMeans.train(scoreDistRDD, noofKmeanClusters, maxIterations=20,
                             initializationMode="random")




    indexedData = scoreDistRDD.zipWithIndex()

    """
    for row in indexedData.collect():
        # print (Vectors.sparse(row[0][0],row[0][1],row[0][2]) )
        #print(row)
        pass
    """

    indexedData = indexedData.map(lambda x: ( clusters.predict(x[0]) , [ x[1] ]))

    #print(clusters.centers)

    classDF = sqlContext.createDataFrame(indexedData, ['class', 'documentId'])



    # = indexedData.toDF("dataPoint", "class" )

    #classDF.show(truncate=False)

    classDF = classDF.rdd.reduceByKey(lambda a,b : a+b)

    classDocList = classDF.toDF()

    #classDocList.show()

    #lenclus = len(clusters.centers)
    #for row in range(classDF) :

    indexedTermsRDD = termsRDD.zipWithIndex()

    termsRDD = indexedTermsRDD.flatMap(lambda term: [(t[0], t[1], term[1]) for t in term[0]])
    termDF = termsRDD.toDF(['term', 'probability', 'topicId'])

    topicDescription = termDF.toPandas().values.tolist()

    #print (topicDescription)

    topicTerms = {}
    for item in topicDescription :
        if item[2] in topicTerms.keys()  :
            topicTerms[item[2]] = topicTerms[item[2]]+[item[0]]
        else :
            topicTerms[item[2]] = [item[0]]

    #print (topicTerms)



    #exit()

    #topics = ldaModel.topicsMatrix()
    print("\nTerms describing the topics modelled by LDA after performing Tf-Idf")
    for key, value in topicTerms.items():
        print("Topic " , key , ": " , value)




    print ("\nTopics Cluster Descriptions : Sorted by Importance of topics for that cluster" )

    for clus in range(len(clusters.centers)):
        enuclus = dict(enumerate(clusters.centers[clus]))
        sorted_enuclus = sorted(enuclus.items(), key=operator.itemgetter(1))
        print ( "Topic Cluster %s : "%( str(clus)), ["Topic-"+ str(i[0]) for i in sorted_enuclus])


    pfclassDocList = classDocList.toPandas()

    pfclassDocList = pfclassDocList.values.tolist()

    print("\nDocument classified based on the above identified Topic Clusters : ")
    for item in pfclassDocList :
        print ("Class-" , item[0] , " : Doc List " , item[1] )




    #termDF.show(truncate=False)

    rawJson = termDF.toJSON(use_unicode=False).collect()


    pyObj = eval(rawJson.__repr__())

    childList = []
    for obj in pyObj:
        #print (obj)
        childList.append(json.loads(obj))

    childDict = {"name": "topics", "children":childList}
    topDict = {"name":"data", "children":[childDict]}

    json_data = json.dumps(topDict)



    with open("clusters.html", "r") as input_file:

        filetext = input_file.read()
        filetext = filetext.replace("var json=;", "var json = %s ;"%json_data)

        with open("clusters-generated.html", "w+") as output_file:
            output_file.write(filetext)






    # Patended Case is 144 roughly half of 358
    trainsample = random.sample(range(0, docId), int(docId * 0.8))

    trainlistLabeledPoints = []
    testlistLabeledPoints = []



    # Very Important Assumption is here


    index = 0

    for scores in scoreDist:

        if (index in trainsample):
            trainlistLabeledPoints.append(LabeledPoint(labelstatus[index], scores))
        else:
            testlistLabeledPoints.append(LabeledPoint(labelstatus[index], scores))

        index += 1




    #print(len(trainlistLabeledPoints))
    #print(trainlistLabeledPoints)

    #print(len(testlistLabeledPoints))
    #print(testlistLabeledPoints)




    train_data = sc.parallelize(trainlistLabeledPoints)
    test_data = sc.parallelize(testlistLabeledPoints)


    # Run training algorithm to build the model
    modelLogistic = LogisticRegressionWithLBFGS.train(train_data)


    # Compute raw scores on the test set
    predictionAndLabels = test_data.map(lambda lp: (float(modelLogistic.predict(lp.features)), lp.label))

    # Instantiate metrics object
    metrics = BinaryClassificationMetrics(predictionAndLabels)


    #def train(training_data, iterations, regParam, step):
    #model = SVMWithSGD.train(training_data, iterations=iterations, regParam=regParam, step=step)
    #return model

    #LogisticRegressionWithSGD

    print("\nUsing LogisticRegressionWithLBFGS classifier :")
    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)

    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)

    modelSVM = SVMWithSGD.train(train_data)

    # Compute raw scores on the test set
    predictionAndLabelsSVM = test_data.map(lambda lp: (float(modelSVM.predict(lp.features)), lp.label))

    # Instantiate metrics object
    metrics = BinaryClassificationMetrics(predictionAndLabelsSVM)


    print("\nUsing SVMWithSGD classifier :")
    # Area under precision-recall curve
    print("Area under PR = %s" % metrics.areaUnderPR)
    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)


    spark.stop()