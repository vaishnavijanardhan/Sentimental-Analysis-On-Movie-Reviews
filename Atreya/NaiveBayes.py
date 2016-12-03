# -*- coding: utf-8 -*-


import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#Path to read from
data_path=os.path.abspath('data')
train_file=os.path.join(data_path,'train.tsv')
test_file=os.path.join(data_path,'test.tsv')

#Read the files from the data folder
print "Reading Training and Test Data: \n"
train=pd.read_csv(train_file,delimiter="\t",header=0)
test=pd.read_csv(test_file,delimiter="\t",header=0)

##shapes of the file
print "Train file shape:  ", train.shape
print "Test file shape:    ", test.shape

print "\nFitting pipeline.. \n"

#Bag of Words from Sklearn
print "Running Feature Extraction.."
vectorizer=CountVectorizer() #initialise Bag of words
train_count=vectorizer.fit_transform(train.Phrase)
print "Bag of words Counts: ", train_count.shape

#Tf-Idf Transformer
print "Running Tf-Idf Transformer"
tf_idf=TfidfTransformer() #initialise Tf-Idf Transformer
train_tf_idf=tf_idf.fit_transform(train_count)
print "Tf-Idf : ", train_tf_idf.shape

#initialise a naive bayes classifier.
model=MultinomialNB()
print "Model Building: Fitting a Naive bayes classifier..\n"
model.fit(train_tf_idf,train.Sentiment)

#Process the test set
print "Processing Test set.. \n"
test_count=vectorizer.transform(test.Phrase)
test_tf_idf=tf_idf.transform(test_count)

#predict with the test data
print "Predicting with the Test data.. \n"
predicted=model.predict(test_tf_idf)

#predicton output to csv
print "Writing output in a csv file \n"
output=pd.DataFrame(data={"PhraseId":test.PhraseId,"Sentiment":predicted})
output.to_csv("NB.csv",index=False,quoting=3)
