import pandas as pd
import numpy as np
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import accuracy_score
import argparse
import yaml
from time import time
import csv
import sys

def print_timing(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        duration = (time() - t0)*1000.0
        print('%s took %0.3f ms' %(func.func_name, duration))
        return res
    return wrapper

def load_train_data():
    train_file = './data/train.tsv'

    # import the training data into a pandas dataframe (rename columns)
    df = pd.read_table(train_file, header=0)
    df.columns = ['phraseid', 'sentenceid', 'phrase', 'sentiment']

    return df.to_records()

def load_test_data():
    test_file = './data/test.tsv'

    # import the training data into a pandas dataframe (rename columns)
    df = pd.read_table(test_file, header=0)
    df.columns = ['phraseid', 'sentenceid', 'phrase']
    return df.to_records()
    
@print_timing
def split_train_data(proportion=0.8):
    train_file = './data/train.tsv'

    # import the training data into a pandas dataframe (rename columns)
    df = pd.read_table(train_file, header=0)
    df.columns = ['phraseid', 'sentenceid', 'phrase', 'sentiment']
    
    # split the data based on the sentence id (unique sentence)
    ids = pd.unique(df.sentenceid.ravel())
    N = int(len(ids) * proportion) # number in the training set
    np.random.shuffle(ids)

    # get the sentence ids in training and testing set, then split the dataframe
    train_ids, test_ids = np.split(ids, [N])
    train = df[df['sentenceid'].isin(train_ids)].to_records()
    test  = df[df['sentenceid'].isin(test_ids)].to_records()

    return train, test

def target(data):
    return np.array([x.sentiment for x in data])

class StatelessTransform(object):
    def __init__(self):
        super(StatelessTransform, self).__init__()

    def fit(self, X, y=None):
        return self

class ExtractText(StatelessTransform):
    def __init__(self, lowercase=False):
        super(ExtractText, self).__init__()
        self.lowercase = lowercase

    def transform(self, X):
        words = (" ".join(nltk.word_tokenize(x.phrase)) for x in X)
        if self.lowercase:
            return [x.lower() for x in words]
        return list(words)

class WordNetFeatures(StatelessTransform):
    def __init__(self):
        super(WordNetFeatures, self).__init__()

    def transform(self, X):
        return [self._text_to_synsets(x) for x in X]

    def _text_to_synsets(self, text):
        result = []
        for word in text.split():
            ss = wn.synsets(word)
            result.extend(str(s) for s in ss if '.n.' not in str(s))
        return " ".join(result)

class SentiWordNetFeatures(StatelessTransform):
    def __init__(self):
        super(SentiWordNetFeatures, self).__init__()

    def transform(self, X):
        return [self._text_to_sentiment(x) for x in X]
    
    def _text_to_sentiment(self, text):
        # get the sentiment score for a word
        def score_tuple(ss):
            return np.array([ss.neg_score(), ss.obj_score(), ss.pos_score()])
        
        # map the numeric index to a word
        senti_dict = {0 : 'NEGATIVE', 1 : 'OBJECTIVE', 2 : 'POSITIVE'}
        def senti_mapper(x):
            return senti_dict[x]
        senti_mapper = np.vectorize(senti_mapper)
 
        result = []
        for word in text.split():
            ss = swn.senti_synsets(word)
            scores = np.array([score_tuple(s) for s in ss if '.n.' not in str(s)])
            if scores.size is not 0:
                sentiment = np.argmax(scores, axis=1)
                sentiment = senti_mapper(sentiment)
                result.extend(sentiment.tolist())
        return " ".join(result)
        
        # get the mean scores and return the index of the highest score
        # i.e. (neg=0, obj=1, pos=2)
        mean_scores = np.mean(result, axis=0)
        return np.argmax(mean_scores)

class SGDOvOAsFeatures(StatelessTransform):
    def __init__(self):
        super(SGDOvOAsFeatures, self).__init__()

    def fit(self, X, y):
        ovo = OneVsOneClassifier(SGDClassifier(), n_jobs=-1).fit(X, y)
        self.classifiers = ovo.estimators_
        return self

    def transform(self, X, y=None):
        xs = [clf.decision_function(X).reshape(-1,1) for clf in self.classifiers]
        return np.hstack(xs)

class Baseline(object):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.array([2 for _ in X])

_classifiers = {'baseline' : Baseline,
                'naivebayes' : MultinomialNB,
                'linear_svm' : LinearSVC,
                'sgd' : SGDClassifier,
                'randomforest' : RandomForestClassifier}

class SentimentAnalysis(object):
    def __init__(self, classifier='randomforest', classifier_args=None,
                 lowercase=True, bagwords=None, kbest=None, sgdovo=False, sentiment_score=False, wordnet=False):
        super(SentimentAnalysis, self).__init__()
        
        # pre-processing common to every feature extraction
        pipeline = [ExtractText(lowercase=lowercase)] 

        # feature extended extractions

        if kbest is not None: # use kbest features from a chi2 test
            ext = [make_pipeline(CountVectorizer(tokenizer=lambda x: x.split(),
                                                 **bagwords),
                                 SelectKBest(chi2, k=kbest))]
        else: # train a SGD classifier in a one-vs-other scheme
            ext = [make_pipeline(CountVectorizer(tokenizer=lambda x: x.split(),
                                                 **bagwords),
                                 SGDOvOAsFeatures())]                     
        if sentiment_score: # use the sentiment score from senti_wordnet
            ext.append(make_pipeline(SentiWordNetFeatures(),
                                     CountVectorizer(tokenizer=lambda x: x.split(),
                                                     **bagwords)))
        if wordnet: # use synsets of each word instead (less variation)
            ext.append(make_pipeline(WordNetFeatures(),
                                     CountVectorizer(tokenizer=lambda x: x.split(),
                                                     **bagwords)))

        # combine all the different feature extractions
        ext = make_union(*ext)
        pipeline.append(ext)
        
        # build classifier
        if classifier_args is None:
            classifier_args = {}
        self.clf_type = classifier
        self.classifier = _classifiers[classifier](**classifier_args)    
        self.pipeline = make_pipeline(*pipeline)
    
    @print_timing
    def fit(self, X):
        # transform data
        y = target(X)
        Z = self.pipeline.fit_transform(X,y)
        if self.clf_type == 'randomforest' and issparse(Z):
            self.classifier.fit(Z.toarray(), y)
        else:
            self.classifier.fit(Z, y)
    
    @print_timing
    def predict(self, X):
        Z = self.pipeline.transform(X)
        if self.clf_type == 'randomforest' and issparse(Z):
            yhat = self.classifier.predict(Z.toarray())
        else:
            yhat = self.classifier.predict(Z)

        return yhat

    def score(self, X):
        y = target(X)
        yhat = self.predict(X)
        return np.mean(yhat == y)

@print_timing
def cross_validation(analysis, K=10):
    scores = np.zeros(K)
    for k in xrange(K):
        train, test = split_train_data()
        predictor = analysis()
        predictor.fit(train)
        score = predictor.score(test)
        scores[k] = score
        print('CV[%i] : score = %f'%(k, score))
    return np.mean(scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Senitment Analysis for Movie Reviews')
    parser.add_argument('yaml', help='yaml init file')
    args = parser.parse_args()
    config = yaml.load(open(args.yaml))
    
    # fit model to training data
    train = load_train_data()
    analysis = SentimentAnalysis(**config)
    analysis.fit(train)


    # get results on testing data
    test = load_test_data()
    yhat = analysis.predict(test)
    
    # write results to csv
    writer = csv.writer(sys.stdout)
    writer.writerow(("PhraseId", "Sentiment"))
    for x, sentiment in zip(test, yhat):
        writer.writerow((x.phraseid, sentiment))
