#!/usr/bin/python

import pandas as pd
import numpy as np
import re
import nltk
import spacy

from nltk.corpus import stopwords

from sklearn.preprocessing import MultiLabelBinarizer, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

nlp = spacy.load("en_core_web_sm")

import joblib
import os


def predictions(text):
    '''
    '''
    
    model_genre_clf = joblib.load(os.path.dirname(__file__) + '/model_genre_clf.pkl')
    
    dict_ = {'plot': [text]}
    datatesting_ =  pd.DataFrame(dict_)
    
    genre_prediction = model_genre_clf.predict(datatesting_['plot'])
    
    return (f'The predicted movies genres are: {le.inverse_transform(genre_prediction)}')

if __name__ == "__main__":
   print(predictions('Comedy','Drama'))


