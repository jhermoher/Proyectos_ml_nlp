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

def preprocess_text(texts):

    def decontracter(text, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, text)

    def cleaning_plot(text):
        text = decontracter(text)
        text = re.sub(r"\'", "", text)
        text = re.sub(r"[^a-zA-Z]"," ",text)
        text = ' '.join(text.lower().split())
        return text
    
    def lemma_nlp(text):
        doc = nlp(text)
        lemmatized_plot= " ".join([token.lemma_ for token in doc])
        return lemmatized_plot

    def lemmatizer(text):
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if len(word) > 1 and word not in stop_words]
        clean_text = ' '.join(words)
        lemmatized_text = lemma_nlp(clean_text)
        return lemmatized_text

    return [lemmatizer(cleaning_plot(decontracter(text))) for text in texts]

def predictions(text):

	model_genre_clf = joblib.load(os.path.dirname(__file__) + '/model_genre_clf.pkl')
	
	dict_ = {'plot': [text]}
	datatesting_ =  pd.DataFrame(dict_)
	genre_prediction = model_genre_clf.predict(datatesting_['plot'])
	return (f'The predicted movies genres are: {le.inverse_transform(genre_prediction)}')

if __name__ == "__main__":
	plot = 'A hilarious comedy about a group of friends'
	print(predictions(plot))


