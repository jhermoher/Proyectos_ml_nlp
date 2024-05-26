#!/usr/bin/python

import pandas as pd
import numpy as np
import re
import nltk
import spacy

from nltk.corpus import stopwords

from sklearn.preprocessing import MultiLabelBinarizer, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

nlp = spacy.load("en_core_web_sm")

import joblib
import os

from pre-processing import Decontracter, CleaningPlot, Lemmatizer

def predictions(text):	    
	model_genre_clf = joblib.load(os.path.dirname(__file__) + '/model_genre_clf_1.pkl')
	le = joblib.load(os.path.dirname(__file__) + 'label_encoder.pkl')
	
	decontracter = Decontracter()
	cleaning_plot = CleaningPlot()
	lemmatizer = Lemmatizer()
	cleaned_text = cleaning_plot.transform([text])
	lemmatized_text = lemmatizer.transform(cleaned_text)
	    
	genre_prediction = model_genre_clf.predict(lemmatized_text)
	return (f'The predicted movies genres are: {le.inverse_transform(genre_prediction)}')

if __name__ == "__main__":
	plot = 'A hilarious comedy about a group of friends'
	print(predictions(plot))
