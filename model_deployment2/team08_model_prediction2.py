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

def predictions(text):
	contractions_dict = {
	    "ain ' t": "am not",
	    "aren ' t": "are not",
	    "can ' t": "cannot",
	    "can ' t ' ve": "cannot have",
	    " ' cause": "because",
	    "could ' ve": "could have",
	    "couldn ' t": "could not",
	    "couldn ' t ' ve": "could not have",
	    "didn ' t": "did not",
	    "doesn ' t": "does not",
	    "don ' t": "do not",
	    "hadn ' t": "had not",
	    "hadn ' t ' ve": "had not have",
	    "hasn ' t": "has not",
	    "haven ' t": "have not",
	    "he ' d": "he would",
	    "he ' d ' ve": "he would have",
	    "he ' ll": "he will",
	    "he ' ll ' ve": "he will have",
	    "he ' s": "he is",
	    "how ' d": "how did",
	    "how ' d ' y": "how do you",
	    "how ' ll": "how will",
	    "how ' s": "how is",
	    "I ' d": "I would",
	    "I ' d ' ve": "I would have",
	    "I ' ll": "I will",
	    "I ' ll ' ve": "I will have",
	    "I ' m": "I am",
	    "I ' ve": "I have",
	    "isn ' t": "is not",
	    "it ' d": "it would",
	    "it ' d ' ve": "it would have",
	    "it ' ll": "it will",
	    "it ' ll ' ve": "it will have",
	    "it ' s": "it is",
	    "let ' s": "let us",
	    "ma ' am": "madam",
	    "mayn ' t": "may not",
	    "might ' ve": "might have",
	    "mightn ' t": "might not",
	    "mightn ' t ' ve": "might not have",
	    "must ' ve": "must have",
	    "mustn ' t": "must not",
	    "mustn ' t ' ve": "must not have",
	    "needn ' t": "need not",
	    "needn ' t ' ve": "need not have",
	    "o ' clock": "of the clock",
	    "oughtn ' t": "ought not",
	    "oughtn ' t ' ve": "ought not have",
	    "shan ' t": "shall not",
	    "sha ' n ' t": "shall not",
	    "shan ' t ' ve": "shall not have",
	    "she ' d": "she would",
	    "she ' d ' ve": "she would have",
	    "she ' ll": "she will",
	    "she ' ll ' ve": "she will have",
	    "she ' s": "she is",
	    "should ' ve": "should have",
	    "shouldn ' t": "should not",
	    "shouldn ' t ' ve": "should not have",
	    "so ' ve": "so have",
	    "so ' s": "so is",
	    "that ' d": "that would",
	    "that ' d ' ve": "that would have",
	    "that ' s": "that is",
	    "there ' d": "there would",
	    "there ' d ' ve": "there would have",
	    "there ' s": "there is",
	    "they ' d": "they would",
	    "they ' d ' ve": "they would have",
	    "they ' ll": "they will",
	    "they ' ll ' ve": "they will have",
	    "they ' re": "they are",
	    "they ' ve": "they have",
	    "to ' ve": "to have",
	    "wasn ' t": "was not",
	    "we ' d": "we would",
	    "we ' d ' ve": "we would have",
	    "we ' ll": "we will",
	    "we ' ll ' ve": "we will have",
	    "we ' re": "we are",
	    "we ' ve": "we have",
	    "weren ' t": "were not",
	    "what ' ll": "what will",
	    "what ' ll ' ve": "what will have",
	    "what ' re": "what are",
	    "what ' s": "what is",
	    "what ' ve": "what have",
	    "when ' s": "when is",
	    "when ' ve": "when have",
	    "where ' d": "where did",
	    "where ' s": "where is",
	    "where ' ve": "where have",
	    "who ' ll": "who will",
	    "who ' ll ' ve": "who will have",
	    "who ' s": "who is",
	    "who ' ve": "who have",
	    "why ' s": "why is",
	    "why ' ve": "why have",
	    "will ' ve": "will have",
	    "won ' t": "will not",
	    "won ' t ' ve": "will not have",
	    "would ' ve": "would have",
	    "wouldn ' t": "would not",
	    "wouldn ' t ' ve": "would not have",
	    "y ' all": "you all",
	    "y ' all ' d": "you all would",
	    "y ' all ' d ' ve": "you all would have",
	    "y ' all ' re": "you all are",
	    "y ' all ' ve": "you all have",
	    "you ' d": "you would",
	    "you ' d ' ve": "you would have",
	    "you ' ll": "you will",
	    "you ' ll ' ve": "you will have",
	    "you ' re": "you are",
	    "you ' ve": "you have"
	}
	contractions_re = re.compile(r'\b(?:%s)\b' % '|'.join(re.escape(key) for key in contractions_dict.keys()))

	class Decontracter(BaseEstimator, TransformerMixin):
	    def fit(self, X, y=None):
	        return self
	    
	    def transform(self, X):
	        return [self.decontract(text) for text in X]
	    
	    def decontract(self, text):
	        def replace(match):
	            return contractions_dict[match.group(0)]
	        return contractions_re.sub(replace, text)
	
	class CleaningPlot(BaseEstimator, TransformerMixin):
	    def fit(self, X, y=None):
	        return self
	    
	    def transform(self, X):
	        return [self.clean(text) for text in X]
	    
	    def clean(self, text):
	        text = Decontracter().decontract(text)
	        text = re.sub(r"\'", "", text)
	        text = re.sub(r"[^a-zA-Z]", " ", text)
	        text = ' '.join(text.lower().split())
	        return text
	
	class Lemmatizer_nlp(BaseEstimator, TransformerMixin):
	    def fit(self, X, y=None):
	        return self
	    
	    def transform(self, X):
	        return [self.lemmatize(text) for text in X]
	    
	    def lemmatize(self, text):
	        doc = nlp(text)
	        lemmatized_plot = " ".join([token.lemma_ for token in doc])
	        return lemmatized_plot
	
	class Lemmatizer(BaseEstimator, TransformerMixin):
	    def fit(self, X, y=None):
	        return self
	    
	    def transform(self, X):
	        return [self.final_lemmatize(text) for text in X]
	    
	    def final_lemmatize(self, text):
	        stop_words = set(stopwords.words('english'))
	        words = text.split()
	        words = [word for word in words if len(word) > 1 and word not in stop_words]
	        clean_text = ' '.join(words)
	        lemmatized_text = Lemmatizer_nlp().lemmatize(clean_text)
	        return lemmatized_text
			    
	model_genre_clf = joblib.load(os.path.dirname(__file__) + '/model_genre_clf.pkl')
	le = joblib.load('label_encoder.pkl')
	
	dict_ = {'plot': [text]}
	datatesting_ =  pd.DataFrame(dict_)
	genre_prediction = model_genre_clf.predict(datatesting_['plot'])
	return (f'The predicted movies genres are: {le.inverse_transform(genre_prediction)}')

if __name__ == "__main__":
	plot = 'A hilarious comedy about a group of friends'
	print(predictions(plot))
