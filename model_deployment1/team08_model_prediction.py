#!/usr/bin/python

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

import joblib
import os

def prediction(year, mileage, state, make, model):
    '''
    '''
    
    model_carspricing = joblib.load(os.path.dirname(__file__) + '/model_carspricing.pkl') 

    dict_ = {
        'Year': [year],
        'Mileage': [mileage],
        'State': [state],
        'Make': [make],
        'Model': [model]}
    
    dataTesting_ = pd.DataFrame(dict_)
    
    dataTesting_['Make-Model'] = dataTesting_['Make'] + '-' + dataTesting_['Model']
    dataTesting_.drop(columns=['Make','Model'], inplace=True)

    car_prediction = model_carspricing.predict(pd.DataFrame(dataTesting_))

    return (f'Pre-owned car price prediction is: {car_prediction[0].astype(int):,.0f}')

if __name__ == "__main__":
   print(prediction(2015, 50000, ' FL', 'Jeep', 'Wrangler'))


