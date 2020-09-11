#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------
Implementing Mean Absolute Percentage Error Using sklearn
Source: https://towardsdatascience.com/metrics-and-python-850b60710e0c
-------------------------------------------------------
-------------------------------------------------------
author:  Chandra Manivannan

version date: 11 September 2020

-------------------------------------------------------
"""

import pandas as pd 
from scipy.stats import boxcox_normmax
from scipy.special import boxcox1p
from sklearn.linear_model import LinearRegression

train = pd.read_csv('../input/home-data-for-ml-course/train.csv')
y = train.SalePrice.reset_index(drop=True)
features = train
end_features = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','MSSubClass','MSZoning']
features = features[end_features]
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
objects = [col for col in features.columns if features[col].dtype == "object"]
features.update(features[objects].fillna('None'))
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = [col for col in features.columns if features[col].dtype in numeric_dtypes]
features.update(features[numerics].fillna(0))
for i in numerics:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
    
X = pd.get_dummies(features).reset_index(drop=True)
#----------------- The model
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)

def MAPE(predict,target):
    return ( abs((target - predict) / target).mean()) * 100
print ('My MAPE: ' + str(MAPE(y_pred,y)) )