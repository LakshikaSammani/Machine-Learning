# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:05:50 2019

@author: Sammani Chandradeva
"""

#Data Preprocessing
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('Data.csv')

#Importing the dataset
X = dataset.iloc[:,:-1].values #Code to import all the columns except the last column
y = dataset.iloc[:, 3].values

#Filling missing data
from sklearn.preprocessing import Imputer
imputer = Imputer (missing_values ='NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder (categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)

#Splitting the dataset as Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

