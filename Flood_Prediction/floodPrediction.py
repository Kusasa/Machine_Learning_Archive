""" -----------------------------------------------------------------------------
Tool Name:          Flood Prediction in Malawi
Version:            1.0
Description:        Tool used to predict the occurence and distribution of flood
                    occurence in Malawi.
Author:             Kusasalethu Sithole
Date:               2020-01-23
Last Revision:      2020-01-23
------------------------------------------------------------------------------ """

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
datasource = pd.read_csv('./data/Train.csv')
nullCells = datasource.isna()   #Test data for nulls
X = datasource.iloc[:, 3:21].values    # Still need to include the landcover column
Y = datasource.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Fitting Kernel SVM to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, Y)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X, y = Y, cv = 10)
accuracies.mean()
accuracies.std()

#Prediction
test_X = datasource.iloc[:, 21:39].values # Still need to include the elevation column
test_X = sc.transform(test_X) 
test_Y = regressor.predict(test_X)
# save .csv with test_Y and grid ID