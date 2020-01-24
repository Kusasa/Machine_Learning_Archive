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
import pandas as pd

# Importing the dataset
dS = pd.read_csv('./data/Train.csv')
nullCells = dS.isna()   #Test data for nulls
training_columns = []
for index in range(5,21):
    training_columns.extend([index])
training_columns = [3] + training_columns + [38]
X = dS.iloc[:, training_columns].values
Y = dS.iloc[:, 2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Fitting Kernel SVM to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, Y)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X, y = Y, cv = 10)
accuracies.mean()
accuracies.std()

#Prediction
pred_columns = []
for index in range(21,38):
    pred_columns.extend([index])
pred_columns = [3] + pred_columns
pred_X = dS.iloc[:, pred_columns].values
pred_X = sc.transform(pred_X) 
pred_Y = regressor.predict(pred_X)

# Export pred_Y and grid ID
Square_ID = dS.iloc[:, 39].values
Square_ID = pd.DataFrame(data=Square_ID, columns=['Square_ID'])
pred_Y = pd.DataFrame(data=pred_Y, columns=['target_2019'])
pred_csv = pd.concat([Square_ID, pred_Y], axis=1, sort=False)
pred_csv.to_csv('./submission.csv')