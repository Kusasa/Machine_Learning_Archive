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
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np

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

# Fitting different algorithns to the Training set
# And applying k-Fold Cross Validation to determine most accurate regressor
algorithms = {'mlr' : LinearRegression(),
              'svr' : SVR(kernel = 'rbf'),
              'dtr' : DecisionTreeRegressor(random_state = 0),
              'rfr' : RandomForestRegressor(n_estimators = 100, random_state = 0)}

means = {'mlr' : '',
         'svr' : '',
         'dtr' : '',
         'rfr' : ''}
squared_means = {'mlr' : '',
                 'svr' : '',
                 'dtr' : '',
                 'rfr' : ''}

for algorithm in algorithms:
    regressor = algorithms[algorithm]
    regressor.fit(X, Y)
    accuracies = cross_val_score(estimator = regressor, X = X, y = Y, cv = 10)
    means[algorithm] = accuracies.mean()
    squared_means[algorithm] = (accuracies.mean())**2

best_regressor = min(means, key=squared_means.get)
best_regressor_value = means[min(means, key=squared_means.get)]
print('The Most Accurate regressor: ' + best_regressor + ' \n With RMSEcv of: ' + str(best_regressor_value))

# Applying Random Search to find the range of best model parameters
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 11)]    # Number of trees in random forest
max_features = ['auto', 'sqrt']          # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]          # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10]          # Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4]          # Minimum number of samples required at each leaf node
bootstrap = [True, False]          # Method of selecting samples for training each tree
random_state = [int(x) for x in np.linspace(0, 50, num = 11)] # Randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node

parameters_random = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'random_state': random_state}

regressor = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = regressor,
                               param_distributions = parameters_random,
                               n_iter = 50,
                               cv = 5,
                               verbose=2,
                               n_jobs = -1)
rf_random = rf_random.fit(X, Y)
best_accuracy_random = rf_random.best_score_
best_parameters_random = rf_random.best_params_

# Applying Grid Search to find the best model parameters
parameters_grid = {'n_estimators': [95, 100, 105],
                   'max_features': ['sqrt'],
                   'max_depth': [9, 10, 11],
                   'min_samples_split': [2, 3],
                   'min_samples_leaf': [1, 2],
                   'bootstrap': [True],
                   'random_state': [25, 30, 35, 40]}

regressor = RandomForestRegressor()
rf_grid = GridSearchCV(estimator = regressor,
                       param_grid = parameters_grid,
                       cv = 5,
                       verbose=2,
                       n_jobs = -1)
rf_grid = rf_grid.fit(X, Y)
best_accuracy_grid = rf_grid.best_score_
best_parameters_grid = rf_grid.best_params_

#Using the most accurate predictor to make prediction
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor = regressor.fit(X, Y)
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