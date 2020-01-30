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

# Fitting different algorithns to the Training set
# And applying k-Fold Cross Validation to determine most accurate regressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

algorithms = {'mlr' : LinearRegression(),
              'svr' : SVR(kernel = 'rbf', gamma = 'auto'),
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

#Using the most accurate predictor to make prediction
regressor = algorithms[best_regressor]
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