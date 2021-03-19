# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt

# to build the model
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
# to evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

# load the train and test set with the engineered variables

# we built and saved these datasets in a previous notebook.
# If you haven't done so, go ahead and check the previous notebooks (step 2)
# to find out how to create these datasets

X_train = pd.read_csv('./data/processed/xtrain.csv')
X_test = pd.read_csv('./data/processed/xtest.csv')
# capture the target (remember that is log transformed)

y_train = X_train['adjusted_price']
y_test = X_test['adjusted_price']

# load the pre-selected features
# ==============================

# we selected the features in the previous notebook (step 3)

# if you haven't done so, go ahead and visit the previous notebook
# to find out how to select the features

features = pd.read_csv('./data/processed/selected_features_one.csv')
features = features['0'].to_list() 

# We will add one additional feature to the ones we selected in the
# previous notebook: LotFrontage

# why?
#=====

# because it needs key feature engineering steps that we want to
# discuss further during the deployment part of the course. 

# features = features

# display final feature set
# features

# reduce the train and test set to the selected features
X_train = X_train[features]
X_test = X_test[features]

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

#lin_model = GradientBoostingRegressor()
lin_model = RandomForestRegressor()
#lin_model = MLPRegressor()
#lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# make predictions for train set
pred = lin_model.predict(X_train)

# # determine mse and rmse
# print('train mse: {}'.format(int(
#     mean_squared_error(np.exp(y_train), np.exp(pred)))))
# print('train rmse: {}'.format(int(
#     sqrt(mean_squared_error(np.exp(y_train), np.exp(pred))))))
# print('train r2: {}'.format(
#     r2_score(np.exp(y_train), np.exp(pred))))
# print()

# # make predictions for test set
# pred = lin_model.predict(X_test)

# # determine mse and rmse
# print('test mse: {}'.format(int(
#     mean_squared_error(np.exp(y_test), np.exp(pred)))))
# print('test rmse: {}'.format(int(
#     sqrt(mean_squared_error(np.exp(y_test), np.exp(pred))))))
# print('test r2: {}'.format(
#     r2_score(np.exp(y_test), np.exp(pred))))
# print()

# print('Average house price: ', int(np.exp(y_train).median()))

# from sklearn import metrics
# print('Mean Absolute Error:', metrics.mean_absolute_error(np.exp(y_test), np.exp(pred)))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(np.exp(y_test), np.exp(pred))))
# print('Root Mean Log Squared Error:', np.sqrt(metrics.mean_squared_log_error(np.exp(y_test), np.exp(pred))))
# print('Mean Absolute Error/Mean :', (metrics.mean_absolute_error(np.exp(y_test), np.exp(pred)) / np.exp(y_train).mean() * 100),'%')

# # let's evaluate our predictions respect to the real sale price
# plt.scatter(np.exp(y_test), np.exp(pred))
# plt.xlabel('True House Price')
# plt.ylabel('Predicted House Price')
# plt.title('Evaluation of Lasso Predictions')

# determine mse and rmse
print('train mse: {}'.format(int(
    mean_squared_error(y_train, pred))))
print('train rmse: {}'.format(int(
    sqrt(mean_squared_error(y_train, pred)))))
print('train r2: {}'.format(
    r2_score(y_train, pred)))
print()

# make predictions for test set
pred = lin_model.predict(X_test)

# determine mse and rmse
print('test mse: {}'.format(int(
    mean_squared_error(y_test, pred))))
print('test rmse: {}'.format(int(
    sqrt(mean_squared_error(y_test, pred)))))
print('test r2: {}'.format(
    r2_score(y_test, pred)))
print()

print('Average house price: ', int(y_train.median()))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print('Root Mean Log Squared Error:', np.sqrt(metrics.mean_squared_log_error(y_test, pred)))
print('Mean Absolute Error/Mean :', (metrics.mean_absolute_error(y_test, pred) / y_train.mean() * 100),'%')

# let's evaluate our predictions respect to the real sale price
# plt.scatter(y_test, pred)
# plt.xlabel('True House Price')
# plt.ylabel('Predicted House Price')
# plt.title('Evaluation of Lasso Predictions')
# plt.show()

errors = y_test - pred
plt.hist(errors, bins=75)
plt.show()