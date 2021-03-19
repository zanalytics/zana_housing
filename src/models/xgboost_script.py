# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
 
xgbr = xgb.XGBRegressor(verbosity=0)
print(xgbr)

xgbr.fit(X_train, y_train)
 
score = xgbr.score(X_train, y_train)   
print("Training score: ", score) 
 
# - cross validataion 
scores = cross_val_score(xgbr, X_train, y_train, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbr, X_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
 
y_pred = xgbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, lw=1, color="blue", label="original", linestyle = 'dashed')
plt.scatter(x_ax, y_pred, s=2, color="red", label="predicted")
plt.legend()
plt.show()