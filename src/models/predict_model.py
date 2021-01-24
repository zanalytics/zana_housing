
#import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.model_selection import train_test_split

# pd.options.plotting.backend = 'matplotlib'
# plt.interactive(True)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_columns', 500)

keep_cols = ['id', 'adjusted_price', 'type', 'new_build', 'land', 'latitude', 'longitude']

df = pd.read_csv("./data/processed/processed.csv", usecols=keep_cols)

print(df)
df = df[(df['adjusted_price'] <= 1000000)].reset_index(drop=True)
df = df.sample(frac=.01)
df.to_csv("./data/processed/one_percent.csv")
# df_type = pd.get_dummies(df.type, prefix='type')
# df_new_build = pd.get_dummies(df.new_build, prefix='new_build')
# df_land = pd.get_dummies(df.land, prefix='land')

# df = pd.concat([df, df_type, df_new_build, df_land], axis=1, sort=False)

# df = df.drop(columns=['type', 'new_build', 'land'])

# df = df[['id', 'latitude', 'longitude', 'type_D', 'type_F',
#        'type_S', 'type_T', 'new_build_Y', 'land_F',
#        'adjusted_price']]

# # df.to_csv("/Users/chris/Documents/data_science/experimentation/machine-learning-big-loop/data.csv")

# X = df.iloc[:, 1:-1]  # independent columns
# y = df.iloc[:, -1]  # target column

# # split our data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# from sklearn.ensemble import RandomForestRegressor
# import numpy as np
# from sklearn.compose import TransformedTargetRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from scipy.stats import boxcox
# from scipy.special import inv_boxcox
# from sklearn.preprocessing import PowerTransformer, QuantileTransformer

# # forest_reg = TransformedTargetRegressor(regressor=RandomForestRegressor(),
# #                                         transformer=QuantileTransformer(output_distribution='normal'))

# forest_reg = RandomForestRegressor()
# forest_reg.fit(X_train, y_train)
# y_pred = forest_reg.predict(X_test)

# from sklearn import metrics
# metrics.regression_report
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('Mean Absolute Error/Mean :', (metrics.mean_absolute_error(y_test, y_pred) / df["adjusted_price"].mean() * 100),
#       '%')

# from matplotlib import pyplot as plt
# from scipy.stats import normaltest
# import numpy as np
# from sklearn.preprocessing import PowerTransformer

# data = np.array(
#     [-0.35714286, -0.28571429, -0.00257143, -0.00271429, -0.00142857, 0., 0., 0., 0.00142857, 0.00285714, 0.00714286,
#      0.00714286, 0.01, 0.01428571, 0.01428571, 0.01428571, 0.01428571, 0.01428571, 0.01428571, 0.02142857, 0.07142857])

# pt = PowerTransformer(method='yeo-johnson')
# data = data.reshape(-1, 1)
# pt.fit(data)
# transformed_data = pt.transform(data)

# k2, p = normaltest(data)
# transformed_k2, transformed_p = normaltest(transformed_data)

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_regression
# from matplotlib import pyplot


# # feature selection
# def select_features(X_train, y_train, X_test):
#     # configure to select all features
#     fs = SelectKBest(score_func=f_regression, k='all')
#     # learn relationship from training data
#     fs.fit(X_train, y_train)
#     # transform train input data
#     X_train_fs = fs.transform(X_train)
#     # transform test input data
#     X_test_fs = fs.transform(X_test)
#     return X_train_fs, X_test_fs, fs


# # feature selection
# X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# # what are scores for the features
# for i in range(len(fs.scores_)):
#     print('Feature %d: %f' % (i, fs.scores_[i]))
# # plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()

# from sklearn import linear_model
# import numpy as np

# clf = linear_model.Lasso(alpha=0.1)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# from sklearn import metrics

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('Mean Absolute Error/Mean :', (metrics.mean_absolute_error(y_test, y_pred) / df["adjusted_price"].mean() * 100),
#       '%')

# from sklearn.ensemble import GradientBoostingRegressor
# import numpy as np

# linear = GradientBoostingRegressor()
# linear.fit(X_train, y_train)
# y_pred = linear.predict(X_test)

# from sklearn import metrics

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('Mean Absolute Error/Mean :', (metrics.mean_absolute_error(y_test, y_pred) / df["adjusted_price"].mean() * 100),
#       '%')

# from sklearn.neural_network import MLPRegressor
# import numpy as np

# mlp = MLPRegressor()

# mlp.fit(X_train, y_train)
# y_pred = mlp.predict(X_test)

# from sklearn import metrics

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('Mean Absolute Error/Mean :', (metrics.mean_absolute_error(y_test, y_pred) / df["adjusted_price"].mean() * 100),
#       '%')

# forest_reg.get_params()

# from sklearn.model_selection import RandomizedSearchCV

# # Number of trees in random forest
# n_estimators = [100]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# print(random_grid)

# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation,
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
#                                random_state=42, n_jobs=-1)
# # Fit the random search model
# rf_random.fit(X_train, y_train)

# rf_random.best_params_

# from sklearn.neural_network import MLPRegressor

# regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# from sklearn import metrics

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('Mean Absolute Error/Mean :', (metrics.mean_absolute_error(y_test, y_pred) / df["adjusted_price"].mean() * 100),
#       '%')

# import seaborn as sns

# plt.figure(figsize=(20, 12))
# ax = sns.regplot(x=y_test, y=y_pred)
# ax.set_title('Actual vs Predicted')
# ax.set_ylabel('Predicted cost of Job')
# ax.set_xlabel('Actual cost of Job')
# #
# # # train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)
# # import seaborn as sns
# #
# # plt.figure(figsize=(20, 12))
# # ax = sns.regplot(x=y_test, y=y_pred)
# # ax.set_title('Actual vs Predicted')
# # ax.set_ylabel('Predicted cost of Job')
# # ax.set_xlabel('Actual cost of Job')
# #
# # df["bx_adjusted_price"], lamb = boxcox(df["adjusted_price"])
# # inv_boxcox(y)
# #
# # qt = QuantileTransformer(output_distribution='normal')
# # Z = qt.fit_transform((df["adjusted_price"].to_numpy().reshape(-1, 1)))
# #
# # df["turn_back"] = np.exp(df["ln_adjusted_price"]).astype(int)
# #
# # sns.displot(df, x="bx_adjusted_price", kind="hist", bins=9)
# #
# # sns.displot(x=Z, kind="kde")
# # Z
