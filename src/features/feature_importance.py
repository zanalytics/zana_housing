import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot

# # feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

if __name__ == '__main__':
    df = pd.read_csv('./data/processed/xtrain.csv')
    df_test = pd.read_csv('./data/processed/xtest.csv')
    y = df['adjusted_price']
    # y_test = df_test['adjusted_price']
    features = list(df)
    features.remove('adjusted_price')
    features.remove('id')
    x = df[features]
    x_test = df_test[features]
    # x_test['land_U'] = x_test['land_U'].fillna(0, inplace = True)
    # x_test = x_test.loc[:, x_test.columns != 'land_U']
    # run_all(x, y, small=False, normalize_x=False)
    print(len(x.columns))
    print(len(x_test.columns))
    print(x.columns)
    print(x_test.columns)
    
    # print(x_test.isnull().sum())

    # # feature selection
    X_train_fs, X_test_fs, fs = select_features(x, y, x_test)
    # what are scores for the features
    for i in range(len(fs.scores_)):
        print(f"Feature {features[i]} :, {fs.scores_[i]}")
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
