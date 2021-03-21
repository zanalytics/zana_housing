import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
from matplotlib import pyplot as plt


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
    features = list(df_test)
    features.remove('adjusted_price')
    x = df[features]
    x_test = df_test[features]
    # # feature selection
    X_train_fs, X_test_fs, fs = select_features(x, y, x_test)
    # what are scores for the features
    feature_column = []
    feature_score = []
    for i in range(len(fs.scores_)):
        feature_column.append(features[i])
        feature_score.append(fs.scores_[i])
    df_feature = pd.DataFrame({'score': feature_score}, index=feature_column)
    df_feature[['score']] = df_feature[['score']].apply(pd.to_numeric)
    df_feature = df_feature.sort_values(by='score')
    df_feature[df_feature['score'] > 3000].plot(kind='bar')
    plt.tight_layout()
    plt.savefig('./reports/figures/feature_importance.png')
    df = df_feature[df_feature['score'] > 3000]
    df.index.names = ['feature']
    df.to_csv("./data/processed/selected_features.csv", index=True)
