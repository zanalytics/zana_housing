# to handle datasets
import pandas as pd
import numpy as np
import logging
# for plotting
import matplotlib.pyplot as plt
# to divide train and test set
from sklearn.model_selection import train_test_split
# feature scaling
from sklearn.preprocessing import MinMaxScaler
# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.simplefilter(action='ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def clean_names(df):
    """
        Lowers all column names and replaces spaces with _

        Parameters:
            - df : dataframe
                Dataframe

        Returns:
            - Renamed dataframe columns
    """
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(" ", '_')
    df.columns = df.columns.str.replace(".", '_')
    return df


def create_dummy(df):
    """
        Lowers all column names and replaces spaces with _

        Parameters:
            - df : dataframe
                Dataframe

        Returns:
            - Renamed dataframe columns
    """
    df = pd.get_dummies(df, columns=["type", "land", "new_build"])
    if 'land_U' in df.columns:
        df = df.drop(['price', 'date', 'month_year', 'current_month', 'hpi_date','record', 'ppd', 'new_build_N', 'land_L', 'land_U'], axis=1)
    else:
        df = df.drop(['price', 'date', 'month_year', 'current_month', 'hpi_date','record', 'ppd', 'new_build_N', 'land_L'], axis=1)
   
    # visualise the dataset
    df['london_zone'] = df['london_zone'].replace(0,df.london_zone.max()+1)
    df['distance_to_station'] = df['distance_to_station'].round(2) + 1
    return df


def create_features(X_train, X_test):
    # make a list of the categorical variables that contain missing values
    data = X_train

    vars_with_na = [
        var for var in data.columns
        if X_train[var].isnull().sum() > 0  and X_train[var].dtypes == 'object'
    ]

    # replace missing values with new label: "Missing"

    X_train[vars_with_na] = X_train[vars_with_na].fillna('Missing')
    X_test[vars_with_na] = X_test[vars_with_na].fillna('Missing')

    # make a list with the numerical variables that contain missing values

    vars_with_na = [
        var for var in data.columns
        if X_train[var].isnull().sum() > 0 and X_train[var].dtypes != 'object'
    ]

    # replace engineer missing values as we described above

    for var in vars_with_na:

        # calculate the mode using the train set
        mode_val = X_train[var].mode()[0]

        # add binary missing indicator (in train and test)
        X_train[var+'_na'] = np.where(X_train[var].isnull(), 1, 0)
        X_test[var+'_na'] = np.where(X_test[var].isnull(), 1, 0)

        # replace missing values by the mode
        # (in train and test)
        X_train[var] = X_train[var].fillna(mode_val)
        X_test[var] = X_test[var].fillna(mode_val)

    # make list of numerical variables
    num_vars = [var for var in data.columns if data[var].dtypes != 'object']
    logging.info(f"Numerical variables:, {num_vars}")

    # for var in [
    # 'london_zone',
    # 'distance_to_station',
    # 'average_income',
    # 'adjusted_price']:
    #     X_train[var] = np.log(X_train[var])
    #     X_test[var] = np.log(X_test[var])

    # # same for train set
    # null_values = [var for var in [
    # 'london_zone',
    # 'distance_to_station',
    # 'average_income',
    # 'adjusted_price'] if X_train[var].isnull().sum() > 0]

    # logging.info(f"Null values after log:, {null_values}")

    # let's capture the categorical variables in a list

    cat_vars = [var for var in X_train.columns if X_train[var].dtype == 'object']

    def list_diff(list1, list2): 
	    return (list(set(list1) - set(list2))) 

    print(list_diff(X_train.columns, X_test.columns))
    print(len(X_train.columns)) 
    print(len(X_test.columns))
    
    def find_frequent_labels(df, var, rare_perc):
        
        # function finds the labels that are shared by more than
        # a certain % of the houses in the dataset

        df = df.copy()
        tmp = df.groupby(var)['adjusted_price'].count() / len(df)
        return tmp[tmp > rare_perc].index


    for var in cat_vars:
        
        # find the frequent categories
        frequent_ls = find_frequent_labels(X_train, var, 0.01)

        # replace rare categories by the string "Rare"
        X_train[var] = np.where(X_train[var].isin(
            frequent_ls), X_train[var], 'Rare')
        
        X_test[var] = np.where(X_test[var].isin(
            frequent_ls), X_test[var], 'Rare')

    # this function will assign discrete values to the strings of the variables,
    # so that the smaller value corresponds to the category that shows the smaller
    # mean house sale price
    print(len(X_train.columns)) 
    print(len(X_test.columns))

    def replace_categories(train, test, var, target):

        # order the categories in a variable from that with the lowest
        # house sale price, to that with the highest
        ordered_labels = train.groupby([var])[target].mean().sort_values().index

        # create a dictionary of ordered categories to integer values
        ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}

        # use the dictionary to replace the categorical strings by integers
        train[var] = train[var].map(ordinal_label)
        test[var] = test[var].map(ordinal_label)

    for var in cat_vars:
        replace_categories(X_train, X_test, var, 'adjusted_price')

    print(len(X_train.columns)) 
    print(len(X_test.columns)) 
    print(list_diff(X_train.columns, X_test.columns))
    print(X_train.isnull().sum())
    print(X_test.isnull().sum())
    
    train_vars = [var for var in X_train.columns if var not in ['id', 'adjusted_price']]

    print(train_vars)
    # create scaler
    # scaler = MinMaxScaler()

    # #  fit  the scaler to the train set
    # scaler.fit(X_train[train_vars]) 

    # # transform the train and test set
    # X_train[train_vars] = scaler.transform(X_train[train_vars])
    # X_test[train_vars] = scaler.transform(X_test[train_vars])

    # let's now save the train and test sets for the next notebook!
    print(len(X_train.columns)) 
    print(len(X_test.columns)) 
    # print(list_diff(X_train.columns, X_test.columns))

    X_train.to_csv('./data/processed/xtrain.csv', index=False)
    X_test.to_csv('./data/processed/xtest.csv', index=False)
    logging.info(f"Files written to processed")

if __name__ == "__main__":
    # load dataset
    train = pd.read_csv('./data/processed/train.csv')
    test = pd.read_csv('./data/processed/validation.csv')

    df_train = create_dummy(train)
    df_test = create_dummy(test)

    create_features(df_train, df_test)