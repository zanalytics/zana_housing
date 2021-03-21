import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import logging

random_seed = 42
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def find_frequent_labels(df, var, rare_perc):
    df = df.copy()
    tmp = df.groupby(var)['adjusted_price'].count() / len(df)
    return tmp[tmp > rare_perc].index


def replace_categories(train, validation,  test, var, target):
    ordered_labels = train.groupby([var])[target].mean().sort_values().index
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
    train[var] = train[var].map(ordinal_label)
    validation[var] = validation[var].map(ordinal_label)
    test[var] = test[var].map(ordinal_label)


def imputation_scaling(df):
    df = df.copy()
    cols_to_drop = ['price', 'date', 'month_year', 'current_month', 'hpi_date', 'ppd', 'record', 'ppd',
                    'lodgement_datetime', 'constituency_y', 'sheating_energy_eff', 'sheating_env_eff', 'id']
    df = df.drop(cols_to_drop, axis=1)
    df = pd.get_dummies(data=df, columns=["type", "land", "new_build"]).drop(['new_build_N', 'land_L'],axis=1)

    london_col = df[["london_zone"]]
    zone_col = SimpleImputer(strategy='constant', fill_value=10.0)
    zone_ft = zone_col.fit_transform(london_col)
    zone_df = pd.DataFrame(zone_ft, columns=london_col.columns)
    df[zone_df.columns] = zone_df

    numeric_cols = df.select_dtypes(include=[np.number])
    imp_median = SimpleImputer(strategy='median')
    numeric_ft = imp_median.fit_transform(numeric_cols)
    numeric_df = pd.DataFrame(numeric_ft, columns=numeric_cols.columns)

    text_cols = df.select_dtypes(include=['object'])
    imp_missing = SimpleImputer(strategy='constant', fill_value='Missing')
    text_ft = imp_missing.fit_transform(text_cols)
    text_df = pd.DataFrame(text_ft, columns=text_cols.columns)

    df[text_df.columns], df[numeric_df.columns] = text_df, numeric_df
    # make a list of the categorical variables that contain missing values

    for var in text_df.columns:
        frequent_ls = find_frequent_labels(df, var, 0.01)
        df[var] = np.where(df[var].isin(frequent_ls), df[var], 'Rare')

    return df, text_df.columns


if __name__ == "__main__":
    # load dataset

    df_train = pd.read_csv("./data/processed/train.csv")
    df_validation = pd.read_csv("./data/processed/validation.csv")
    df_test = pd.read_csv("./data/processed/test.csv")

    train, cat_vars = imputation_scaling(df_train)
    validation, cat_vars = imputation_scaling(df_validation)
    test, cat_vars = imputation_scaling(df_test)

    for var in cat_vars:
        replace_categories(train, validation, test, var, 'adjusted_price')

    train_vars = [var for var in train.columns if var not in ['adjusted_price']]

    # create scaler
    scaler = MinMaxScaler()
    scaler.fit(train[train_vars])
    train[train_vars] = scaler.transform(train[train_vars])
    validation[train_vars] = scaler.transform(validation[train_vars])
    test[train_vars] = scaler.transform(test[train_vars])

    # Save the split files
    train.to_csv("./data/processed/xtrain.csv", index=False)
    validation.to_csv("./data/processed/xvalidation.csv", index=False)
    test.to_csv("./data/processed/xtest.csv", index=False)

    logging.info(f"Data transformation completed")
