import pandas as pd
import os

column_names = ['id', 'price', 'date', 'postcode',
                'type', 'new_build', 'land', 'primary_address',
                'secondary_address', 'street', 'locality', 'town_city',
                'district', 'county', 'ppd', 'record']

# Process price paid data
df_price_paid = pd.read_csv(os.getcwd() + "/data/raw/pp-complete.csv", names=column_names, parse_dates=['date'])

# Read in house price index
df_house_index = pd.read_csv("/data/raw/house_price_index.csv")

# Read in postcode data
df_postcode = pd.read_csv("../data/raw/postcodes.csv")


def remove_duplicates(df):
    df = df.drop_duplicates(subset=df.columns[1:], keep="first")
    df = df[df['postcode'].notnull()]
    return df


def price_cut(df, min, max):
    df = df[(df['price'] <= max) & (df['price'] >= min)].reset_index(drop=True)
    df['month_year'] = df['date'].astype('datetime64[M]')
    return df


def drop_columns(df, cols, string, use_string=False):
    if use_string:
        df = df.drop(columns=cols)
    else:
        df = df.drop(columns=df.columns[df.columns.str.contains(pat=string)])
    return df


def rename_columns(df):
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(' ', '_')
    return df


def col_to_dates(df, cols):
    df[cols] = df[cols].apply(pd.to_datetime)
    return df


# Run pipeline the prise paid data.
df_price_paid = (df_price_paid.pipe(remove_duplicates)
                 .pipe(price_cut, min=10000, max=5000000)
                 .pipe(drop_columns, cols=['locality', 'town_city', 'district', 'county'])
                 )

# Run pipeline the prise paid data.
df_house_index = (df_house_index.pipe(rename_columns)
                  .pipe(drop_columns, string='change|price', use_string=True)
                  .pipe(col_to_dates, cols=['date'])
                  )
