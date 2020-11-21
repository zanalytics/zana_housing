from typing import List, Any, Union

import pandas as pd
from datetime import datetime
import dateutil.relativedelta

column_names = ['id', 'price', 'date', 'postcode',
                'type', 'new_build', 'land', 'primary_address',
                'secondary_address', 'street', 'locality', 'town_city',
                'district', 'county', 'ppd', 'record']

# Process price paid data
df_price_paid = pd.read_csv("./data/raw/pp-complete.csv", names=column_names, parse_dates=['date'])

# Read in house price index
df_house_index = pd.read_csv("./data/raw/house_price_index.csv")

# Read in postcode data
df_postcode = pd.read_csv("./data/raw/postcodes.csv")


def remove_duplicates(df):
    df = df.drop_duplicates(subset=df.columns[1:], keep="first")
    df = df[df['postcode'].notnull()]
    return df


def price_paid_process(df, min, max, number_of_months):
    df = df[(df['price'] <= max) & (df['price'] >= min)].reset_index(drop=True)
    df['month_year'] = df['date'].astype('datetime64[M]')
    df['current_month'] = pd.to_datetime(datetime.today().date().replace(day=1)) - dateutil.relativedelta. \
        relativedelta(months=number_of_months)
    return df


def drop_columns(df, cols, string, use_string=False):
    if use_string:
        df = df.drop(columns=cols)
    else:
        df = df.drop(columns=df.columns[df.columns.str.contains(pat=string)])
    return df


def clean_names(df):
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace(' ', '_')
    return df


def col_to_dates(df, cols):
    df[cols] = df[cols].apply(pd.to_datetime)
    return df


def rename_columns(df, dictionary):
    df = df.rename(dictionary, axis='columns')
    return df


def mean_column(df, column_name, avg_list):
    df[column_name] = df[avg_list].mean(axis=1)
    return df

def adjust_price(df, type ,current_index, pp_index):
    df.loc[df['type'] == type, 'adjusted_price'] = (df[current_index] / df[pp_index]) * df['price']
    return df

# Run pipeline the prise paid data.
df_price_paid = (df_price_paid.pipe(remove_duplicates)
                 .pipe(price_paid_process, min=10000, max=5000000, number_of_months=3)
                 .pipe(drop_columns, cols=['locality', 'town_city', 'district', 'county'])
                 )

# Run pipeline the prise paid data.
df_house_index = (df_house_index.pipe(clean_names)
                  .pipe(drop_columns, string='change|price', use_string=True)
                  .pipe(col_to_dates, cols=['date'])
                  .rename({'date': 'hpi_date'},  axis='columns')
                  )

# Run pipeline the prise paid data.
postcode_columns: List[Union[str, Any]] = ['postcode', 'latitude', 'longitude', 'grid_ref', 'county', 'district',
                                           'ward', 'district_code', 'ward_code', 'county_code', 'constituency',
                                           'region', 'london_zone', 'middle_layer_super_output_area', 'postcode_area',
                                           'postcode_district']

df_postcode = (df_postcode.pipe(clean_names)
               .pipe(drop_columns, cols=postcode_columns)
               )

pp_index_columns = {'detached_index': 'pp_detached_index',
                    'semi_detached_index': 'pp_semi_detached_index',
                    'terraced_index': 'pp_terraced_index',
                    'flat_index': 'pp_flat_index',
                    }

pp_avg_columns = ['pp_detached_index', 'pp_semi_detached_index', 'pp_terraced_index', 'pp_flat_index']
avg_columns = ['detached_index','semi_detached_index','terraced_index','flat_index']


df_price_paid = (df_price_paid
                 .merge(df_postcode, on='postcode')
                 .merge(df_house_index, how='left', left_on=['district_code', 'month_year'],
                                                    right_on=['area_code', 'hpi_date'])
                 .rename(pp_index_columns,  axis='columns')
                 .pipe(mean_column, 'pp_average_index', pp_avg_columns)
                 .merge(df_house_index, how='left', left_on=['district_code','current_month'],
                                                    right_on=['area_code', 'date'])
                 .pipe(mean_column, 'average_index', avg_columns)
                 .adjust_price('T' ,'terraced_index', 'pp_terraced_index')
                 .adjust_price('S' ,'semi_detached_index', 'pp_semi_detached_index')
                 .adjust_price('D' ,'detached_index', 'pp_detached_index')
                 .adjust_price('F' ,'flat_index', 'pp_flat_index')
                 .adjust_price('O' ,'current_average_index', 'pp_current_average_index'))

df_price_paid = df_price_paid[df_price_paid['adjusted_price'].notnull()]
df_price_paid['adjusted_price'] = df_price_paid['adjusted_price'].astype(int)

df_price_paid.to_csv("../data/processed/processed.csv", index = False)
