from typing import List, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
import dateutil.relativedelta
import transform_functions
from transform_functions import *
import logging
from scipy import stats
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
random_seed = 42
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
logging.info(f"Executing transform.py")

column_names = ['id', 'price', 'date', 'postcode',
                'type', 'new_build', 'land', 'primary_address',
                'secondary_address', 'street', 'locality', 'town_city',
                'district', 'county', 'ppd', 'record']

# Read in price paid data
df_price_paid = pd.read_csv("./data/processed/pp_sample_ext.csv",
                            parse_dates=['date'])

# Read in house price index
df_house_index = pd.read_csv("./data/raw/house_price_index.csv")

# Read in postcode data
df_postcode = pd.read_csv("./data/raw/postcodes.csv")

logging.info(f"Datasets Read")

county_rename = {'county_x': 'county'}

df_price_paid = (df_price_paid
                 .pipe(drop_columns, string='lmk_key')
                 .rename(county_rename, axis='columns')
                 .pipe(remove_duplicates)
                 .pipe(price_paid_process,
                 min=10000, max=5000000, number_of_months=4)
                 .drop(columns=['locality', 'town_city', 'district', 'county'])
                 )

df_house_index = (df_house_index.pipe(clean_names)
                  .pipe(drop_columns, string='change|price')
                  .pipe(col_to_dates, cols=['date'])
                  .rename({'date': 'hpi_date'}, axis='columns')
                  )

postcode_columns: List[Union[str, Any]] = ['postcode', 'latitude', 'longitude',
                                           'grid_ref', 'county', 'district',
                                           'ward', 'district_code',
                                           'ward_code', 'county_code',
                                           'constituency', 'region',
                                           'london_zone',
                                           'middle_layer_super_output_area',
                                           'postcode_area',
                                           'postcode_district', 
                                           'index_of_multiple_deprivation', 
                                           'quality', 'user_type', 
                                           'last_updated', 'nearest_station', 
                                           'distance_to_station', 
                                           'postcode_area', 'postcode_district', 
                                           'police_force', 'water_company', 
                                           'plus_code', 'average_income', 
                                           'sewage_company', 
                                           'travel_to_work_area', 'rural_urban',
                                           'altitude']

df_postcode = (df_postcode.pipe(clean_names)
               .loc[:, postcode_columns]
               )

pp_index_columns = {'detached_index': 'pp_detached_index',
                    'semi_detached_index': 'pp_semi_detached_index',
                    'terraced_index': 'pp_terraced_index',
                    'flat_index': 'pp_flat_index',
                    }

pp_avg_columns = ['pp_detached_index', 'pp_semi_detached_index',
                  'pp_terraced_index', 'pp_flat_index']
avg_columns = ['detached_index', 'semi_detached_index',
               'terraced_index', 'flat_index']

df_price_paid = (df_price_paid
                 .merge(df_postcode, on='postcode')
                 .merge(df_house_index, how='left',
                        left_on=['district_code', 'month_year'],
                        right_on=['area_code', 'hpi_date'])
                 .rename(pp_index_columns, axis='columns')
                 .pipe(mean_column, 'pp_average_index', pp_avg_columns)
                 .merge(df_house_index, how='left',
                        left_on=['district_code', 'current_month'],
                        right_on=['area_code', 'hpi_date'])
                 .pipe(mean_column, 'average_index', avg_columns)
                 .pipe(adjust_price, 'T', 'terraced_index',
                       'pp_terraced_index')
                .pipe(adjust_price, 'S', 'semi_detached_index',
                       'pp_semi_detached_index')
                 .pipe(adjust_price, 'D', 'detached_index',
                       'pp_detached_index')
                 .pipe(adjust_price, 'F', 'flat_index', 'pp_flat_index')
                 .pipe(adjust_price, 'O', 'average_index', 'pp_average_index'))

duplicate_list = ['date', 'postcode', 'type', 'new_build', 'land',
                  'primary_address', 'secondary_address', 'street',
                  'ppd', 'record', 'month_year', 'current_month',
                  'latitude', 'longitude', 'grid_ref', 'county',
                  'district', 'ward', 'district_code', 'ward_code',
                  'county_code', 'region',
                  'london_zone', 'middle_layer_super_output_area',
                  'postcode_area', 'postcode_district',
                  'hpi_date', 'region_name', 'area_code']

df_price_paid = (df_price_paid
                 .pipe(drop_columns, string='e_y|index')
                 .rename({'hpi_date_x': 'hpi_date',
                          'region_name_x': 'region_name',
                          'area_code_x': 'area_code'}, axis='columns')
                 .sort_values(by=['date'])
                 .drop_duplicates(subset=duplicate_list, keep="last"))

df_price_paid = df_price_paid[df_price_paid['adjusted_price'].notnull()]
df_price_paid['adjusted_price'] = df_price_paid['adjusted_price'].astype(int)
df_price_paid = df_price_paid[(np.abs(stats.zscore(df_price_paid["adjusted_price"])) < 3)]

df["london_zone"] = df["london_zone"].fillna(df["london_zone"].max() + 1, inplace=True)

logging.info("Completed processing")

# Lets drop the duplicates, keeping only the first instance.
df_price_paid.to_csv("./data/processed/processed.csv", index=False)

# Split the data - train, validation and test
train_set, test_set = train_test_split(df_price_paid,
                                       test_size=0.30,
                                       random_state=random_seed)

test_set, validation_set = train_test_split(test_set,
                                            test_size=0.20,
                                            random_state=random_seed)

logging.info(f"Training shape: {train_set.shape}")
logging.info(f"Validation shape: {validation_set.shape}")
logging.info(f"Test shape: {test_set.shape}")

# Save the split files
train_set.to_csv("./data/processed/train.csv", index=False)
validation_set.to_csv("./data/processed/validation.csv", index=False)
test_set.to_csv("./data/processed/test.csv", index=False)

logging.info(f"Transforming and splitting complete")