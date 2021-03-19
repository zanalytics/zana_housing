from typing import List, Any, Union
import pandas as pd
from datetime import datetime
import dateutil.relativedelta
import logging

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
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace(".", '_')
    df.columns = df.columns.str.replace("/", '_')
    return df


def remove_duplicates(df):
    """
        Removes duplicates and only takes rows with the postcode.

        Parameters:
            - df : dataframe
                The price paid dataset.

        Returns:
            - filtered dataframe.
    """
    df = df.drop_duplicates(subset=df.columns[1:], keep="first")
    df = df[df['postcode'].notnull()]
    return df


def price_paid_process(df, min, max, number_of_months):
    """
        filters by price, creates a month-year and a current-month column.

        Parameters:
            - df : dataframe
                Price paid dataframe.
            - min : integer
                Minimum house price
            - max : integer
                Maximum house price
            - number_of_months : integer
                Number of months to set your current_month
                If data is not up to date

        Returns:
            - filtered dataframe with month_year and current_month.
    """
    df = df[(df['price'] <= max) & (df['price'] >= min)].reset_index(drop=True)
    df['month_year'] = df['date'].astype('datetime64[M]')
    this_month = pd.to_datetime(datetime.today().date().replace(day=1))
    df['current_month'] = this_month - dateutil.relativedelta.relativedelta(
                                                    months=number_of_months)
    return df


def drop_columns(df, string):
    """
        Drop the columns using a pattern string

        Parameters:
            - df : dataframe
                Dataframe
            - string : string
                String containing the pattern. The columns will be dropped.

        Returns:
            - Dataframe with reduced number of columns.
    """
    df = df.drop(columns=df.columns[df.columns.str.contains(pat=string)])
    return df


def col_to_dates(df, cols):
    """
        Mutate columns to dates

        Parameters:
            - df : dataframe
                Dataframe
            - cols : list
                Column names you want to change to datetime.

        Returns:
            - Dataframe with columns specified as datetime.
    """
    df[cols] = df[cols].apply(pd.to_datetime)
    return df


def mean_column(df, column_name, avg_list):
    """
        Creates new column of with the mean of multiple columns.

        Parameters:
            - df : dataframe
                Dataframe
            - column_name : string
                New column name
            - avg_list : list
                List of columns to average


        Returns:
            - writes the file to destination.
    """
    df[column_name] = df[avg_list].mean(axis=1)
    return df


def adjust_price(df, house_type, current_index, pp_index):
    """
        House Price Index (HPI) calculation

        Parameters:
            - df : datfarame
                Dataframe
            - house_type : string
                House type as index
            - current_index : string
                Column to use as index. The latest HPI.
            - pp_index : string
                Column to use for the index. The HPI when house was purchased.

        Returns:
            - writes the file to destination.
    """
    df.loc[df['type'] == house_type, 'adjusted_price'] = (
        df[current_index] / df[pp_index]) * df['price']
    return df
