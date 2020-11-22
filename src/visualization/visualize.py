from matplotlib import pyplot as plt
import pandas as pd
pd.options.plotting.backend = "plotly"

keep_cols = ['id', 'price', 'date', 'postcode', 'type', 'new_build', 'land', 'month_year',
             'latitude', 'longitude', 'county', 'district', 'postcode_area', 'postcode_district',
             'region_name']

df = pd.read_csv("./data/processed/processed.csv", parse_dates=['date', 'month_year'], usecols=keep_cols)

# plot data
fig, ax = plt.subplots(figsize=(20, 10))
# use unstack()
data.groupby(['date', 'type']).count()['amount'].unstack().plot(ax=ax)

fig = last_sale['month_year'].value_counts().sort_index().plot.line()

fig = last_sale['adjusted_price'].plot.hist(bins=20)
fig.show()
