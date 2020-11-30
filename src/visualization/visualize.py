import matplotlib.pyplot as plt
import pandas as pd
import folium
from sklearn.model_selection import train_test_split
pd.options.plotting.backend = 'matplotlib'
plt.interactive(True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_columns', 500)

keep_cols = ['id', 'adjusted_price', 'type', 'new_build', 'land', 'latitude', 'longitude']

df = pd.read_csv("./data/processed/processed.csv", parse_dates=['date', 'month_year'])

df.loc['primary_address', 'secondary_address'].isna().sum()

luton = df[df["postcode"] == "LU5 4GY"]

df.describe()
df["type"].value_counts()
df["new_build"].value_counts()
df["land"].value_counts()
df["region"].value_counts()

df.hist(bins=50, figsize=(20,15))
plt.show()

df = df[(df['price'] <= 1000000) & (df['price'] >= 10000)].reset_index(drop=True)

df.hist(bins=10, figsize=(20,15))
plt.show()

df = df[(df['adjusted_price'] <= 1000000)].reset_index(drop=True)

df.hist(bins=10, figsize=(20,15))
plt.show()

df.describe()

df.describe(include='all')

# create a test set
train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

train_set.to_csv("./data/processed/train_set.csv", index=False)
test_set.to_csv("./data/processed/test_set.csv", index=False)


# plot data
fig, ax = plt.subplots(figsize=(20, 7))

# Histogram

train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

df.plot(kind="scatter", x="longitude", y="latitude",
               alpha=0.4, label="price paid locations",
               figsize=(20,14), c="adjusted_price",
                cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()


financial_crash = df['month_year'][df['month_year'] == '2007-12-01'].value_counts()[0]
uk_referendum = df['month_year'][df['month_year'] == '2016-06-01'].value_counts()[0]
first_lockdown = df['month_year'][df['month_year'] == '2020-03-01'].value_counts()[0]
fig2 = df['month_year'].value_counts().sort_index().plot.line()
fig2.add_annotation(x = '2016-06-23',
        y = uk_referendum,
        xref = 'x',
        yref = 'y',
        text = 'UK Referendum',
        showarrow = True,
        arrowhead = 7,
        ax = 0,
        ay = -40)
fig2.add_annotation(x = '2007-12-01',
        y = financial_crash,
        xref = 'x',
        yref = 'y',
        text = 'Financial Crash',
        showarrow = True,
        arrowhead = 7,
        ax = 0,
        ay = -40)
fig2.add_annotation(x = '2020-03-16',
        y = first_lockdown,
        xref = 'x',
        yref = 'y',
        text = 'First Lock Down',
        showarrow = True,
        arrowhead = 7,
        ax = 0,
        ay = -40)

fig2.update_layout(showlegend=False)
fig2.show()

group_by = ['month_year', 'type']

# use unstack()
(train_set.groupby(group_by)
    .median()['adjusted_price']
    .unstack()
    .plot(kind='line')
    .set_xlabel("x label")
    .set_ylabel("y label")
    .show())

# Run pipeline the prise paid data.
df_house_index = (df_house_index.pipe(clean_names)
                  .pipe(drop_columns, string='change|price')
                  .pipe(col_to_dates, cols=['date'])
                  .rename({'date': 'hpi_date'}, axis='columns')
                  )



fig = df['month_year'].value_counts().sort_index().plot.line()



df_small = df.iloc[:100000,:]
#view the dataset
print(df_small.head())
center = [1.372, 52.436]
map_kenya = folium.Map(location=center, zoom_start=1)
for index, franchise in df.iterrows():
    location = [franchise['latitude'], franchise['longitude']]
    folium.Marker(location, popup = f'Name:{franchise["postcode"]}\n Revenue($):{franchise["adjusted_price"]}').add_to(map_kenya)

# save map to html file
map_kenya.save('./src/visualization/index.html')

# Make an empty map
m = folium.Map(location=[1.372, 52.436], tiles="Mapbox Bright", zoom_start=2)

# I can add marker one by one on the map
for i in range(0,len(data)):
   folium.Circle(
      location=[data.iloc[i]['longitude'], data.iloc[i]['latitude']],
      popup=data.iloc[i]['postcode'],
      radius=data.iloc[i]['adjusted_price'],
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)

# Save it as html
m.save('./src/visualization/mymap.html')

import plotly.express as px
px.set_mapbox_access_token("pk.eyJ1IjoiY2hyaXNwM24iLCJhIjoiY2todDhzaHMwMDZwZTJ5b3l3N3AydzNmdiJ9.WCnWtndkd5Z2J_QxyTFkVA")
df = px.data.carshare()
fig = px.scatter_mapbox(df_small, lat="latitude", lon="longitude",     color="postcode", size="adjusted_price",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
fig.show()



# brew install geos
# pip3 install https://github.com/matplotlib/basemap/archive/master.zip
