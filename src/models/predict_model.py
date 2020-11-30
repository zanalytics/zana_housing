import matplotlib.pyplot as plt
import pandas as pd
import folium
from sklearn.model_selection import train_test_split
pd.options.plotting.backend = 'matplotlib'
plt.interactive(True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_columns', 500)

keep_cols = ['id', 'adjusted_price', 'type', 'new_build', 'land', 'latitude', 'longitude']

df = pd.read_csv("./data/processed/processed.csv", usecols=keep_cols)

df = df[(df['adjusted_price'] <= 1000000)].reset_index(drop=True)

df_type = pd.get_dummies(df.type, prefix='type')
df_new_build = pd.get_dummies(df.new_build, prefix='new_build')
df_land = pd.get_dummies(df.land, prefix='land')

df = df.drop(columns=['type', 'new_build', 'land'])

df = pd.concat([df, df_type, df_new_build, df_land], axis=1, sort=False)

df = df[['id',
         'latitude',
         'longitude',
         'type_D',
         'type_F',
         'type_O',
         'type_S',
         'type_T',
         'new_build_N',
         'new_build_Y',
         'land_F',
         'land_L',
         'land_U',
         'adjusted_price']]

X = df.iloc[:,1:14]  #independent columns
y = df.iloc[:,-1]    #target column

#split our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
linear = GradientBoostingRegressor()
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print ('Mean Absolute Error/Mean :',(metrics.mean_absolute_error(y_test, y_pred) / df["adjusted_price"].mean()* 100),'%')

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor()

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print ('Mean Absolute Error/Mean :',(metrics.mean_absolute_error(y_test, y_pred) / df["adjusted_price"].mean()* 100),'%')

import seaborn as sns
plt.figure(figsize=(20, 12))
ax = sns.regplot(x=y_test, y=y_pred)
ax.set_title('Actual vs Predicted')
ax.set_ylabel('Predicted cost of Job')
ax.set_xlabel('Actual cost of Job')

# train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)


