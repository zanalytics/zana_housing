import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import xgboost as xgb

pd.options.plotting.backend = 'matplotlib'
plt.interactive(True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_columns', 500)

keep_cols = ['id', 'adjusted_price', 'type', 'new_build', 'land', 'latitude', 'longitude']

df = pd.read_csv("./data/processed/processed.csv", usecols=keep_cols)

df = df[(df['adjusted_price'] <= 1000000)].reset_index(drop=True)
df = df.sample(frac=.10)

df_type = pd.get_dummies(df.type, prefix='type')
df_new_build = pd.get_dummies(df.new_build, prefix='new_build')
df_land = pd.get_dummies(df.land, prefix='land')

df = pd.concat([df, df_type, df_new_build, df_land], axis=1, sort=False)

df = df.drop(columns=['type', 'new_build', 'land'])

df = df[['id', 'latitude', 'longitude', 'type_D', 'type_T', 'adjusted_price']]

# df.to_csv("/Users/chris/Documents/data_science/experimentation/machine-learning-big-loop/data.csv")

X = df.iloc[:, 1:5]  # independent columns
y = df.iloc[:, -1]  # target column

data_dmatrix = xgb.DMatrix(data=X, label=y)

# split our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

xg_reg = xgb.XGBRegressor()

xg_reg.fit(X_train, y_train)

y_pred = xg_reg.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Error/Mean :', (metrics.mean_absolute_error(y_test, y_pred) / df["adjusted_price"].mean() * 100),
      '%')

import seaborn as sns

plt.figure(figsize=(20, 12))
ax = sns.regplot(x=y_test, y=y_pred)
ax.set_title('Actual vs Predicted')
ax.set_ylabel('Predicted cost of Job')
ax.set_xlabel('Actual cost of Job')

# train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)
