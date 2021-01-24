random_seed = 42
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as no


# import some data to play with
df = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8"\
                "a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")

train_set, test_set = train_test_split(df,
                                   test_size=0.30, random_state=random_seed)
                                          
test_set, validation_set = train_test_split(test_set,
                                 test_size=0.20, random_state=random_seed)

print(df.shape)
print(train_set.shape)
print(test_set.shape)
print(validation_set.shape)
print(type(train_set))