import pickle
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 1000)
pd.set_option('max_colwidth',80)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

file = open("new_tag_data.pkl", "rb")

data = pickle.load(file)


print(data.columns.tolist())
print(data)
print(set(data.Tag))

data.to_csv("Reddit_Comments.csv", sep=';', encoding='utf-8', index =False)