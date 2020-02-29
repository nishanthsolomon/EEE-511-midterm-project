import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#from sklearn import preprocessing

import glob
import os


path="./dataset/FinalData_all.csv"

combined_csv=pd.read_csv(path)

y = combined_csv['Succeeded']/combined_csv['Attempted']

x = combined_csv[['Temperature AVG','Relative Humidity AVG','Wind Speed Daily AVG']]
#standardized_X = preprocessing.scale(x)
#standardized_Y = preprocessing.scale(y)
#normalized_X = preprocessing.normalize(x)
#normalized_Y = preprocessing.normalize(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

train = pd.concat([y_train, x_train], axis=1, sort=False)
test = pd.concat([y_test, x_test], axis=1, sort=False)

train.to_csv(os.path.splitext("./dataset/train")[0]+ '.csv',index=False,sep=",")
test.to_csv(os.path.splitext("./dataset/test")[0]+ '.csv',index=False,sep=",")
