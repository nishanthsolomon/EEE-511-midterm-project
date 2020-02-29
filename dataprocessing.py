import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import glob
import os
#######

csv_data=pd.read_csv('./dataset/Rainier_Weather.csv')

labels = ['Date','Temperature AVG','Relative Humidity AVG','Wind Speed Daily AVG']

data = csv_data[labels]
print(type(data))
print(data)
dict_dates = {}

for index, row in data.iterrows():
    dict_dates[row['Date']] = index
    break

def get_data_for_date(date):
    return data.iloc[dict_dates[date]]

#######
l=[]
path="./dataset/climbing_statisticsNew.csv"
#print(glob.glob(path))
for filename in glob.glob(path):
    with open(filename) as csvfile:
        csvdata=pd.read_csv(csvfile)
        print(csvdata)
        csvdata=csvdata.rename(columns={'ï»¿Date':'Date'})
        #datedata=csvdata['Date']
        df=csvdata.groupby('Date',sort=False).sum()
        print(df)
        l.append(df)
combined_csv=pd.concat(l,axis=0,ignore_index=True,sort=False)
combined_csv.to_csv(os.path.splitext("./dataset/CombinedData")[0]+ '_all.csv',index=False,sep=",")

t=[]
labels=['Attempted','Succeeded']
d=df[labels]
dfinal=data.merge(d,on="Date",how="right")
print(dfinal)
def convert_to_year(date_in_some_format):
    date_as_string = str(date_in_some_format)
    s=date_as_string.split('/')
    d = s[0]+'/'+s[1]
    return d
dfinal=dfinal.interpolate(method='linear',axis=0,limit_direction='forward')
dfinal=dfinal.interpolate(method='linear',axis=0,limit_direction='backward')
print(dfinal)
t.append(dfinal)
combined_csv=pd.concat(t,axis=0,ignore_index=True,sort=False)
combined_csv.to_csv(os.path.splitext("./dataset/FinalData")[0]+ '_all.csv',index=False,sep=",")