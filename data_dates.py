import pandas as pd

csv_data=pd.read_csv('./dataset/Rainier_Weather.csv')

labels = ['Date','Temperature AVG','Relative Humidity AVG','Wind Speed Daily AVG']

data = csv_data[labels]

dict_dates = {}

for index, row in data.iterrows():
    dict_dates[row['Date']] = index
    break

date = '12/31/2015'

def get_data_for_date(date):
    return data.iloc[dict_dates[date]]
