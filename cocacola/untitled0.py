# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e-6056eaX-SuPn5xNqBnGDhv4JvQ1Mrd
"""

from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from google.colab import files
files.upload()

df = pd.read_csv('KO.csv')
df

actual_price = df.tail(1)
actual_price

df = df.head(len(df)-1)
df

days = list()
adj_close_prices = list()

df_days = df.loc[:, 'Date']
df_adj_close = df.loc[:, 'Adj Close']

for day in df_days:
  days.append([int(day.split('-')[2])])
#depenedent data sets
for adj_close_price in df_adj_close:
  adj_close_prices.append(float(adj_close_price))

days 
adj_close_prices

#creating 3 models
lin_svr = SVR(kernel='linear', C=1000.0)
lin_svr.fit(days, adj_close_prices)

poly_svr = SVR(kernel='poly', C=1000.0, degree=2) 
poly_svr.fit(days, adj_close_prices)

rbf_svr = SVR(kernel='rbf', C=1000.0 ,gamma=0.85)
rbf_svr.fit(days, adj_close_prices)

#plot the models
plt.figure(figsize=(32,16))
plt.scatter(days, adj_close_prices, color = 'black', label = 'Data')
plt.plot(days, rbf_svr.predict(days), color = 'green', label= 'RBF Model')
plt.plot(days, poly_svr.predict(days), color = 'orange', label= 'polynomial Model')
plt.plot(days, lin_svr.predict(days), color = 'blue', label= 'Linear Model')
plt.xlabel('Days')
plt.ylabel('Adj Close Price ($)')
plt.legend
plt.show

day = [[22]]

print ('The RBF SVR Predicted price: ', rbf_svr.predict(day))
print ('The Linear SVR Predicted price: ', lin_svr.predict(day))
print ('The Polyomial SVR Predicted price: ', poly_svr.predict(day))

print('The actual price ', actual_price['Adj Close'])