#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:50:38 2021

@author: poulomi
"""
'''
This project is to find anomaly in stock price using the concept of Autoencoders. We use the GE stock price as our dataset.
'''

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Dropout, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Model
import seaborn as sns


data = pd.read_csv("GE.csv")

#convert pandas dataframe to numpy array
#data.values

df = data[['Date', 'Close']]

df['Date'] = pd.to_datetime(df['Date'])

sns.lineplot(x=df['Date'], y=df['Close'])
plt.savefig("Data.png")

print('Start date is: ', df['Date'].min())
print('End date is: ', df['Date'].max())

#dividing the data set into training and test sets, using location instead of 
train, test = df.loc[df['Date'] <= '2003-12-31'], df.loc[df['Date'] > '2003-12-31']

#normalizing the data set since LSTM is sensitive to large values
#standardscaler standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler = scaler.fit(train[['Close']])

#now we have a model that we can use to transform our train and test data 
train['Close'] = scaler.transform(train[['Close']])
test['Close'] = scaler.transform(test[['Close']])

#As required by LSTM networks we need to reshape an input data into n samples X timesteps
seq_size = 60 # number of timesteps to look back, larger sequences might improve forecasting

def sequence(x,y,seq_size=1):
    x_values=[]
    y_values=[]
    
    for i in range(len(x) - seq_size):
        #print(i)
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)


X_train, y_train = sequence(train[['Close']], train['Close'], seq_size)
X_test, y_test = sequence(test[['Close']], test['Close'], seq_size)


#Model architecture
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))

model.add(RepeatVector(X_train.shape[1]))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()


#fitting the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.savefig("Loss.png")


#######################################################################
#Anomaly is detected where the reconstruction error is large
#we can define a value beyond which we call it an anomaly
#let us look at MAE in training prediction

pred_train = model.predict(X_train)
train_mae = np.mean(np.abs(pred_train-X_train), axis=1)
plt.hist(train_mae, bins=30)

max_train_mae = 0.32

pred_test = model.predict(X_test)
test_mae = np.mean(np.abs(pred_test-X_test), axis=1)
plt.hist(test_mae, bins=30)


#Gathering all details in a dataframe to plot easily
anomaly = pd.DataFrame(test[seq_size:])
anomaly['test_mae'] = test_mae
anomaly['max_train_mae'] = max_train_mae
anomaly['anomaly'] = anomaly['test_mae'] > anomaly['max_train_mae']
anomaly['Close'] = test[seq_size:]['Close']


#plotting test_mae vs max_train_mae
sns.lineplot(x = anomaly['Date'], y = anomaly['test_mae'])
sns.lineplot(x = anomaly['Date'], y = anomaly['max_train_mae'])

actual_anomalies = anomaly.loc[anomaly['anomaly'] == True]

#plotting the actual anomalies
sns.lineplot(x=anomaly['Date'], y=scaler.inverse_transform(anomaly['Close']))
sns.scatterplot(x=actual_anomalies['Date'], y=scaler.inverse_transform(actual_anomalies['Close']), color='r')
sns.savefig("Final output.png")

















