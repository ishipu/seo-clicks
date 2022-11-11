from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pandas as pd
import numpy as np
from .files_handlers import load_data

def preprocess_data(n_future=5):
    finalDf = load_data()
    
    base = finalDf.groupby('Date').sum().reset_index()
    cols = list(base)[1:4]
    wholeDf_for_training = base[cols].astype('float')

    scaler = MinMaxScaler()
    wholeDf_for_training_scaled = scaler.fit_transform(wholeDf_for_training)
    
    WholetrainX = []
    n_f = 1
    n_p = 7

    for i in range(n_p, len(wholeDf_for_training_scaled)- n_f+1):
        WholetrainX.append(wholeDf_for_training_scaled[i - n_p:i, 0:wholeDf_for_training.shape[1]])
 
    WholetrainX = np.array(WholetrainX)
    
    
    forecast_period_dates = pd.date_range(list(base['Date'])[-1], periods=n_future, freq='1d').tolist()
    print(WholetrainX)
    print(WholetrainX.shape)

    return WholetrainX, forecast_period_dates, scaler


def preprocess_data_for_queries(finalDf, query):

    print('\nFiltering and Preparing the data...\n')
    cols = ['Impressions', 'Average Position', 'Clicks']

    finalDf[finalDf.Query == query].sort_values('Date')
    temp = finalDf[finalDf.Query == query].sort_values('Date')

    temp.set_index('Date', inplace=True)
    new_index = pd.date_range(temp.index.min(), temp.index.max())

    temp_new = temp.reindex(new_index)
    temp_new.interpolate(method='linear', inplace=True)
    temp_new.reset_index(inplace=True)

    temp_new.rename(columns = {'index':'Date'}, inplace = True)

    temp_new.drop(['Query', 'Site CTR'], axis=1, inplace=True)

    df_for_training = temp_new[cols].astype('float')

    print('\nData Scaling...\n')
    scaler = MinMaxScaler()
    df_for_training_scaled = scaler.fit_transform(df_for_training)

    print('\nData Speration (X, Y)...\n')
    trainX = []
    trainY = []

    n_future = 1
    n_past = 7

    for i in range(n_past, len(df_for_training_scaled)- n_future+1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 2])

    trainX, trainY = np.array(trainX), np.array(trainY)

    print('trainX shape = {}.'.format(trainX.shape))
    print('trainY shape = {}.'.format(trainY.shape))

    return trainX, trainY, temp_new, scaler
