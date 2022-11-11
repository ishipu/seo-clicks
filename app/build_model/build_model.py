
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pandas as pd
import numpy as np
from random import gauss
from pandas.plotting import autocorrelation_plot
from random import random
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Activation, Dropout
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf


cwd = Path.cwd().name
print("CWD: ", Path.cwd())

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if cwd != 'deploy-python-ml':
    print("Run this script from the project home directory (e.g. ~/home/whatever/deploy-python-ml)")
    print("On the terminal, `python build_model.py`")

# from app.preprocessing import clean_text

def __build_start_message():
    print("--Building the ML example model--")
    import time
    for i in range(3):
        print(i+1, '...')
        time.sleep(0.5)
    print("go!")


def load_data():
    df_1 = pd.read_csv('data/June 21 - Sep 21 - Tesco.csv', usecols=['Date', 'Impressions', 'Average Position', 'Clicks']
                 ,parse_dates = ['Date'], low_memory=False, dayfirst=True)
    df_2 = pd.read_csv('data/Sep 22 - Oct 2021 - Tesco.csv', usecols=['Date', 'Impressions', 'Average Position', 'Clicks']
                    ,parse_dates = ['Date'], low_memory=False, dayfirst=True)
    
    frames = [df_1, df_2]
    finalDf = pd.concat(frames)
    base = finalDf.groupby('Date').sum().reset_index()

    return base


def build():
    __build_start_message()

    # load data (ignore test)
    base = load_data()
    train_df = base[0:-30]
    test_df  = base[-30:]
    cols = list(base)[1:4]
    df_for_training = train_df[cols].astype('float')
    df_for_testing = test_df[cols].astype('float')

    scaler = MinMaxScaler()
    df_for_training_scaled = scaler.fit_transform(df_for_training)
    df_for_testing_scaled = scaler.transform(df_for_testing)


    testX = []
    testY = []

    n_future = 1
    n_past = 7

    for i in range(n_past, len(df_for_testing_scaled)- n_future+1):
        testX.append(df_for_testing_scaled[i - n_past:i, 0:df_for_testing.shape[1]])
        testY.append(df_for_testing_scaled[i + n_future - 1:i + n_future, 2])
        
    testX, testY = np.array(testX), np.array(testY)

    print('testX shape = {}.'.format(testX.shape))
    print('testY shape = {}.'.format(testY.shape))

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
    

    model = Sequential() 
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences= False)) #,return_sequences= True))
    model.add(Dropout(0.25))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    print("First Training")
    history = model.fit(trainX, trainY, epochs=150, batch_size=16,validation_data = (testX, testY), verbose=1)
    
    wholeDf_for_training = base[cols].astype('float')

    scaler = MinMaxScaler()
    wholeDf_for_training_scaled = scaler.fit_transform(wholeDf_for_training)

    WholetrainX = []
    wholetrainY = []

    n_future = 1
    n_past = 7

    for i in range(n_past, len(wholeDf_for_training_scaled)- n_future+1):
        WholetrainX.append(wholeDf_for_training_scaled[i - n_past:i, 0:wholeDf_for_training.shape[1]])
        wholetrainY.append(wholeDf_for_training_scaled[i + n_future - 1:i + n_future, 2])
        
    WholetrainX, wholetrainY = np.array(WholetrainX), np.array(wholetrainY)

    print('trainX shape = {}.'.format(WholetrainX.shape))
    print('trainY shape = {}.'.format(WholetrainX.shape))

    history = model.fit(WholetrainX, wholetrainY, epochs=1000, batch_size=32, verbose=1)

    model.save("model/clicks_predictor_model.h5")
    return "Success :)"


if __name__ == "__main__":
    build()

