# %%
import pandas as pd
import numpy as np
from random import gauss
from pandas.plotting import autocorrelation_plot
import warnings
import itertools
from random import random
from datetime import datetime
import re
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
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
import seaborn as sns
plotsize = (13, 5)

# ignore warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.1f}'.format
%matplotlib inline

# %%
df_1 = pd.read_csv('data/June 21 - Sep 21 - Tesco.csv', usecols=['Date', 'Impressions', 'Average Position', 'Clicks']
                 ,parse_dates = ['Date'], low_memory=False, dayfirst=True)
df_2 = pd.read_csv('data/Sep 22 - Oct 2021 - Tesco.csv', usecols=['Date', 'Impressions', 'Average Position', 'Clicks']
                 ,parse_dates = ['Date'], low_memory=False, dayfirst=True)

# %%
df_2.head()

# %%
frames = [df_1, df_2]

finalDf = pd.concat(frames)
finalDf.info()


# %%
base = finalDf.groupby('Date').sum().reset_index()

base.info()

# %%
base = finalDf.groupby('Date').sum().reset_index()

base.info()

# %%
train_df = base[0:-30]
test_df  = base[-30:]

# %%
cols = list(base)[1:4]
cols

# %%
base[['Clicks']].plot(figsize=plotsize, title='Clicks')
base[['Impressions']].plot(figsize=plotsize, title='Impressions')
base[['Average Position']].plot(figsize=plotsize, title='Average Position');

# %%
df_for_training = train_df[cols].astype('float')
df_for_testing = test_df[cols].astype('float')


scaler = MinMaxScaler()
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)

# %%
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

# %%
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

# %%
train_df[cols]

# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# %%
model = Sequential() 
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences= False)) #,return_sequences= True))
# model.add(LSTM(32, activation='relu',return_sequences= False))
model.add(Dropout(0.25))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# %%
history = model.fit(trainX, trainY, epochs=150, batch_size=16,validation_data = (testX, testY), verbose=1)

# %%
plt.figure(figsize=(16,9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'])
plt.show()

# %% [markdown]
# # Retrain on the whole dataset

# %%
wholeDf_for_training = base[cols].astype('float')

scaler = MinMaxScaler()
wholeDf_for_training_scaled = scaler.fit_transform(wholeDf_for_training)

# %%


# %%
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

# %%
history = model.fit(WholetrainX, wholetrainY, epochs=1000, batch_size=32, verbose=1)

# %%
plt.figure(figsize=(16,9))
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend('train loss')
plt.show()

# %%
model.save("clicks_predictor_model.h5")

# %%
import tarfile
   
tarfile_name='clicks_predictor_model.tar.gz'

with tarfile.open(tarfile_name, mode='w:gz') as archive:
    archive.add('clicks_predictor_model.h5')

# %%
def predict(n_future):
    
    forecast_period_dates = pd.date_range(list(base['Date'])[-1], periods=n_future, freq='1d').tolist()
    forecast = model.predict(WholetrainX[-n_future:])
    forecast_copies = np .repeat(forecast, 3, axis=-1)
    
    print(forecast_copies)
    y_pred_future = scaler.inverse_transform(forecast_copies)[:,2]
    
    forecast_dates = []

    for time_i in forecast_period_dates:
        forecast_dates.append(time_i.date())

    df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Clicks':y_pred_future})
    df_forecast['Clicks'] = df_forecast['Clicks'].astype('int')

    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
    
    return df_forecast

# %%
df_forecast = predict(50)
df_forecast.head()

# %%
df_forecast.info()

# %%
type(WholetrainX)

# %%
WholetrainX.shape

# %%
df_forecast.to_csv('test_results.csv')

# %%
plt.figure(figsize=(16,9))

sns.lineplot(base['Date'], base['Clicks'], label="train")
sns.lineplot(df_forecast['Date'], df_forecast['Clicks'], label="predictions");

plt.title('Clicks Prediction for the next 30 days');

# %% [markdown]
# # Second Part of Queries

# %%
df_1 = pd.read_csv('data/June 21 - Sep 21 - Tesco.csv', parse_dates = ['Date'], low_memory=False, dayfirst=True)
df_2 = pd.read_csv('data/Sep 22 - Oct 2021 - Tesco.csv', parse_dates = ['Date'], low_memory=False, dayfirst=True)
#df_3 = pd.read_csv('3.csv', parse_dates = ['Date'], low_memory=False, dayfirst=True)

# %%
frames = [df_1, df_2]

finalDf = pd.concat(frames)
finalDf.info()

# %%
# 300 is our threshold
valid_Queries = (finalDf.Query.value_counts()>300)
valid_Queries = valid_Queries[valid_Queries].index
print(*valid_Queries, sep='\n')

# %%
query = input('Please enter avalid query from above cell: ')

if query in  valid_Queries:
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

    print('\nModel preparation...\n')
    model = Sequential() 
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences= False)) #,return_sequences= True))
    #model.add(LSTM(32, activation='relu',return_sequences= False))
    model.add(Dropout(0.25))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    print('\nModel fiting...\n')
    model.fit(trainX, trainY, epochs=50, batch_size=8, verbose=0)

    df_forecast = predict(model, temp_new, trainX)
    display(df_forecast)

    plt.figure(figsize=(16,9))

    sns.lineplot(temp_new['Date'], temp_new['Clicks'], label="train")
    sns.lineplot(df_forecast['Date'], df_forecast['Clicks'], label="predictions");

    plt.title('Clicks Prediction for the next 30 days');

else:
    
    print('Invalid query!!')

# %%
df_forecast.to_csv('test_results.csv')

# %%
df_5 = pd.read_csv('June 21 - Sep 21 - Tesco.csv')

# %%
df_5.head()

# %%
tf.__version__

# %% [markdown]
# # Deployment Of First Model

# %%
from sagemaker import get_execution_role
from sagemaker import Session
role = get_execution_role()
sess = Session()
bucket = sess.default_bucket()
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
tf_framework_version = tf.__version__
import h5py
import numpy as np
import os
import tarfile
from sagemaker.tensorflow.serving import Model

# %%
new_model = tf.keras.models.load_model('clicks_predictor_model.h5')

# %%
new_model.summary()

# %%
model_version = '1'
export_dir = 'export/models/' + model_version
tf.saved_model.save(new_model, export_dir)
model_archive = 'model.tar.gz'
with tarfile.open(model_archive, mode='w:gz') as archive:
    archive.add('export', recursive=True)
model_data = sess.upload_data(path=model_archive, key_prefix='model')

# %%
instance_type = 'ml.t2.medium'
sm_model = Model(model_data=model_data, framework_version="2.3", role=role)
predictor = sm_model.deploy(initial_instance_count=1, instance_type=instance_type)

# %%
predictor.endpoint

# %%
model_data


