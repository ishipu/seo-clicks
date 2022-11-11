import os
import pandas as pd
import numpy as np
import io
import datetime
# import boto3
# s3 = boto3.client('s3')

def load_data():
    df_1 = pd.read_csv('data/June 21 - Sep 21 - Tesco.csv', usecols=['Date', 'Impressions', 'Average Position', 'Clicks']
                 ,parse_dates = ['Date'], low_memory=False, dayfirst=True)
    df_2 = pd.read_csv('data/Sep 22 - Oct 2021 - Tesco.csv', usecols=['Date', 'Impressions', 'Average Position', 'Clicks']
                    ,parse_dates = ['Date'], low_memory=False, dayfirst=True)
    
    frames = [df_1, df_2]
    finalDf = pd.concat(frames)

    return finalDf

def load_data_query():
    df_1 = pd.read_csv('data/June 21 - Sep 21 - Tesco.csv', parse_dates = ['Date'], low_memory=False, dayfirst=True)
    df_2 = pd.read_csv('data/Sep 22 - Oct 2021 - Tesco.csv', parse_dates = ['Date'], low_memory=False, dayfirst=True)
    frames = [df_1, df_2]
    finalDf = pd.concat(frames)

    return finalDf

def load_s3_data(filename, bucket=None):
    print("-- Loading s3 data --")
    # load test data
    key = filename
    print("Requesting object from Bucket: {} and Key: {}".format(bucket, key))
    obj = s3.get_object(Bucket=bucket, Key=key)
    print("Got object from S3")
    data = io.StringIO(obj['Body'].read().decode('utf-8')) 
    return pd.read_csv(data)


def save_s3_results(df, bucket=None, key='predictions.csv'):
    now_dt = str(datetime.datetime.now())
    s3_client = boto3.client('s3')
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    print(f'file written to {bucket} --{key}')

    return True
