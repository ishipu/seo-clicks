import pandas as pd
pd.set_option("display.max_columns", None)
# import boto3
# s3 = boto3.client('s3')
import os
import numpy as np
import datetime
import json


from model.get_model import get_model
from preprocessing.input_data_preprocessing_q1 import (
    preprocess_data,
    preprocess_data_for_queries,
)
from preprocessing.find_valid_queries import find_valid_queries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from build_model.train_for_queries import train_model_for_queries
from preprocessing.files_handlers import load_data_query

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

running_locally = os.getenv("RUNNING_LOCAL") is not None
if running_locally:
    print("Running locally: ", running_locally)
    from pathlib import Path

    if Path.cwd().name != "deploy-python-ml":
        raise Exception(
            "Please run this from within the top directory of `deploy-python-ml`"
        )


def seo_clicks_rediction(n_future):
    print("Running Seo clicks model")

    WholetrainX, forecast_period_dates, scaler = preprocess_data(n_future)
    model = get_model("model/clicks_predictor_model.h5")
    forecast = model.predict(WholetrainX[-n_future:])
    # forecast_copies = np.repeat(forecast['predictions'], 3, axis=-1)
    forecast_copies = np.repeat(forecast, 3, axis=-1)

    y_pred_future = scaler.inverse_transform(forecast_copies)[:, 2]
    forecast_dates = []

    for time_i in forecast_period_dates:
        forecast_dates.append(time_i.date())

    df_forecast = pd.DataFrame(
        {"Date": np.array(forecast_dates), "Clicks": y_pred_future}
    )
    df_forecast["Clicks"] = df_forecast["Clicks"].astype("int")

    df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])

    # return df_forecast

    # save_s3_results(df, bucket=bucket)
    print(df_forecast)

    return True


def predict(model, data, trainX, scaler):
    n_future = 30
    forecast_period_dates = pd.date_range(
        list(data["Date"])[-1], periods=n_future, freq="1d"
    ).tolist()

    forecast = model.predict(trainX[-n_future:])
    forecast_copies = np.repeat(forecast, 3, axis=-1)

    y_pred_future = scaler.inverse_transform(forecast_copies)[:, 2]
    forecast_dates = []

    for time_i in forecast_period_dates:
        forecast_dates.append(time_i.date())

    df_forecast = pd.DataFrame(
        {"Date": np.array(forecast_dates), "Clicks": y_pred_future}
    )
    df_forecast["Clicks"] = df_forecast["Clicks"].astype("int")

    df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])

    return df_forecast


def query_clicks_prediction(query, valid_Queries, finalDf):
    print("2nd Part Is running")

    if query in valid_Queries:
        trainX, trainY, temp_new, scaler = preprocess_data_for_queries(
            finalDf, query
        )
        model = train_model_for_queries(trainX, trainY)
        df_forecast = predict(model, temp_new, trainX, scaler)
        print(df_forecast)

    else:
        print("Invalid query!!")


def handler(event, n_future):
    print("-- Running ML --")
    bucket = None
    key = os.getenv("FILENAME", "")

    print(event)
    if event['type'] == "forecast":
        n_future = event['n_future']
        seo_clicks_rediction(n_future)
        return "Success :) - 1"
    elif event['type'] == "query_forecast":
        query = input("Please enter avalid query from above cell: ")
        finalDf = load_data_query()
        valid_Queries = find_valid_queries(finalDf)
        query_clicks_prediction(query, valid_Queries, finalDf)

        return "Success :) - 2"


    return json.dumps({"code": 400, "message": "Hello Word From Lambda, But nothing ran"})
    # if running_locally:
    #     bucket=None
    #     key = os.getenv('FILENAME', '')
    # else:
    #     # s3 bucket
    #     bucket = event['Records'][0]['s3']['bucket']['name']
    #     # key = filename = s3 path
    #     key = event['Records'][0]['s3']['object']['key']

    # load the data
    # df = load_s3_data(filename=key, bucket=bucket)


if __name__ == "__main__":
    print("NLP ML App")
    handler(None, 15)