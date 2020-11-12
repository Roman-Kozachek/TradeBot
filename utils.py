from datetime import datetime
import numpy as np
import pandas as pd
from pylab import plt
import keras
import tensorflow as tf
import math
from time import sleep
from sklearn.preprocessing import OneHotEncoder

data_path = "data/calgo/"
model_path = "models/"

fx_dateparse = lambda x: pd.datetime.strptime(str(x[:-4]), '%Y-%m-%d %H:%M:%S')

pd.options.mode.chained_assignment = None

def process_fx_file(path):
    """
    Processes raw forex file, saves it and returns valid dataframe
    :param path: path to forex file
    :return data: returns valid dataframe
    """
    data = pd.read_csv(path, delimiter=';', decimal=',', parse_dates=["Date"], date_parser=fx_dateparse)
    data.to_csv(path.split(".")[0] + "_pr." + path.split(".")[1], float_format='%.6f', index=False)
    
    return data

    
def upload_df(data):
    """
    Uploads dataframe 
    :param path: path to dataframe
    :return data: returns valid dataframe
    """
    if isinstance(data, str):
        data = pd.read_csv(data)
        data.Date = pd.to_datetime(data.Date, yearfirst=True, format='%Y-%m-%d %H:%M:%S')      
        return data
    elif isinstance(data, pd.DataFrame):  
        data["Hour"] = data.Date.apply(lambda x: x.hour)
        return data
    else:
        return
    
    
def add_ohe(data, validation_data, r_l, r_h, lit, column, func):

    a = np.arange(r_l, r_h)
    enc = OneHotEncoder()
    enc.fit(a.reshape(-1, 1))

    cols = [lit + str(i) for i in a]

    data[column] = data.Date.apply(lambda x: eval(func))
    ohe = enc.transform([[i] for i in data[column].to_numpy()]).toarray()
    for i, col in enumerate(cols):
        data[col] = ohe[:, i]
    data.drop(column, axis=1, inplace=True)

    validation_data[column] = validation_data.Date.apply(lambda x: eval(func))
    ohe = enc.transform([[i] for i in validation_data[column].to_numpy()]).toarray()
    for i, col in enumerate(cols):
        validation_data[col] = ohe[:, i]
    validation_data.drop(column, axis=1, inplace=True)
    
    return data, validation_data
    
    
def add_prophet_columns(data, prophet, freq="H"):
    data["ds"] = data.Date
    data["y"] = data.Close
    future = prophet.make_future_dataframe(periods=1, freq=freq)
    f = prophet.predict(future)
    
    data["trend"] = f.trend
    data["daily"] = f.daily
    data["weekly"] = f.weekly
    data["yearly"] = f.yearly
    data["yhat"] = np.roll(f.yhat.to_numpy(), -1)[:-1]
    data["yhat_lower"] = np.roll(f.yhat_lower.to_numpy(), -1)[:-1]
    data["yhat_upper"] = np.roll(f.yhat_upper.to_numpy(), -1)[:-1]

    
    data.drop(["ds", "y"], axis=1, inplace=True)
    
    return data

def graph_abs_mean(data):
    """
    This method shows absolute mean of hourly price changes per hour 
    :param data: dataframe with stock price history
    """
    data["Diff"]= data.Open.diff().apply(abs)
    data.Diff = data.Diff.fillna(0)
    data['Hour'] = data.Date.apply(lambda x: x.hour)
    diff_mean = data.groupby("Hour").mean().Diff
    
    x = diff_mean.index
    y = diff_mean
    plt.figure(figsize=(14,6))
    plt.bar(x, y)
    plt.title("Absolute mean of hourly price changes from %s to %s" % (data.iloc[0].Date.strftime("%Y-%m-%d"),\
                                                                       data.iloc[-1].Date.strftime("%Y-%m-%d")), fontsize=16)
    plt.show()

    
def graph_error_mean_per_hour(dataset, pred, column):
    data = dataset.iloc[dataset.shape[0]-len(pred):]
    data["Pred"] = np.around(pred,5)
    if column == "Diff":
        res = np.delete(data["Close"].to_numpy(), -1)
        res = np.insert(res, 0, 0)
        data["Pred"] = data["Pred"] + res
    data["AEM"] = np.abs(np.around(data.Close - data.Pred, 5))
    data['Hour'] = data.Date.apply(lambda x: x.hour)
    data["Diff"]= data.Close.diff().apply(abs).fillna(0)
    
    plt.figure(figsize=(14,6))
    plt.hist(data.where((data.Hour>5) & (data.Hour <20)).dropna()["AEM"], 150, density=True, range=(0,0.003))
    plt.show()
    
    diff_mean = data.groupby("Hour").mean().Diff
    x_ch = diff_mean.index
    y_ch = diff_mean
    
    diff_mean_aem = data.groupby("Hour").mean().AEM
    x = diff_mean_aem.index
    y = diff_mean_aem

    plt.figure(figsize=(14,6))
    plt.bar(x_ch, y_ch, color="green")
    plt.bar(x, y, color="r", alpha=0.7)
    plt.title("Absolute mean of error in predictions per hour compared to absolute mean of price changes", fontsize=16)
    plt.show()