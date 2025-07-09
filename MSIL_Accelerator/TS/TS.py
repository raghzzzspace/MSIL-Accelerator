import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

### -------- Feature Engineering -------- ###
def preprocess_time_series(df, datetime_col, target_col, freq='D'):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)
    df.set_index(datetime_col, inplace=True)
    df = df.asfreq(freq)
    df[target_col] = df[target_col].interpolate()
    df = df.reset_index()
    return df

### -------- Models -------- ###
class ARIMAModel:
    def __init__(self, order=(5, 1, 0)):
        self.order = order

    def fit(self, y):
        self.model = ARIMA(y, order=self.order).fit()

    def predict(self, steps):
        return self.model.forecast(steps=steps)

class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def fit(self, X, y):
        df = pd.DataFrame({'ds': X.squeeze(), 'y': y})
        self.model.fit(df)

    def predict(self, X):
        future = pd.DataFrame({'ds': X.squeeze()})
        forecast = self.model.predict(future)
        return forecast['yhat'].values

class LinearTimeSeriesModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

### -------- Utilities -------- ###
def get_model(name):
    if name == 'arima':
        return ARIMAModel()
    elif name == 'prophet':
        return ProphetModel()
    elif name == 'linear':
        return LinearTimeSeriesModel()
    else:
        raise ValueError("Unsupported model")

def split_time_series(df, target_col, test_size=30):
    test_size = min(test_size, len(df) // 3)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[target_col].values
    return X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]

def forecast_metrics(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
    }