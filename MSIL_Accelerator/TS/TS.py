import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

### -------- DATETIME + LAG FEATURES -------- ###
def add_datetime_features(df, datetime_col):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['year'] = df[datetime_col].dt.year
    df['month'] = df[datetime_col].dt.month
    df['day'] = df[datetime_col].dt.day
    df['dayofweek'] = df[datetime_col].dt.dayofweek
    df['weekofyear'] = df[datetime_col].dt.isocalendar().week
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[datetime_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[datetime_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[datetime_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[datetime_col].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[datetime_col].dt.is_year_start.astype(int)
    df['is_year_end'] = df[datetime_col].dt.is_year_end.astype(int)
    return df

def add_lag_features(df, target_col, lags=[1, 7, 14]):
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def preprocess_time_series(df, datetime_col, target_col, freq='D', lags=[1, 7, 14]):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)
    df.set_index(datetime_col, inplace=True)
    df = df.asfreq(freq)
    df[target_col] = df[target_col].interpolate()
    df = df.reset_index()
    df = add_datetime_features(df, datetime_col)
    df = add_lag_features(df, target_col, lags)
    df = df.dropna().reset_index(drop=True)
    return df

### -------- EDA -------- ###
def perform_time_series_eda(df, datetime_col, target_col, freq='D'):
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=datetime_col)
    df.set_index(datetime_col, inplace=True)
    df = df.asfreq(freq)
    
    print("\n📅 Time Range:", df.index.min(), "→", df.index.max())
    print("📉 Missing values in target:", df[target_col].isnull().sum())

    plt.figure(figsize=(12, 4))
    plt.plot(df[target_col], label='Time Series')
    plt.title("Time Series Line Plot")
    plt.xlabel("Time")
    plt.ylabel(target_col)
    plt.legend()
    plt.show()

    window = 7
    plt.figure(figsize=(12, 4))
    plt.plot(df[target_col], label='Original')
    plt.plot(df[target_col].rolling(window).mean(), label='Rolling Mean')
    plt.plot(df[target_col].rolling(window).std(), label='Rolling Std')
    plt.legend()
    plt.title("Rolling Stats")
    plt.show()

    try:
        decomposition = seasonal_decompose(df[target_col].dropna(), model='additive', period=window)
        decomposition.plot()
        plt.suptitle("Seasonal Decomposition", fontsize=16)
        plt.show()
    except Exception as e:
        print("⚠ Decomposition failed:", e)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(df[target_col].dropna(), ax=axes[0])
    plot_pacf(df[target_col].dropna(), ax=axes[1])
    plt.tight_layout()
    plt.show()

    print("\n🧪 Augmented Dickey-Fuller (ADF) Test:")
    result = adfuller(df[target_col].dropna())
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.4f}")
    for key, val in result[4].items():
        print(f"   {key}: {val:.4f}")
    if result[1] < 0.05:
        print("✅ Stationary")
    else:
        print("❌ Non-Stationary")

    return df

### -------- MODEL CLASSES -------- ###
class ARIMAModel(BaseEstimator, RegressorMixin):
    def __init__(self, order=(5, 1, 0), max_diff=2):
        self.order = order
        self.max_diff = max_diff
        self.model_fit = None
        self.differencing_order = 0
        self.last_value = None

    def _check_stationarity(self, y):
        series = y.copy()
        for d in range(self.max_diff + 1):
            if len(series.dropna()) < 10:
                raise ValueError(f"⚠ Series too short after differencing (d={d}). Minimum 10 points required.")
            result = adfuller(series.dropna())
            if result[1] < 0.05:
                self.differencing_order = d
                return y if d == 0 else y.diff(d).dropna()
            series = series.diff()
        raise ValueError("Series is non-stationary even after max differencing.")

    def fit(self, X, y):
        y = pd.Series(y).reset_index(drop=True)
        try:
            stationary_y = self._check_stationarity(y)
            self.last_value = y.iloc[-1]
            new_order = (self.order[0], self.differencing_order, self.order[2])
            self.model_fit = ARIMA(y, order=new_order).fit()
        except np.linalg.LinAlgError:
            print("⚠ LinAlgError: fallback to ARIMA(1,1,0)")
            self.model_fit = ARIMA(y, order=(1, 1, 0)).fit()
            self.order = (1, 1, 0)
            self.differencing_order = 1
        return self

    def predict(self, X):
        forecast = self.model_fit.forecast(steps=len(X))
        if self.differencing_order > 0:
            forecast = np.r_[self.last_value, forecast].cumsum()[1:]
        return forecast

    def get_params(self, deep=True):
        return {"order": self.order, "max_diff": self.max_diff}

    def set_params(self, **params):
        self.order = params.get("order", self.order)
        self.max_diff = params.get("max_diff", self.max_diff)
        return self

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

### -------- UTILITY FUNCTIONS -------- ###
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
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }

def hyperparams(model_name):
    if model_name == 'arima':
        return {'order': [(5,1,0), (2,1,2), (3,1,1)]}
    return {}

def visualize_results(search_obj, param_name='order'):
    if search_obj and hasattr(search_obj, 'cv_results_'):
        results = search_obj.cv_results_
        params = [str(p) for p in results['params']]
        scores = results['mean_test_score']
        plt.figure(figsize=(10, 4))
        sns.barplot(x=params, y=scores)
        plt.xticks(rotation=45)
        plt.title("Hyperparameter Tuning Results")
        plt.ylabel("Score")
        plt.show()

def tune_model(model, param_grid, X_train, y_train, scoring='neg_root_mean_squared_error', operation='GridSearchCV', cv=3):
    if operation == 'GridSearchCV':
        search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    elif operation == 'RandomizedSearchCV':
        search = RandomizedSearchCV(model, param_grid, n_iter=5, scoring=scoring, cv=cv, n_jobs=-1, random_state=42)
    elif operation == 'None':
        return model, model.get_params(), None
    else:
        raise ValueError("Invalid search operation")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search

def logger(model_str, params):
    print(f"\n🔧 Tuned Model: {model_str}")
    print("🔍 Best Params:", params)

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"📦 Model saved as {filename}")

