
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(series):
    model = ARIMA(series, order=(2,1,2)).fit()
    forecast = model.get_forecast(30)
    return forecast.predicted_mean, forecast.conf_int()
