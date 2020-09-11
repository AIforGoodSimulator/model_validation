# AI for Good COVID-19 Simulator
## Module: Model Validation
## Last updated: 9/11/2020
   
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error

# current stable version of sklearn does not MAPE
#from sklearn.metrics import mean_absolute_percentage_error


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Forecasting Model Metrics
def model_metrics(y,pred):
    metrics = pd.DataFrame({'r2_score':r2_score(y, pred),
                           }, index=[0])
    metrics['mean_absolute_error'] = mean_absolute_error(y, pred)
    metrics['median_absolute_error'] = median_absolute_error(y, pred)
    metrics['mse'] = mean_squared_error(y, pred)
    metrics['msle'] = mean_squared_log_error(y, pred)
    metrics['mape'] = mean_absolute_percentage_error(y, pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    return metrics


