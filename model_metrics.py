# AI for Good COVID-19 Simulator
## Module: Model Validation
## Last updated: 9/11/2020
   
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
#MAPE is only available in latest Dev versio of sklearn
#from sklearn.metrics import mean_absolute_percentage_error

def mean_absolute_percentage_error(y_true, y_pred): 

    #Remove Null and Zeros 
    y_true= y_true.dropna() 
    y_pred = y_pred.dropna()
    rows = y_true!=0
    y_true = y_true[rows]
    y_pred = y_pred[rows]
        
    y_true, y_pred = np.array(y_true), np.array(y_pred)
 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Forecasting Model Metrics
def model_metrics(y,pred):
    metrics= pd.DataFrame(columns=['MAPE','RMSE','MSE', 'MeanAE', 'MedianAE','R2_Score', 'MSLE' ], index=[0])
    metrics['R2_Score'] = r2_score(y, pred)                          
    metrics['MAPE'] = mean_absolute_percentage_error(y, pred)
    metrics['MeanAE'] = mean_absolute_error(y, pred)
    metrics['MedianAE'] = median_absolute_error(y, pred)
    metrics['MSE'] = mean_squared_error(y, pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MSLE'] = mean_squared_log_error(y, pred)
    return metrics


