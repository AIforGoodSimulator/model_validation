# AI for Good COVID-19 Simulator
## Module: Model Validation
## Last updated: 10/5/2020

import pandas as pd
import numpy as np
from datetime import date, timedelta
from model_metrics import model_metrics
from model_validation_plot import plot_series, plot_acf_pacf, plot_actual_pred
pd.set_option("display.max_rows", None, "display.max_columns", None)


model_output="CM_output_sample1.csv"
#model_output="CM_output_Moria.csv"
age_categories=pd.read_csv("age_categories.csv")['age'].to_list()
case_cols=pd.read_csv("cm_output_columns.csv")['columns'].to_list()
df_cm=pd.read_csv(model_output)

# population of the camp
population = 18700


#n_simul=1000
#n_days=200
# num of days 
df = df_cm["Time"]
n_days = df.nunique()
n_rows = df.shape[0]
# num of simuls
n_simul=df[df==0].count()

print("Processing CM output for model validation:" + model_output)
print("num of days = ", n_days)
print("num of simulations= ", n_simul)

# Get df for population
# Use this as the benchmark for the age group
cols_overall = ["Time"] + case_cols
df_all_simul = df_cm[cols_overall]
df_all_sum=df_all_simul.groupby(['Time']).sum()*population
df_all=df_all_sum/n_simul
df_all_mean=df_all.mean()
df_all_std=df_all.std()

cols_results = ["age", "case"]
df_model_metrics = pd.DataFrame(columns=cols_results) 

# Process for each age group:
for age in age_categories:

    # get columns for the age group
    cols = [ col + ": " + age for col in case_cols]
    cols.append("Time")
    df_age_simul = df_cm[cols]
    
    #Calculate averages for all simulations
    df_age_sum=df_age_simul.groupby(['Time']).sum()*population
    df_age=df_age_sum/n_simul
    df_age_mean=df_age.mean()
    df_age_std=df_age.std()
    
    #Call Model Metrics for each case Col
    for col in case_cols:
        col_age = col + ": " + age
        y=df_all[col]
        pred=df_age[col_age]
        
        #filter out nan or zero values of y;
        """
        # this is done inside model_metrics now
        y= y.dropna() 
        pred = pred.dropna()
        rows = y!=0
        y = y[rows]
        pred = pred[rows]
        """
        results=model_metrics(y,pred)
        results['age']= age
        results['case'] = col
        #print("Model Metrics for Age Group = {0}, Case={1}: ".format(age, col))
        #print(results)
        df_model_metrics=df_model_metrics.append(results, ignore_index=True)

df_model_metrics.reset_index(drop=True, inplace=True)

print(df_model_metrics)
df_model_metrics.to_csv("cm_model_validation_metrics.csv")      
print("Model Metrics by Age Group is saved in cm_model_validation_metrics.csv")
