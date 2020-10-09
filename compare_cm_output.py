# AI for Good COVID-19 Simulator
## Module: Model Validation
## Last updated: 10/5/2020

import pandas as pd
import numpy as np
from datetime import date, timedelta
from model_metrics import model_metrics
from model_validation_plot import plot_series, plot_acf_pacf, plot_actual_pred
pd.set_option("display.max_rows", None, "display.max_columns", None)

# population of the camp
population = 18700

baseline_output="CM_output_sample1.csv"
model_output="CM_output_sample2.csv"


age_categories=pd.read_csv("age_categories.csv")['age'].to_list()
case_cols=pd.read_csv("cm_output_columns.csv")['columns'].to_list()

# Process Baseline First;
df_baseline=pd.read_csv(baseline_output)
df = df_baseline["Time"]
baseline_n_days = df.nunique()
baseline_n_rows = df.shape[0]
# num of simuls
baseline_n_simul=df[df==0].count()

print("Processing baseline output for model validation:" + baseline_output)
print("num of days = ", baseline_n_days)
print("num of simulations= ", baseline_n_simul)

# Get df for population
# Use this as the benchmark for the age group
cols_overall = ["Time"] + case_cols
df_baseline_all_simul = df_baseline[cols_overall]
df_baseline_all_sum=df_baseline_all_simul.groupby(['Time']).sum()*population
df_baseline_all=df_baseline_all_sum/baseline_n_simul
df_baseline_all_mean=df_baseline_all.mean()
df_baseline_all_std=df_baseline_all.std()

cols_results = ["age", "case"]
df_model_metrics = pd.DataFrame(columns=cols_results) 


# Process Model Output and compare with baseline;
df_model=pd.read_csv(model_output)
df = df_model["Time"]
n_days = df.nunique()
n_rows = df.shape[0]
# num of simuls
n_simul=df[df==0].count()

print("Processing Model output for model validation:" + model_output)
print("num of days = ", n_days)
print("num of simulations= ", n_simul)

# Get df for population
# Use this as the benchmark for the age group
cols_overall = ["Time"] + case_cols
df_model_all_simul = df_model[cols_overall]
df_model_all_sum=df_model_all_simul.groupby(['Time']).sum()*population
df_model_all=df_model_all_sum/n_simul
df_model_all_mean=df_model_all.mean()
df_model_all_std=df_model_all.std()

cols_results = ["age", "case"]
df_model_metrics = pd.DataFrame(columns=cols_results) 


# Process for each age group:
for age in age_categories:

    # get columns for the age group
    cols = [ col + ": " + age for col in case_cols]
    cols.append("Time")

    #baseline
    df_baseline_age_simul = df_baseline[cols]    
    #Calculate averages for all simulations
    df_baseline_age_sum=df_baseline_age_simul.groupby(['Time']).sum()*population
    df_baseline_age=df_baseline_age_sum/baseline_n_simul
    df_baseline_age_mean=df_baseline_age.mean()
    df_baseline_age_std=df_baseline_age.std()
    
    #Model
    df_model_age_simul = df_model[cols]    
    #Calculate averages for all simulations
    df_model_age_sum=df_model_age_simul.groupby(['Time']).sum()*population
    df_model_age=df_model_age_sum/n_simul
    df_model_age_mean=df_model_age.mean()
    df_model_age_std=df_model_age.std()
    
    #Call Model Metrics for each case Col
    for col in case_cols:
        col_age = col + ": " + age
        y=df_baseline_age[col_age]
        pred=df_model_age[col_age]
        
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
        df_model_metrics=df_model_metrics.append(results)

print(df_model_metrics)
df_model_metrics.to_csv("cm_model_comparison_metrics.csv")      
print("Model Metrics by Age Group is saved in cm_model_comparison_metrics.csv")
