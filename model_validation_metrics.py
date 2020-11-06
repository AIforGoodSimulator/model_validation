# AI for Good COVID-19 Simulator
## Module: Model Validation
## Last updated: 11/5/2020

import pandas as pd
import numpy as np
from datetime import date, timedelta
from model_metrics import model_metrics

#from model_validation_plot import plot_series, plot_acf_pacf, plot_actual_pred
#pd.set_option("display.max_rows", None, "display.max_columns", None)

# Required the following arguments:
#   - population: camp population in integer
#   - model: model name such as CM, NM or ABM
#   - baseline_output:  csv file name for baseline output
#   - model_output: csv file name for model output
#   - optional: save_outout: csv file name for saving model validation output

def model_validation_metrics(population:int, model:str, baseline_output:str, model_output:str, save_output=""):

    #baseline_output="CM_output_sample1.csv"
    #model_output="CM_output_sample2.csv"

    print("input arguments: model=" + model + " baseline_output: " + baseline_output + " model_output: " + model_output)

    age_categories=pd.read_csv("age_categories.csv")['age'].to_list()
    time_col = "Time"
    case_cols=[]
    cols_results = ["age", "case"]
    df_model_metrics = pd.DataFrame(columns=cols_results)

    if model.upper() =="CM":
        print("validating a compartment model ...")
        case_cols=pd.read_csv("cm_output_columns.csv")['columns'].to_list()

        # Process Baseline First;
        df_baseline=pd.read_csv(baseline_output)
        df = df_baseline[time_col]
        baseline_n_days = df.nunique()
        baseline_n_rows = df.shape[0]
        # num of simuls
        baseline_n_simul=df[df==0].count()
    
        print("Processing baseline output for model validation:" + baseline_output)
        print("num of days = ", baseline_n_days)
        print("num of simulations= ", baseline_n_simul)
    
        # Get df for population
        # Use this as the benchmark for the age group
        cols_overall = [time_col] + case_cols
        df_baseline_all_simul = df_baseline[cols_overall]
        df_baseline_all_sum=df_baseline_all_simul.groupby([time_col]).sum()*population
        df_baseline_all=df_baseline_all_sum/baseline_n_simul
        df_baseline_all_mean=df_baseline_all.mean()
        df_baseline_all_std=df_baseline_all.std()
    
        cols_results = ["age", "case"]
        df_model_metrics = pd.DataFrame(columns=cols_results)
    
    
        # Process Model Output and compare with baseline;
        df_model=pd.read_csv(model_output)
        df = df_model[time_col]
        n_days = df.nunique()
        n_rows = df.shape[0]
        # num of simuls
        n_simul=df[df==0].count()
    
        print("Processing Model output for model validation:" + model_output)
        print("num of days = ", n_days)
        print("num of simulations= ", n_simul)
    
        # Get df for population
        # Use this as the benchmark for the age group
        cols_overall = [time_col] + case_cols
        df_model_all_simul = df_model[cols_overall]
        df_model_all_sum=df_model_all_simul.groupby([time_col]).sum()*population
        df_model_all=df_model_all_sum/n_simul
        df_model_all_mean=df_model_all.mean()
        df_model_all_std=df_model_all.std()

        # Process for each age group:
        for age in age_categories:
    
            # get columns for the age group
            cols = [ col + ": " + age for col in case_cols]
            cols.append(time_col)
    
            #baseline
            df_baseline_age_simul = df_baseline[cols]
            #Calculate averages for all simulations
            df_baseline_age_sum=df_baseline_age_simul.groupby([time_col]).sum()*population
            df_baseline_age=df_baseline_age_sum/baseline_n_simul
            df_baseline_age_mean=df_baseline_age.mean()
            df_baseline_age_std=df_baseline_age.std()
    
            #Model
            df_model_age_simul = df_model[cols]
    
            #Calculate averages for all simulations
            df_model_age_sum=df_model_age_simul.groupby([time_col]).sum()*population
            df_model_age=df_model_age_sum/n_simul
            df_model_age_mean=df_model_age.mean()
            df_model_age_std=df_model_age.std()
    
            #Call Model Metrics for each case Col
            for col in case_cols:
                col_age = col + ": " + age
                y=df_baseline_age[col_age]
                pred=df_model_age[col_age]
    
                # filter out nan or zero values of y;
                rows = y > 0
                y = y[rows]
                pred = pred[rows]
                y = y.dropna()
                pred = pred.dropna()
                results=model_metrics(y,pred)
                results['age']= age
                results['case'] = col
                #print("Model Metrics for Age Group = {0}, Case={1}: ".format(age, col))
                #print(results)
                df_model_metrics=df_model_metrics.append(results, ignore_index=True)


    elif model.upper() == "ABM":
        print("validating an agent-based model ...")
        time_col="DAY"
        case_cols = pd.read_csv("abm_output_columns.csv")['columns'].to_list()

        # Process Baseline First;
        df_baseline=pd.read_csv(baseline_output)
        df = df_baseline[time_col]
        baseline_n_days = df.nunique()
        baseline_n_rows = df.shape[0]
        # num of simuls
        baseline_n_simul=df[df==1].count()
        
        print("Processing baseline output for model validation:" + baseline_output)
        print("num of days = ", baseline_n_days)
        print("num of simulations= ", baseline_n_simul)
        
        # Get df for population
        # Use this as the benchmark for the age group
        cols_overall = [time_col] + case_cols
        df_baseline_all_simul = df_baseline[cols_overall]
        df_baseline_all_sum=df_baseline_all_simul.groupby([time_col]).sum()
        #df_baseline_all=df_baseline_all_sum/baseline_n_simul
        #df_baseline_all_mean=df_baseline_all.mean()
        #df_baseline_all_std=df_baseline_all.std()
        df_baseline_all_mean=df_baseline_all_sum/baseline_n_simul
        df_baseline_all_std=df_baseline_all_simul.groupby([time_col]).std()
        
        cols_results = ["age", "case"]
        df_model_metrics = pd.DataFrame(columns=cols_results) 
        
        
        # Process Model Output and compare with baseline;
        df_model=pd.read_csv(model_output)
        df = df_model[time_col]
        n_days = df.nunique()
        n_rows = df.shape[0]
        # num of simuls
        n_simul=df[df==1].count()
        
        print("Processing Model output for model validation:" + model_output)
        print("num of days = ", n_days)
        print("num of simulations= ", n_simul)
        
        # Get df for population
        # Use this as the benchmark for the age group
        cols_overall = [time_col] + case_cols
        df_model_all_simul = df_model[cols_overall]
        df_model_all_sum=df_model_all_simul.groupby(['DAY']).sum()
        #df_model_all=df_model_all_sum/n_simul
        #df_model_all_mean=df_model_all.mean()
        #df_model_all_std=df_model_all.std()
        df_model_all_mean=df_model_all_sum/n_simul
        df_model_all_std=df_model_all_simul.groupby(['DAY']).std()

        
        case_cols.remove('HOSPITALIZED')
        # Process for each age group:
        
        # Process for each age group:
        for age in age_categories:
        
            # get columns for the age group
        
            cols = [col + "_AGE" + age for col in case_cols]
            cols.append(time_col)
        
            # baseline
            df_baseline_age_simul = df_baseline[cols]
            # Calculate averages for all simulations
            df_baseline_age_sum = df_baseline_age_simul.groupby(['DAY']).sum()
            df_baseline_age = df_baseline_age_sum / baseline_n_simul
            # df_baseline_age_mean=df_baseline_age.mean()
            # df_baseline_age_std=df_baseline_age.std()
            df_baseline_age_mean = df_baseline_age_sum / baseline_n_simul
            df_baseline_age_std = df_baseline_age_simul.groupby(['DAY']).std()
        
            # Model
            df_model_age_simul = df_model[cols]
            # Calculate averages for all simulations
            df_model_age_sum = df_model_age_simul.groupby(['DAY']).sum()
            df_model_age = df_model_age_sum / n_simul
            # df_model_age_mean=df_model_age.mean()
            # df_model_age_std=df_model_age.std()
            df_model_age_mean = df_model_age_sum / n_simul
            df_model_age_std = df_model_age_simul.groupby(['DAY']).std
        
            # Call Model Metrics for each case Col
            for col in case_cols:
                col_age = col + "_AGE" + age
                y = df_baseline_age[col_age]
                pred = df_model_age[col_age]
                # filter out nan or zero values of y;
                rows = y > 0
                y = y[rows]
                pred = pred[rows]
                y = y.dropna()
                pred = pred.dropna()
                ln = min(len(y), len(pred))
                results = model_metrics(y.iloc[0:ln], pred.iloc[0:ln])
                results['age'] = age
                results['case'] = col
                # print("Model Metrics for Age Group = {0}, Case={1}: ".format(age, col))
                # print(results)
                df_model_metrics = df_model_metrics.append(results, ignore_index=True)


    elif model.upper() == "NM":
        print("validating a network model ...")

        case_cols = pd.read_csv("nm_output_columns.csv")['columns'].to_list()

        # Process Baseline First;
        df_baseline = pd.read_csv(baseline_output)
        df = df_baseline[time_col]
        baseline_n_days = df.nunique()
        baseline_n_rows = df.shape[0]
        # num of simuls
        baseline_n_simul = df[df == 1].count()

        print("Processing baseline output for model validation:" + baseline_output)
        print("num of days = ", baseline_n_days)
        print("num of simulations= ", baseline_n_simul)

        # Get df for population
        # Use this as the benchmark for the age group
        cols_overall = [time_col] + case_cols
        df_baseline_all_simul = df_baseline[cols_overall]
        df_baseline_all_sum = df_baseline_all_simul.groupby(['Time']).sum()
        df_baseline_all = df_baseline_all_sum / baseline_n_simul
        df_baseline_all_mean = df_baseline_all.mean()
        df_baseline_all_std = df_baseline_all.std()

        cols_results = ["case"]
        df_model_metrics = pd.DataFrame(columns=cols_results)

        # Process Model Output and compare with baseline;
        df_model = pd.read_csv(model_output)
        df = df_model[time_col]
        n_days = df.nunique()
        n_rows = df.shape[0]
        # num of simuls
        n_simul = df[df == 1].count()

        print("Processing Model output for model validation:" + model_output)
        print("num of days = ", n_days)
        print("num of simulations= ", n_simul)

        # Get df for population
        # Use this as the benchmark for the age group
        cols_overall = [time_col] + case_cols
        df_model_all_simul = df_model[cols_overall]
        df_model_all_sum = df_model_all_simul.groupby(['Time']).sum()
        df_model_all = df_model_all_sum / n_simul
        df_model_all_mean = df_model_all.mean()
        df_model_all_std = df_model_all.std()

        cols_results = ["case"]
        df_model_metrics = pd.DataFrame(columns=cols_results)

        # Call Model Metrics for each case Col
        for col in case_cols:
            y = df_baseline_all[col]
            pred = df_model_all[col]
            # filter out nan or zero values of y;
            rows = y > 0
            y = y[rows]
            pred = pred[rows]
            y = y.dropna()
            pred = pred.dropna()
            results = model_metrics(y, pred)
            results['case'] = col
            df_model_metrics = df_model_metrics.append(results, ignore_index=True)


    df_model_metrics.reset_index(drop=True, inplace=True)

    if save_output:
        df_model_metrics.to_csv(save_output)
        print("Model Validation Metrics is saved in " + save_output)

    return (df_model_metrics)