# AI for Good COVID-19 Simulator
## Module: Model Validation
## Last updated: 11/3/2020

import pandas as pd
from model_metrics import model_metrics

def abm_add_asymptomatic_columns (df_abm, age_categories):
    df_abm['ASYMPTOMATIC'] = df_abm['ASYMPTOMATIC1'] + df_abm['ASYMPTOMATIC2']
    for age in age_categories:
        suffix = "_AGE" + age
        df_abm['ASYMPTOMATIC'+suffix] = df_abm['ASYMPTOMATIC1'+suffix] + df_abm['ASYMPTOMATIC2'+suffix]
    return df_abm

# population of the camp
population = 18700

cm_output = "CM_output_sample1.csv"
abm_output = "ABM_output_sample1.csv"

age_categories = pd.read_csv("age_categories.csv")['age'].to_list()
df_cols = pd.read_csv("output_columns.csv")
case_cols = df_cols['columns'].to_list()
case_cols_cm = df_cols['columns_cm'].to_list()
case_cols_abm = df_cols['columns_abm'].to_list()

# Loading CM output into a dataframe
df_cm = pd.read_csv(cm_output)
df_cm_time = df_cm["Time"]
cm_n_days = df_cm_time.nunique()
cm_n_rows = df_cm_time.shape[0]
# num of simuls
cm_n_simul = df_cm_time[df_cm_time == 0].count()

cols_overall_cm = ["Time"] + case_cols_cm
df_cm_all_simul = df_cm[cols_overall_cm]
df_cm_all_sum = df_cm_all_simul.groupby(['Time']).sum()*population
df_cm_all = df_cm_all_sum/cm_n_simul

# Loading ABM output into a dataframe
df_abm = pd.read_csv(abm_output)
df_abm = abm_add_asymptomatic_columns(df_abm, age_categories)
df_abm_time = df_abm["DAY"]
abm_n_days = df_abm_time.nunique()
abm_n_rows = df_abm_time.shape[0]
# num of simuls
abm_n_simul = df_abm_time[df_abm_time == 1].count()

cols_overall_abm = ["DAY"] + case_cols_abm
df_abm_all_simul = df_abm[cols_overall_abm]
df_abm_all_sum = df_abm_all_simul.groupby(['DAY']).sum()
df_abm_all = df_abm_all_sum/abm_n_simul

n_rows = min(cm_n_days, abm_n_days)

cols_results = ["age", "case"]
df_model_metrics = pd.DataFrame(columns=cols_results)

# removing hospitalised as abm doesnt have hospitalised data for different age categories
case_cols.remove('Hospitalised')
case_cols_cm.remove('Hospitalised')
case_cols_abm.remove('HOSPITALIZED')

# Process for each age group:
for age in age_categories:

    # get columns for the age group for cm
    cols_cm = [col + ": " + age for col in case_cols_cm]
    cols_cm.append("Time")

    # get columns for the age group for abm
    cols_abm = [col + "_AGE" + age for col in case_cols_abm]
    cols_abm.append("DAY")

    # baseline
    df_cm_age_simul = df_cm[cols_cm]
    # Calculate averages for all simulations
    df_cm_age_sum = df_cm_age_simul.groupby(['Time']).sum() * population
    df_cm_age = df_cm_age_sum / cm_n_simul

    # Model
    df_abm_age_simul = df_abm[cols_abm]
    # Calculate averages for all simulations
    df_abm_age_sum = df_abm_age_simul.groupby(['DAY']).sum()
    df_abm_age = df_abm_age_sum / abm_n_simul

    # Call Model Metrics for each case Col
    for col_ind in range(len(case_cols)):

        col_age_cm = case_cols_cm[col_ind] + ": " + age
        y1 = df_cm_age[col_age_cm][0:n_rows]

        col_age_abm = case_cols_abm[col_ind] + "_AGE" + age
        y2 = df_abm_age[col_age_abm][0:n_rows]

        y1.reset_index(inplace=True, drop=True)
        y2.reset_index(inplace=True, drop=True)

        results = model_metrics(y1, y2)
        results['age'] = age
        results['case'] = case_cols[col_ind]
        df_model_metrics = df_model_metrics.append(results, ignore_index=True)

df_model_metrics.reset_index(drop=True, inplace=True)

print(df_model_metrics)
df_model_metrics.to_csv("abm_cm_model_comparison_metrics.csv")
print("Model Metrics by Age Group is saved in abm_cm_model_comparison_metrics.csv")
