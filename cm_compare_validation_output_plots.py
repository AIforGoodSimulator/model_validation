# AI for Good COVID-19 Simulator
# Module: Model Validation
# Last updated: 10/5/2020

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib.backends.backend_pdf import PdfPages

pd.set_option("display.max_rows", None, "display.max_columns", None)
plt.rcParams.update({'figure.max_open_warning': 0})

# population of the camp
population = 18700

baseline_output = "CM_output_sample1.csv"
model_output = "CM_output_sample2.csv"

print("Processing baseline output for model validation:" + baseline_output)
age_categories = pd.read_csv("age_categories.csv")['age'].to_list()
case_cols = pd.read_csv("cm_output_columns.csv")['columns'].to_list()

# Process Baseline First;
df_baseline = pd.read_csv(baseline_output)
df = df_baseline["Time"]
baseline_n_days = df.nunique()
baseline_n_rows = df.shape[0]
# num of simuls
baseline_n_simul = df[df == 0].count()

print("num of days = ", baseline_n_days)
print("num of simulations= ", baseline_n_simul)

# Get df for population
# Use this as the benchmark for the age group
cols_overall = ["Time"] + case_cols
df_baseline_all_simul = df_baseline[cols_overall]
df_baseline_all_sum = df_baseline_all_simul.groupby(['Time']).sum() * population
df_baseline_all = df_baseline_all_sum / baseline_n_simul
df_baseline_all_mean = df_baseline_all.mean()
df_baseline_all_std = df_baseline_all.std()

# Process Model Output and compare with baseline;
df_model = pd.read_csv(model_output)
df = df_model["Time"]
n_days = df.nunique()
n_rows = df.shape[0]
# num of simuls
n_simul = df[df == 0].count()

print("Processing Model output for model validation:" + model_output)
print("num of days = ", n_days)
print("num of simulations= ", n_simul)

# Get df for population
# Use this as the benchmark for the age group
cols_overall = ["Time"] + case_cols
df_model_all_simul = df_model[cols_overall]
df_model_all_sum = df_model_all_simul.groupby(['Time']).sum() * population
df_model_all = df_model_all_sum / n_simul
df_model_all_mean = df_model_all.mean()
df_model_all_std = df_model_all.std()

# no. of subplots
Tot = len(age_categories)
# no. of columns
Cols = 4
# no. of rows
Rows = Tot // Cols
Rows += Tot % Cols

x = [i+1 for i in range(n_days)]

i = 0  # used for getting id of figures
figs = []
for col in case_cols:
    n1 = 1  # used for position of subplots
    # different types of figures eg:series, histogram, distribution, auto-correlation, partial auto-correlation
    n_figures = 5
    for f in range(n_figures):
        figs.append(plt.figure((n_figures*i)+f, figsize=(10, 3)))
    for age in age_categories:
        cols = [col + ": " + age, "Time"]
        # baseline
        df_baseline_age_simul = df_baseline[cols]
        # Calculate averages for all simulations
        df_baseline_age_sum = df_baseline_age_simul.groupby(['Time']).sum() * population
        df_baseline_age = df_baseline_age_sum / baseline_n_simul

        # Model
        df_model_age_simul = df_model[cols]
        # Calculate averages for all simulations
        df_model_age_sum = df_model_age_simul.groupby(['Time']).sum() * population
        df_model_age = df_model_age_sum / n_simul

        col_age = col + ": " + age
        y = df_baseline_age[col_age]
        pred = df_model_age[col_age]

        # plotting the series
        figs[n_figures*i].suptitle('Comparison of ' + col + ' over different age groups')
        ax1 = figs[n_figures*i].add_subplot(Rows, Cols, n1)
        ax1.set_title("Age : " + age)
        ax1.grid(True)
        ax1.plot(x, y, label='Actual')
        ax1.plot(x, pred, label='Predicted')
        ax1.legend()
        ax1.set(xlabel='Days', ylabel=col)

        # plotting the histogram
        figs[n_figures*i + 1].suptitle('Histogram of ' + col + ' over different age groups')
        ax2 = figs[n_figures*i + 1].add_subplot(Rows, Cols, n1)
        ax2.set_title("Age : " + age)
        ax2.hist(y, ls='dashed', lw=3, fc=(0, 0, 0.7, 0.3), label='Actual')
        ax2.hist(pred, lw=3, fc=(0, 0, 0, 0.7),  label='Predicted')
        ax2.legend()

        # plotting the kde distribution
        figs[n_figures*i + 2].suptitle('Distribution of ' + col + ' over different age groups')
        ax3 = figs[n_figures*i + 2].add_subplot(Rows, Cols, n1)
        ax3.set_title("Age : " + age)
        y.plot(kind='kde', ax=ax3, title='Distribution', color='tab:blue', label='Actual')
        pred.plot(kind='kde', ax=ax3, title='Distribution', color='tab:red', label='Predicted')
        ax3.legend()

        # plotting the auto correlation
        figs[n_figures*i + 3].suptitle('ACF of ' + col + ' over different age groups')
        ax4 = figs[n_figures*i + 3].add_subplot(Rows, Cols, n1)
        plot_acf(pred, ax4, lags=30, title='Autocorrelation '+'Age : ' + age)

        # plotting the partial auto correlation
        figs[n_figures*i + 4].suptitle('PACF of ' + col + ' over different age groups')
        ax5 = figs[n_figures*i + 4].add_subplot(Rows, Cols, n1)
        plot_pacf(pred, ax5, lags=50, title='Partial Autocorrelation'+'Age : ' + age)

        print("col " + col + ' age ' + age)
        n1 = n1 + 1

    i = i + 1
plt.tight_layout()
plt.show()
