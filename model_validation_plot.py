# AI for Good COVID-19 Simulator
## Module: Model Validation
## Last updated: 9/11/2020
   

# current stable version of sklearn does not MAPE
#from sklearn.metrics import mean_absolute_percentage_error

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt


#plotting ACF and PACF


# ACF and PACF plots
# Review the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots

def plot_series(x):

    fig=plt.figure(1)
    plt.title('Series Histogram')
    plt.subplot(211)
    x.hist()
    plt.subplot(212)
    x.plot(kind='kde')
    plt.show()
    return fig

def plot_acf_pacf(x):

    fig=plt.figure()
    plt.subplot(211)
    plt.title('ACF')
    plot_acf(x, ax=plt.gca(), lags = 30)
    plt.subplot(212)
    plt.title('PACF')
    plot_pacf(x, ax=plt.gca(), lags = 30)
    plt.show()
    return fig



def plot_actual_pred(df):

    fig, ax = plt.subplots(figsize=(10, 6))
    # Same as above
    ax.set_xlabel('Date')
    ax.set_ylabel('# of cases')
    ax.set_title('actual vs prediction')
    ax.grid(True)

    # Plotting on the first y-axis
    ax.plot(df['date'], df['actual'], color='tab:blue', label='Actual')
    ax.plot(df['date'], df['pred'], color='tab:orange', linestyle='--', label='Prediction')
    ax.legend(loc='upper left');

    return ax
