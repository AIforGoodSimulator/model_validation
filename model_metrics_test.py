# AI for Good COVID-19 Simulator
## Module: Model Validation
## Last updated: 9/11/2020

import numpy as np
import pandas as pd
from datetime import date, timedelta
from model_metrics import model_metrics
from model_validation_plot import plot_series, plot_acf_pacf, plot_actual_pred

pd.set_option("display.max_rows", None, "display.max_columns", None)

df = pd.DataFrame(columns=["date", "actual", "pred"])

today = date.today()
begin = today - timedelta(days=99)
df["date"] =pd.date_range(begin, today)
data = np.random.randint(100,150,size=(100,2))
df[["actual", "pred"]]=pd.DataFrame(data, columns=["actual", "pred"])
y=df["actual"]
pred=df["pred"]
results=model_metrics(y,pred)
print(results)
plot_series(y)
plot_series(pred)
plot_acf_pacf(pred)
plot_actual_pred(df)