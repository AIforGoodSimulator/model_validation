# model validation
This is a  repo for the Model Validation module of the AI for Good COVID-19 Simulator project.

Please take a look at model_metrics_test.py for a sample usage.

### Forecast errors

A forecast “error” is the difference between an observed value and its forecast. Here “error” does not mean a mistake, it means the unpredictable part of an observation. 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9bc81908-8460-40d9-8bce-c4fb34518d33/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9bc81908-8460-40d9-8bce-c4fb34518d33/Untitled.png)

Note that forecast errors are different from residuals in two ways. First, residuals are calculated on the *training* set while forecast errors are calculated on the *test* set. Second, residuals are based on *one-step* forecasts while forecast errors can involve *multi-step* forecasts.

We can measure forecast accuracy by summarising the forecast errors in different ways.

### Scale-dependent errors

The forecast errors are on the same scale as the data. Accuracy measures that are based only on etet are therefore scale-dependent and cannot be used to make comparisons between series that involve different units.

The two most commonly used scale-dependent measures are based on the absolute errors or squared errors: 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a0f9978c-de43-41ca-98c1-281c5f93fabd/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a0f9978c-de43-41ca-98c1-281c5f93fabd/Untitled.png)

When comparing forecast methods applied to a single time series, or to several time series with the same units, the MAE is popular as it is easy to both understand and compute. A forecast method that minimises the MAE will lead to forecasts of the median, while minimising the RMSE will lead to forecasts of the mean. Consequently, the RMSE is also widely used, despite being more difficult to interpret.

### Percentage errors

The percentage error is given by:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c41170dc-e5b6-4c2a-8653-5496cc4581bf/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c41170dc-e5b6-4c2a-8653-5496cc4581bf/Untitled.png)

Percentage errors have the advantage of being unit-free, and so are frequently used to compare forecast performances between data sets. The most commonly used measure is:Mean absolute percentage error: MAPE=mean(|pt|).

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/969e1d60-83d3-441d-9174-755ef663394e/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/969e1d60-83d3-441d-9174-755ef663394e/Untitled.png)


# How to Use Model Validation:

There are two Model Validation Modules developed:

1. Model Metrics (model_validation/model_metrics.py
2. Model Validation Plots (model_validation/model_validation_plot.py)

The sample calls for these two modules could be found in this test program:

[https://github.com/AIforGoodSimulator/model_validation/blob/master/model_validation_test.ipynb](https://github.com/AIforGoodSimulator/model_validation/blob/master/model_validation_test.ipynb)

We also have developed the following use cases for Model validation:

## (1) Compare age group cases against the overall population

Here is an sample for CM model : model_validation/validate_cm_output.py

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4055612f-d210-41b4-ad7c-8ef08ede897d/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4055612f-d210-41b4-ad7c-8ef08ede897d/Untitled.png)

The full CSV sample output could be found in this file in GitHub
[https://github.com/AIforGoodSimulator/model_validation/blob/master/cm_model_validation_metrics.csv](https://github.com/AIforGoodSimulator/model_validation/blob/master/cm_model_validation_metrics.csv)

## (2) Compare two different model outputs

Here is an sample for CM model : model_validation/compare_cm_output.py

The sample model validation output for comparing two different runs:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/156595cf-206f-4562-84c3-54c119f13825/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/156595cf-206f-4562-84c3-54c119f13825/Untitled.png)


### References for Model Validation

1) IHME COVID-19 Model Comparison Team. Predictive performance of international COVID-19 mortality forecasting models. MedRxiv. 14 July 2020. [https://www.medrxiv.org/content/10.1101/2020.07.13.20151233v4](https://www.medrxiv.org/content/10.1101/2020.07.13.20151233v4)

2) Forecasting: Principles and Practice, Rob J Hyndman and George Athanasopoulos.
