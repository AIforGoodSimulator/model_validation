from model_validation_metrics import model_validation_metrics

# population of the camp
population = 18700
baseline_output="CM_output_sample1.csv"
model_output="CM_output_sample2.csv"
model="CM"

save_output = "model_validation_metrics_cm.csv"
print("Calling model_validation_metrics for model="+model + " baseline_output: " +baseline_output + " model_output: " + model_output)
df_output = model_validation_metrics(population,model, baseline_output, model_output)
print(df_output)


print("Calling model_validation_metrics for model="+model + " baseline_output: " +baseline_output + " model_output: " + model_output)
df_output = model_validation_metrics(population,model, baseline_output, model_output, save_output)

# population of the camp
population = 18700
baseline_output="NM_output_sample1.csv"
model_output="NM_output_sample2.csv"
model="NM"
save_output = "model_validation_metrics_nm.csv"

print("Calling model_validation_metrics for model="+model + " baseline_output: " +baseline_output + " model_output: " + model_output)
df_output = model_validation_metrics(population,model, baseline_output, model_output, save_output)
print(df_output)


# population of the camp
population = 18700
baseline_output="ABM_output_sample1.csv"
model_output="ABM_output_sample2.csv"
model="ABM"
save_output = "model_validation_metrics_abm.csv"

print("Calling model_validation_metrics for model="+model + " baseline_output: " +baseline_output + " model_output: " + model_output)
df_output = model_validation_metrics(population,model, baseline_output, model_output, save_output)
print(df_output)