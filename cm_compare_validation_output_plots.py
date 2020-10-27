import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff

import dash
import dash_core_components as dcc
import dash_html_components as html

#To remove
import pprint as pp

pd.set_option("display.max_rows", None, "display.max_columns", None)

# population of the camp
population = 18700

baseline_output = "CM_output_sample1.csv"
model_output = "CM_output_sample2.csv"

# read example csvs
age_categories = pd.read_csv("age_categories.csv")['age'].to_list()
case_cols = pd.read_csv("cm_output_columns.csv")['columns'].to_list()

# Process baseline csv
df_baseline = pd.read_csv(baseline_output)
df = df_baseline["Time"]
baseline_n_days = df.nunique() # Count distinct observations over requested axis.
baseline_n_rows = df.shape[0]
# num of simuls
baseline_n_simul = df[df == 0].count()

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

# x is a list of days in data set starting at 1
# y_baseline is a pandas dataframe of the model output for baseline
# y_pred     is a pandas dataframe of the model output for baseline

# Need to produce a figure for each of the model outputs for each of the age categories. 
# Tempted to make age category a drop down selection.


# Generates the y data for the graph using the collumn name and df
def generate_y_data(model_output_df, col_age): 
    y_data = model_output_df[col_age]
    return y_data

# Generates a string of the column title we want from the df
def generate_col_age(col, age):
    col_age = f"{col}: {age}"
    return col_age

def generate_model_age_df(category, age, df_model, n_simul):
    cols = [category + ": " + age, "Time"]
    df_model_age_simul = df_model[cols]
    # Calculate averages for all simulations
    df_model_age_sum = df_model_age_simul.groupby(['Time']).sum() * population
    df_model_age = df_model_age_sum / n_simul   
    return df_model_age 

def gen_traces_to_show(traces, index):
        traces_to_show = traces
        actual_index = 2*index # as two plots per age category. first index must be 0
        traces_to_show[actual_index]     = True
        traces_to_show[(actual_index+1)] = True
        return traces_to_show

# Generates the drop down list for selecting age category graphs
def generate_drop_down_list(traces_to_show_all_false, age_categories):
    buttons_list = []
    traces_to_show_all_true = []
    for i in range (0,len(traces_to_show_all_false)):
        traces_to_show_all_true.append(not traces_to_show_all_false[i])

    #pp.pprint (traces_to_show_all_true)
    buttons_list.append(
        dict(label = "All",
        method = "update",
        args = [{"visible": traces_to_show_all_true},
            {"showlegend": True}])
    )

    # adds drop down option for each age category
    for i in range(len(age_categories)):
        traces_to_show_all_false_copy =traces_to_show_all_false
        traces_to_show = gen_traces_to_show(traces_to_show_all_false_copy, i)
        pp.pprint(traces_to_show_all_false)
        print("")
        buttons_list.append(
            dict(label = age_categories[i],
            method = "update",
            args = [{"visible": traces_to_show},
                {"showlegend": True}])
        )
    #pp.pprint(buttons_list)
    #print("")
    return buttons_list

# Plots series graph with drop down menu of each age category
def plot_series(x, df_baseline_age, df_model_age, category, age_categories, baseline_n_simul, n_simul):
    # plotting the series
    fig = go.Figure()
    traces_to_show = []
    for age in age_categories:

        df_baseline_age = generate_model_age_df(category, age, df_baseline, baseline_n_simul)
        df_model_age    = generate_model_age_df(category, age, df_model, n_simul)

        col_age = generate_col_age(category, age) # generated once for each age group for efficiency

        fig.add_trace(go.Scatter(x=x, y=generate_y_data(df_baseline_age, col_age),
            mode = "lines+markers",
            name = f"Baseline {age}"))

        fig.add_trace(go.Scatter(x=x, y=generate_y_data(df_model_age, col_age),
            mode = "lines+markers",
            name = f"Predicted {age}"))

        traces_to_show.append(False) # Probably a cleaner way of doing this but need an item in list for every trace with default value of True
        traces_to_show.append(False)

    # Add title and axis labels
    fig.update_layout(
        title=f"Comparison of {category} over different age groups",
        xaxis_title="Day",
        yaxis_title="Output",
        # Drop down menu
        #updatemenus=[go.layout.Updatemenu(
        #    active = 0,
        #    buttons=generate_drop_down_list(traces_to_show, age_categories)
        #)]      
        # Drop down menu proved too difficult within time constraints - didn't update properly
        annotations = [dict(x=0.5,
            y=-0.25,
            showarrow=False,
            text = "To isolate two traces double click on one in the legend and then single click on the second one to show."
        )]
    )

    return fig

def plot_histogram(x, df_baseline_age, df_model_age, category, age_categories, baseline_n_simul, n_simul):
    fig = go.Figure()

    traces_to_show = []
    for age in age_categories:
        col_age = generate_col_age(category, age)

        df_baseline_age = generate_model_age_df(category, age, df_baseline, baseline_n_simul)
        df_model_age    = generate_model_age_df(category, age, df_model, n_simul)

        # Add histogram for baseline
        fig.add_histogram(x=x, y=generate_y_data(df_baseline_age, col_age),
            name = f"Baseline {age}"
        )
        # Add histogram for predicted
        fig.add_histogram(x=x, y=generate_y_data(df_model_age, col_age),
            name = f"Predicted {age}"
        )
        
        traces_to_show.append(False) # Probably a cleaner way of doing this but need an item in list for every trace with default value of True
        traces_to_show.append(False)

    fig.update_layout(
        # Add title and axis labels
        title=f"Histogram of {category} over different age groups",
        xaxis_title="Day",
        yaxis_title="Output",
        # Overlay both histograms
        barmode="overlay",
        # Drop down menu
        #updatemenus=[go.layout.Updatemenu(
        #    active = 0,
        #    buttons=generate_drop_down_list(traces_to_show, age_categories)
        #)]      
        #  Drop down menu proved too difficult within time constraints - didn't update properly  
        annotations = [dict(x=0.5,
            y=-0.25,
            showarrow=False,
            text = "To isolate two traces double click on one in the legend and then single click on the second one to show."
        )]
    )

    #Reduce Opacity to see both histograms
    fig.update_traces(opacity=0.75)

    return fig

# Plots a graph of kde distribution
def plot_distribution(x, df_baseline_age, df_model_age, category, age_categories, baseline_n_simul, n_simul):

    
    traces_to_show = []
    data_to_plot = []
    group_labels = []
    for age in age_categories:

        df_baseline_age = generate_model_age_df(category, age, df_baseline, baseline_n_simul)
        df_model_age    = generate_model_age_df(category, age, df_model, n_simul)

        col_age = generate_col_age(category, age) # generated once for each age group for efficiency

        data_to_plot.append(generate_y_data(df_baseline_age, col_age))
        data_to_plot.append(generate_y_data(df_model_age, col_age))

        group_labels.append(f"Baseline {age}")
        group_labels.append(f"Predicted {age}")
        
        traces_to_show.append(False) # Probably a cleaner way of doing this but need an item in list for every trace with default value of True
        traces_to_show.append(False)
    
    fig = ff.create_distplot(data_to_plot, group_labels, show_hist=False)
    # Add title and axis labels
    fig.update_layout(
        title=f"Distribution of {category} over different age groups",
        xaxis_title="Day",
        yaxis_title="Output",
        # Drop down menu
        #updatemenus=[go.layout.Updatemenu(
        #    active = 0,
        #    buttons=generate_drop_down_list(traces_to_show, age_categories)
        #)]
        # Drop down menu proved too difficult within time constraints - didn't update properly
        annotations = [dict(x=0.5,
            y=-0.25,
            showarrow=False,
            text = "To isolate two traces double click on one in the legend and then single click on the second one to show."
        )]
    )
    return fig

case_cols = pd.read_csv("cm_output_columns.csv")['columns'].to_list()
x = [i+1 for i in range(n_days)]

graph_divs = []
for col in case_cols:
    graph_divs.append(html.Div(dcc.Graph(figure = plot_series(x, df_baseline, df_model, col, age_categories, baseline_n_simul, n_simul))))
    graph_divs.append(html.Div(dcc.Graph(figure = plot_histogram(x, df_baseline, df_model, col, age_categories, baseline_n_simul, n_simul))))
    graph_divs.append(html.Div(dcc.Graph(figure = plot_distribution(x, df_baseline, df_model, col, age_categories, baseline_n_simul, n_simul))))
app = dash.Dash()
app.layout = html.Div(graph_divs)
app.run_server(debug=True)
