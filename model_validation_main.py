from model_validation_metrics import model_validation_metrics
from model_validation_plots import model_validation_plots

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table

population = 18700
baseline_output="CM_output_sample1.csv"
model_output="CM_output_sample2.csv"
model="CM"

df_output = model_validation_metrics(population,model, baseline_output, model_output)

graph_divs = model_validation_plots(population,model, baseline_output, model_output)

layout_table = dash_table.DataTable(
    style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
    },
    id = "Model_Validation_Metrics",
    columns = [{"name": i, "id": i} for i in df_output.columns],
    data=df_output.to_dict('records'),
)

app = dash.Dash()
app.layout = html.Div([
                html.Div(layout_table),
                html.Div(graph_divs)
            ])
app.run_server(debug=True)
