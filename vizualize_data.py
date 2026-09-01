import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output  # Correct import
import plotly.graph_objs as go

# Load data
csv_file = 'data\dec_4\9610PBUE.csv'
df = pd.read_csv(csv_file)

# Initialize Dash app
app = Dash(__name__)

# Layout with dropdown filter and graph
app.layout = html.Div([
    dcc.Dropdown(
        id='variable-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns],
        value=df.columns[0],  # Default value
        multi=True  # Allow selection of multiple variables
    ),
    dcc.Graph(id='line-graph')
])

# Callback to update the graph based on selected variables
@app.callback(
    Output('line-graph', 'figure'),  # Corrected Output syntax
    Input('variable-dropdown', 'value')  # Input for dropdown selection
)
def update_graph(selected_vars):
    if isinstance(selected_vars, str):  # Handle single selection case
        selected_vars = [selected_vars]

    fig = go.Figure()
    for var in selected_vars:
        fig.add_trace(go.Scatter(x=df.index, y=df[var], mode='lines', name=var))
    
    fig.update_layout(
        title="Interactive Plot of Selected Variables",
        xaxis_title="Index",
        yaxis_title="Value"
    )
    return fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
