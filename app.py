import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import pickle

import dash_core_components as dcc
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

df = pd.read_csv("data/cleaned_beijing.csv", parse_dates=["tradeTime"])

district_split_fig = px.line(df, x='district', y='totalPrice', color='district')

app.layout = html.Div(className = 'row',
children=[
    html.Div(className='four columns div-user-controls',
    children = [
        html.Div(children = [
            html.H2('House price prediction app'),
            html.P('''Predicting the sales price of houses in beijing'''),
            html.P('''Using gradient boosted regression model in sklearn.''')
        ]), 
        html.Div(children = [
        dcc.Dropdown(
            id='dropdown',
            options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
            value='LA'
        ),  
        html.Div(id='display-value')
        ]),   
    ]),
    html.Div(className='eight columns div-for-charts bg-grey',
    children = [
        dcc.Graph(id='district_split_fig',
        config={'displayModeBar': False},
        animate=True,
        figure=district_split_fig
        )
    ]) 
])


@app.callback(dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    model = pickle.load(open('model/sklearn_model.sav', 'rb'))
    app.run_server(debug=True)