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

## importing external stylesheets - css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

## dash app + server initialize
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

## data
df = pd.read_csv("data/cleaned_beijing.csv", parse_dates=["tradeTime"])
ch = "mean"

## graph subsetting
temp = df["totalPrice"].groupby(df["district"]).agg([np.mean,np.median])

## graph
district_split_fig = px.bar(color = temp.index, x = temp.index, y = temp[ch])
def create_figure(value):
    return  px.bar(color = temp.index, x = temp.index, y = temp[value])

## app layout
app.layout = html.Div(
    className = 'row',
    children = [
        html.Div(
            className = 'row',
            children = [
                html.Div(
                    className='four columns div-user-controls',
                    children = [
                        html.Div(children = [
                            html.H2('House price prediction app'),
                            html.P('''Predicting the sales price of houses in beijing'''),
                            html.P('''Using gradient boosted regression model in sklearn.''')                
                        ]),
                    ]
                ),
                html.Div(className='eight columns div-for-charts bg-grey',
                    children = [
                        html.Div(children = [
                            html.Div(style={'height':'2em'}),
                            html.H4('Select a measure to plot'),
                            dcc.Dropdown(
                                id='dropdown',
                                options=[{'label': i, 'value': i} for i in ['mean','median']],
                                value='mean'
                            ),
                            html.Div(id = 'display-value', style = {'display':'none'}), 
                            dcc.Graph(
                                id = 'district_split_fig',
                                config={'displayModeBar': False},
                                animate=True
                            )
                        ]) 
                    ])
            ]
        ),
        html.Div(
            className = 'row'
        )
    ]
)

## input output callback handle
@app.callback(dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def display_value(value):
    return value
@app.callback(Output('district_split_fig', 'figure'), Input('display-value', 'children'))
def display_graph(value):
    figure = create_figure(value)
    return figure


## Main
if __name__ == '__main__':
    model = pickle.load(open('model/sklearn_model.sav', 'rb'))
    app.run_server(debug=True)