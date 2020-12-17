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
    return  px.bar(
            x = temp.index,
            y = temp[value],
            labels={'x': 'Districts', 'y':''},
            color_discrete_sequence =['skyblue']*len(df)
        ).update_layout(
            {
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)'
            }
        )

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
            className = 'row',
            children = [
                html.Div(
                className = 'four columns div-user-controls',
                children = [
                        dcc.Slider(
                            id='year--slider',
                            min=df['tradeTime'].min(),
                            max=df['tradeTime'].max(),
                            value=df['tradeTime'].max(),
                            marks={str(year): str(year) for year in df['tradeTime'].unique()},
                            step=None
                        ),
                        dcc.Slider(
                            id='square-slider',
                            min=df['square'].min(),
                            max=df['square'].max(),
                            value=df['square'].median(),
                            marks={str(sqr): str(sqr) for sqr in df['square'].unique()},
                            step=None
                        ),
                        dcc.Slider(
                            id='CA-slider',
                            min=df['communityAverage'].min(),
                            max=df['communityAverage'].max(),
                            value=df['communityAverage'].median(),
                            marks={str(ca): str(ca) for ca in df['communityAverage'].unique()},
                            step=None
                        )
                    ]
                ),
                html.Div(
                className = 'four columns div-user-controls',
                children = [
                        
                    ]
                ),
                html.Div(
                className = 'four columns div-user-controls',
                children = [
                        
                    ]
                ),
            ]
        ),
        html.Div(
            className = 'row',
            children = [
                html.A(
                    href = "https://www.kaggle.com/aquaregis32/beijing-house-price-prediction",
                    title = "Link to kaggle notebook",
                    children = [
                        html.P('''The data cleaning and modelling can be viewed on my kaggle page'''),
                    ]
                ),
            ]
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