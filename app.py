import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from datetime import date
import re

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
district = ["DongCheng","FengTai","DaXing","FaXing","FangShang",
"ChangPing","ChaoYang","HaiDian","ShiJingShan","XiCheng","TongZhou",
"ShunYi","MenTouGou"
]
fiveYearProperty = ["Ownership<5y","Ownership>5y"]
subway = ["Nearby","Far"]
elevator = ["Present","Absent"]
buildingStructure = ["Unavailable","Mixed","Brick/Wood","Brick/Concrete","Steel","Steel/Concrete"]
buildingType = ["Tower","Bunglow","Plate/Tower","Plate"]
renovationCondition = ["Other","Rough","Simplicity","Hardcover"]

## graph subsetting
temp = df["totalPrice"].groupby(df["district"]).agg([np.mean,np.median])

## graph
district_split_fig = px.bar(color = temp.index, x = temp.index, y = temp[ch])
def create_figure(value):
    return  px.bar(
            x = temp.index,
            y = temp[value],
            labels={'x': 'Districts', 'y':''},
            color_discrete_sequence =['salmon']*len(df)
        ).update_layout(
            {
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)'
            }
        )

## app layout
app.layout = html.Div(
    className = 'row',
    #style = {'padding':'1em'},
    children = [
        html.Div(
            className = 'row',
            style = {'margin':'1em'},
            children = [
                html.Div(
                    className='four columns div-user-controls',
                    style = {'padding':'2.4em','border-radius':'25px'},
                    children = [
                        html.Div(children = [
                            html.H2('House price prediction app'),
                            html.P('''Predicting the sales price of houses in beijing'''),
                            html.P('''Using gradient boosted regression model in sklearn.''')                
                        ]),
                    ]
                ),
                html.Div(
                    className='eight columns div-for-charts',
                    style = {'padding':'2.4em','border-radius':'25px'},
                    children = [
                        html.Div(children = [
                            html.Div(style={'height':'2em'}),
                            html.H4('Select a measure to plot'),
                            dcc.Dropdown(
                                id='dropdown-plot-1-in',
                                options=[{'label': i, 'value': i} for i in ['mean','median']],
                                value='mean'
                            ),
                            dcc.Graph(
                                id = 'dropdown-plot-1-fig',
                                config={'displayModeBar': False},
                                animate=True
                            )
                        ]) 
                    ])
            ]
        ),
        html.Div(
            className = 'row',
            style = {'padding':'2.4em', 'background-color':'skyblue','border-radius':'15px','margin':'1em'},
            children = [
                html.Div(
                    className = 'four columns div-user-controls',
                    style = {'padding':'1.4em'},
                    children = [
                        html.Div(
                            children = [
                                html.H6('''Year trading in'''),
                                html.Div(
                                    style = {'padding-left':'1.2em','padding-bottom':'1.2em'},
                                    children = [
                                        dcc.DatePickerSingle(
                                            id='year-picker-in',
                                            min_date_allowed=date(1950, 1, 1),
                                            max_date_allowed=date(2021, 1, 1),
                                            initial_visible_month=date(2017, 8, 5),
                                            date=date(2017, 8, 25)
                                        ),
                                    ]
                                ),
                                html.Div(id = 'year-picker-display')
                            ]
                        ), 
                        html.Div(
                            children = [
                                html.H6('''Square plot area'''),
                                dcc.Slider(
                                    id='square-slider-in',
                                    min=df['square'].min(),
                                    max=df['square'].max(),
                                    value=df['square'].median(),
                                    #marks={str(sqr): str(sqr) for sqr in df['square'].unique()},
                                    step=1
                                ),
                                html.Div(id = 'square-slider-display')
                            ]
                        ),
                        html.Div(
                            children = [
                                html.H6('''Community average'''),
                                dcc.Slider(
                                    id='CA-slider-in',
                                    min=df['communityAverage'].min(),
                                    max=df['communityAverage'].max(),
                                    value=df['communityAverage'].median(),
                                    #marks={str(ca): str(ca) for ca in df['communityAverage'].unique()},
                                    step=1
                                ),
                                html.Div(id = 'CA-slider-display')
                            ]
                        )
                    ]
                ),
                html.Div(
                    className = 'four columns div-user-controls',
                    style = {'padding':'1.4em'},
                    children = [
                        html.H6('District'),
                        dcc.Dropdown(
                            id='dropdown-district-in',
                            options=[{'label': i, 'value': i} for i in district],
                            value='ChaoYang'
                        ),
                        html.H6('Property ownership period'),
                        dcc.Dropdown(
                            id='dropdown-fiveYearProperty-in',
                            options=[{'label': i, 'value': i} for i in fiveYearProperty],
                            value='Ownership<5y'
                        ),
                        html.H6('Building structure'),
                        dcc.Dropdown(
                            id='dropdown-buildingStructure-in',
                            options=[{'label': i, 'value': i} for i in buildingStructure],
                            value='Brick/Concrete'
                        ),
                        html.H6('Renovation condition'),
                        dcc.Dropdown(
                            id='dropdown-renovationCondition-in',
                            options=[{'label': i, 'value': i} for i in renovationCondition],
                            value='Simplicity'
                        ),
                        html.H6('Building type'),
                        dcc.Dropdown(
                            id='dropdown-buildingType-in',
                            options=[{'label': i, 'value': i} for i in buildingType],
                            value='Tower'
                        ),
                    ]
                ),
                html.Div(
                    className = 'four columns div-user-controls',
                    style = {'padding':'1.4em'},
                    children = [
                        html.Div(
                            children = [
                                html.H6('''Living rooms'''),
                                dcc.Slider(
                                    id='livingRoom-slider-in',
                                    min=df['livingRoom'].min(),
                                    max=df['livingRoom'].max(),
                                    value=df['livingRoom'].median(),
                                    marks={str(ca): str(ca) for ca in df['livingRoom'].unique()},
                                    step=1
                                ),
                                #html.Div(id = 'livingRoom-slider-display')
                            ]
                        ),
                        html.Div(
                            children = [
                                html.H6('''Drawing rooms'''),
                                dcc.Slider(
                                    id='drawingRoom-slider-in',
                                    min=df['drawingRoom'].min(),
                                    max=df['drawingRoom'].max(),
                                    value=df['drawingRoom'].median(),
                                    marks={str(ca): str(ca) for ca in df['drawingRoom'].unique()},
                                    step=1
                                ),
                                #html.Div(id = 'drawingRoom-slider-display')
                            ]
                        ),
                        html.Div(
                            children = [
                                html.H6('''Kitchens'''),
                                dcc.Slider(
                                    id='kitchen-slider-in',
                                    min=df['kitchen'].min(),
                                    max=df['kitchen'].max(),
                                    value=df['kitchen'].median(),
                                    marks={str(ca): str(ca) for ca in df['kitchen'].unique()},
                                    step=1
                                ),
                                #html.Div(id = 'kitchen-slider-display')
                            ]
                        ),
                        html.Div(
                            children = [
                                html.H6('''Bathrooms'''),
                                dcc.Slider(
                                    id='bathRoom-slider-in',
                                    min=df['bathRoom'].min(),
                                    max=df['bathRoom'].max(),
                                    value=df['bathRoom'].median(),
                                    marks={str(ca): str(ca) for ca in df['bathRoom'].unique()},
                                    step=1
                                ),
                                #html.Div(id = 'bathRoom-slider-display')
                            ]
                        )
                    ]
                ),
            ]
        ),
        html.Div(
            className = 'row',
            style = {'padding-top':'3em','padding-bottom':'3em', 'margin':'1em','border-radius':'25px'},
            children = [
                html.P('''The data cleaning and modelling is done on kaggle'''),
                html.A(
                    href = "https://www.kaggle.com/aquaregis32/beijing-house-price-prediction",
                    title = "Link to kaggle notebook",
                    children = [
                        html.P('''Link to kaggle workbook'''),
                    ]
                ),
            ]
        ),
        ## Hidden tags
        html.Div(id = 'dropdown-plot-1-out', style = {'display':'none'}),
        html.Div(id = 'year-picker-out', style = {'display':'none'}),
        html.Div(id = 'square-slider-out', style = {'display':'none'}),
        html.Div(id = 'CA-slider-out', style = {'display':'none'}),
        html.Div(id = 'bathRoom-slider-out', style = {'display':'none'}),
        html.Div(id = 'drawingRoom-slider-out', style = {'display':'none'}),
        html.Div(id = 'kitchen-slider-out', style = {'display':'none'}),
        html.Div(id = 'livingRoom-slider-out', style = {'display':'none'}),
    ]
)

## input output callback handle
@app.callback(dash.dependencies.Output('dropdown-plot-1-out', 'children'),
              [dash.dependencies.Input('dropdown-plot-1-in', 'value')])
def display_value(value):
    return value
@app.callback(Output('dropdown-plot-1-fig', 'figure'), Input('dropdown-plot-1-out', 'children'))
def display_graph(value):
    figure = create_figure(value)
    return figure

## Year picker in, out, and display
@app.callback(
    Output('year-picker-out', 'children'),
    Input('year-picker-in', 'date'))
def year_picker_out(date_value):
    string_prefix = 'You have selected: '
    if date_value is not None:
        date_object = date.fromisoformat(date_value)
        date_string = date_object.strftime('%B %d, %Y')
        return string_prefix + date_string
@app.callback(
    dash.dependencies.Output('year-picker-display', 'children'),
    [dash.dependencies.Input('year-picker-out', 'children')])
def square_slider_display(value):
    return '{}'.format(value)

## Square slider in, out, and display
@app.callback(
    dash.dependencies.Output('square-slider-out', 'children'),
    [dash.dependencies.Input('square-slider-in', 'value')])
def square_slider_out(value):
    return '{}'.format(value)
@app.callback(
    dash.dependencies.Output('square-slider-display', 'children'),
    [dash.dependencies.Input('square-slider-out', 'children')])
def square_slider_display(value):
    return 'Selected: {}'.format(value)

## CA slider in, out, and display
@app.callback(
    dash.dependencies.Output('CA-slider-out', 'children'),
    [dash.dependencies.Input('CA-slider-in', 'value')])
def CA_slider_out(value):
    return '{}'.format(value)
@app.callback(
    dash.dependencies.Output('CA-slider-display', 'children'),
    [dash.dependencies.Input('CA-slider-out', 'children')])
def CA_slider_display(value):
    return 'Selected: {}'.format(value)

## Dropdowns

## Slider col 3
## CA slider in, out, and display
@app.callback(
    dash.dependencies.Output('livingRoom-slider-out', 'children'),
    [dash.dependencies.Input('livingRoom-slider-in', 'value')])
def livingRoom_slider_out(value):
    return '{}'.format(value)
# @app.callback(
#     dash.dependencies.Output('livingRoom-slider-display', 'children'),
#     [dash.dependencies.Input('livingRoom-slider-out', 'children')])
# def livingRoom_slider_display(value):
#     return 'Selected: {}'.format(value)

## CA slider in, out, and display
@app.callback(
    dash.dependencies.Output('drawingRoom-slider-out', 'children'),
    [dash.dependencies.Input('drawingRoom-slider-in', 'value')])
def drawingRoom_slider_out(value):
    return '{}'.format(value)
# @app.callback(
#     dash.dependencies.Output('drawingRoom-slider-display', 'children'),
#     [dash.dependencies.Input('drawingRoom-slider-out', 'children')])
# def drawingRoom_slider_display(value):
#     return 'Selected: {}'.format(value)

## CA slider in, out, and display
@app.callback(
    dash.dependencies.Output('kitchen-slider-out', 'children'),
    [dash.dependencies.Input('kitchen-slider-in', 'value')])
def kitchen_slider_out(value):
    return '{}'.format(value)
# @app.callback(
#     dash.dependencies.Output('kitchen-slider-display', 'children'),
#     [dash.dependencies.Input('kitchen-slider-out', 'children')])
# def kitchen_slider_display(value):
#     return 'Selected: {}'.format(value)

## CA slider in, out, and display
@app.callback(
    dash.dependencies.Output('bathRoom-slider-out', 'children'),
    [dash.dependencies.Input('bathRoom-slider-in', 'value')])
def bathRoom_slider_out(value):
    return '{}'.format(value)
# @app.callback(
#     dash.dependencies.Output('bathRoom-slider-display', 'children'),
#     [dash.dependencies.Input('bathRoom-slider-out', 'children')])
# def bathRoom_slider_display(value):
#     return 'Selected: {}'.format(value)

## Main
if __name__ == '__main__':
    model = pickle.load(open('model/sklearn_model.sav', 'rb'))
    app.run_server(debug=True)