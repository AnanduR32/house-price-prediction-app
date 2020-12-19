import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from datetime import date, datetime as dt
import time
import re

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

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
temp1 = df["totalPrice"].groupby(df["district"]).agg([np.mean,np.median])
temp2 = df["communityAverage"].groupby(df["district"]).agg([np.mean,np.median])

## graph
#district_split_fig = px.bar(color = temp1.index, x = temp1.index, y = temp1[ch])
def create_figure(value):
    return  px.bar(
            x = temp1.index,
            y = temp1[value],
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
                    className = 'row',
                    children = [
                        html.Div(
                            className = 'four columns div-user-controls',
                            style = {'padding':'1.4em'},
                            children = [
                                html.Div(
                                    children = [
                                        html.H6('''Time trading in'''),
                                        dcc.Slider(
                                            id = 'time-slider-in',
                                            min = 731002,
                                            max = 736722,
                                            value = 735853,
                                            step = 1
                                        ),
                                        html.Div(id = 'time-slider-display')
                                
                                    ]
                                ), 
                                html.Hr(style = {'width':'70%'}),
                                html.Div(
                                    children = [
                                        html.H6('''Square plot area'''),
                                        dcc.Slider(
                                            id='square-slider-in',
                                            min=df['square'].min(),
                                            max=df['square'].max(),
                                            value=df['square'].median(),
                                            marks={str(sqr): str(sqr) for sqr in np.linspace(df['square'].min(),df['square'].max(), num = 5)},
                                            step=1
                                        ),
                                        html.Div(id = 'square-slider-display')
                                    ]
                                ),
                                html.Hr(style = {'width':'70%'}),
                                html.Div(
                                    children = [
                                        html.H6(id = 'CA-slider-display'),
                                        dcc.Slider(
                                            id='CA-slider-in',
                                            min=df['communityAverage'].min(),
                                            max=df['communityAverage'].max(),
                                            disabled = True,
                                            #marks={str(ca): str(ca) for ca in np.linspace(df['communityAverage'].min(),df['communityAverage'].max(), num = 5)},
                                            step=1
                                        ),
                                        # html.Div(id = 'CA-slider-display')
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
                    children = [
                        html.H5(
                            id = 'prediction-display'
                        )
                    ]
                )
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

        # data for predicting
        html.Div(id = 'time-slider-out', style = {'display':'none'}),
        html.Div(id = 'square-slider-out', style = {'display':'none'}),
        html.Div(id = 'CA-slider-out', style = {'display':'none'}),
        html.Div(id = 'bathRoom-slider-out', style = {'display':'none'}),
        html.Div(id = 'drawingRoom-slider-out', style = {'display':'none'}),
        html.Div(id = 'kitchen-slider-out', style = {'display':'none'}),
        html.Div(id = 'livingRoom-slider-out', style = {'display':'none'}),
        html.Div(id = 'dropdown-district-out', style = {'display':'none'}),
        html.Div(id = 'dropdown-fiveYearProperty-out', style = {'display':'none'}),
        html.Div(id = 'dropdown-buildingStructure-out', style = {'display':'none'}),
        html.Div(id = 'dropdown-renovationCondition-out', style = {'display':'none'}),
        html.Div(id = 'dropdown-buildingType-out', style = {'display':'none'}),

        # prediction output
        html.Div(id = 'prediction-out', style = {'display':'none'})
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

## Col 1 
## Time slider in, out, and display
@app.callback(
    Output('time-slider-out', 'children'),
    [Input('time-slider-in', 'value')])
def time_slider_out(value):
    return value
@app.callback(
    Output('time-slider-display', 'children'),
    Input('time-slider-out', 'children'))
def time_slider_display(value):
    value = dt.fromordinal(value).strftime('%d %B, %Y')
    return 'Selected: {}'.format(value)

## Square slider in, out, and display
@app.callback(
    Output('square-slider-out', 'children'),
    [Input('square-slider-in', 'value')])
def square_slider_out(value):
    return str(value)
@app.callback(
    Output('square-slider-display', 'children'),
    Input('square-slider-out', 'children'))
def square_slider_display(value):
    return 'Selected: {}'.format(value)

## CA slider in, out, and display
@app.callback(
    Output('CA-slider-out', 'children'),
    Input('CA-slider-in', 'value'))
def CA_slider_out(value):
    return str(value)
@app.callback(
    Output('CA-slider-display', 'children'),
    Input('CA-slider-out', 'children'))
def CA_slider_display(value):
    return 'Community average: {}'.format(value)

## Dropdowns
@app.callback(
    Output('dropdown-district-out', 'children'),
    Output('CA-slider-in', 'value'),
    Input('dropdown-district-in', 'value'))
def dropdown_district_in(value):
    CA_med = temp2.loc[str(value),"median"]
    return value, CA_med

@app.callback(
    Output('dropdown-fiveYearProperty-out', 'children'),
    Input('dropdown-fiveYearProperty-in', 'value'))
def dropdown_fiveYearProperty_in(value):
    return str(value)

@app.callback(
    Output('dropdown-buildingStructure-out', 'children'),
    Input('dropdown-buildingStructure-in', 'value'))
def dropdown_buildingStructure_in(value):
    return str(value)

@app.callback(
    Output('dropdown-renovationCondition-out', 'children'),
    Input('dropdown-renovationCondition-in', 'value'))
def dropdown_renovationCondition_in(value):
    return str(value)

@app.callback(
    Output('dropdown-buildingType-out', 'children'),
    Input('dropdown-buildingType-in', 'value'))
def dropdown_buildingType_in(value):
    return str(value)

## Slider col 3
## CA slider in, out, and display
@app.callback(
    Output('livingRoom-slider-out', 'children'),
    Input('livingRoom-slider-in', 'value'))
def livingRoom_slider_out(value):
    return str(value)

## CA slider in, out, and display
@app.callback(
    Output('drawingRoom-slider-out', 'children'),
    Input('drawingRoom-slider-in', 'value'))
def drawingRoom_slider_out(value):
    return str(value)

## CA slider in, out, and display
@app.callback(
    Output('kitchen-slider-out', 'children'),
    Input('kitchen-slider-in', 'value'))
def kitchen_slider_out(value):
    return str(value)

## CA slider in, out, and display
@app.callback(
    Output('bathRoom-slider-out', 'children'),
    Input('bathRoom-slider-in', 'value'))
def bathRoom_slider_out(value):
    return str(value)

## Predicting - model working
@app.callback(
    Output('prediction-out', 'children'),
    Input('time-slider-out', 'children'),
    Input('square-slider-out', 'children'),
    Input('CA-slider-out', 'children'),
    Input('bathRoom-slider-out', 'children'),
    Input('drawingRoom-slider-out', 'children'),
    Input('kitchen-slider-out', 'children'),
    Input('livingRoom-slider-out', 'children'),
    Input('dropdown-district-out', 'children'),
    Input('dropdown-fiveYearProperty-out', 'children'),
    Input('dropdown-buildingStructure-out', 'children'),
    Input('dropdown-renovationCondition-out', 'children'),
    Input('dropdown-buildingType-out', 'children'))
def prediction_out(tradeTime, square, communityAverage, bathRoom, drawingRoom, kitchen, livingRoom,
 district, fiveYearProperty, buildingStructure, renovationCondition, buildingType):
    df_dict = {
        'tradeTime': tradeTime,
        'square':int(square),
        'communityAverage':int(communityAverage),
        'livingRoom':livingRoom,
        'drawingRoom':drawingRoom,
        'kitchen':kitchen,
        'bathRoom':bathRoom,
        'buildingType':buildingType,
        'renovationCondition':renovationCondition,
        'buildingStructure':buildingStructure,
        'fiveYearsProperty':fiveYearProperty,
        'district':district
    }    
    df_new = pd.DataFrame(df_dict,index = [0])
    cat_cols = ['livingRoom','drawingRoom','kitchen','bathRoom','buildingType','renovationCondition','buildingStructure','fiveYearsProperty','district']
    encoded_array = enc.transform(df_new[cat_cols])
    encoded_df = pd.DataFrame(encoded_array, columns = enc.get_feature_names(input_features = cat_cols))
    df_enc = pd.concat([df_new, encoded_df], axis=1).drop(columns = cat_cols, axis=1)
    if df_enc is not None and df_enc is not '':
        try:
            pred = model.predict(df_enc)
            pred = np.round(pred[0], 4)
            return pred
        except:
            return 'Unable to predict'

@app.callback(
    Output('prediction-display', 'children'),
    Input('prediction-out', 'children'))
def prediction_display(value):
    return 'The predicted price: {}'.format(value)

## Main
if __name__ == '__main__':
    model = pickle.load(open('model/sklearn_model.sav', 'rb'))
    enc = pickle.load(open('encoders/encoder.sav', 'rb'))
    app.run_server(debug=True)