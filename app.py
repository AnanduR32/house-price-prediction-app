import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from datetime import date, datetime as dt
import time
import re
from scipy.stats import percentileofscore 

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

import dash_core_components as dcc
import plotly.express as px

## importing external stylesheets - css
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

## dash app + server initialize
app = dash.Dash(__name__) #, external_stylesheets=external_stylesheets
server = app.server

## data
df = pd.read_csv("data/cleaned_beijing.csv", parse_dates=["tradeTime"])
data_pop = pd.read_csv("data/encoded_beijing.csv", parse_dates=["tradeTime"])
ch = "mean"
district = ["DongCheng","FengTai","DaXing","FaXing","FangShang",
"ChangPing","ChaoYang","HaiDian","ShiJingShan","XiCheng","TongZhou",
"ShunYi","MenTouGou"
]
fiveYearProperty = ["Ownership<5y","Ownership>5y"]
subway = ["Nearby","Far"]
elevator = ["Present","Absent"]
buildingStructure = ["Mixed","Brick/Wood","Brick/Concrete","Steel","Steel/Concrete"]
buildingType = ["Tower","Bunglow","Plate/Tower","Plate"]
renovationCondition = ["Other","Rough","Simplicity","Hardcover"]
elevator = ["Present","Absent"]
subway = ["Nearby","Far"]

## Colors
title_main = '#ED7B84'
graph_titles = '#c3e6ea'
color_list = ['#D4EBD4']*13

def encode_df(df):
    cat_cols = ['livingRoom','drawingRoom','kitchen','bathRoom','buildingType','renovationCondition',
    'buildingStructure','elevator', 'fiveYearsProperty', 'subway','district']
    encoded_array = enc.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded_array, columns = enc.get_feature_names(input_features = cat_cols))
    df_enc = pd.concat([df, encoded_df], axis=1).drop(columns = cat_cols, axis=1)
    return(df_enc)
def prediction(df):
    try:
        pred = model.predict(df)
        pred = np.round(pred[0], 4)
        return pred
    except:
        return 'Unable to predict'

chosen_pop = ['square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'buildingType',
            'renovationCondition', 'buildingStructure' , 'elevator',
            'fiveYearsProperty','subway', 'district']
buildingType_re = {
    "Tower":1,
    "Bunglow":2,
    "Plate/Tower":3,
    "Plate":4
}
renovationCondition_re = {
    "Other":1,
    "Rough":2,
    "Simplicity":3,
    "Hardcover":4
}
buildingStructure_re = {
    "Unavailable":1,
    "Mixed":2,
    "Brick/Wood":3,
    "Brick/Concrete":4,
    "Steel":5,
    "Steel/Concrete":6
}
elevator_re = {
    "Present":1,
    "Absent":0
}
subway_re = {
    "Nearby":1,
    "Far":0
}
fiveYearProperty_re = {
    "Ownership<5y":1,
    "Ownership>5y":0
}
district_re = {
    "DongCheng":1,
    "FengTai":2,
    "DaXing":3,
    "FaXing":4,
    "FangShang":5,
    "ChangPing":6,
    "ChaoYang":7,
    "HaiDian":8,
    "ShiJingShan":9,
    "XiCheng":10,
    "TongZhou":11,
    "ShunYi":12,
    "MenTouGou":13
}
correct_label = {
    7:buildingType_re,
    8:renovationCondition_re,
    9:buildingStructure_re,
    10:elevator_re,
    11:fiveYearProperty_re,
    12:subway_re,
    13:district_re
}            
def pop_measure(data):
    res = [0]*12
    try:
        for val,i in enumerate(chosen_pop):
            res[val] = data[i].apply(lambda x: percentileofscore(data_pop[i],int(x))).values[0]
        res = int(np.array(res).mean())
        return res
    except:
        return 'Unable to compute popularity'



## Model
model = pickle.load(open('model/sklearn_model.sav', 'rb'))

## Encoder
enc = pickle.load(open('encoders/one_hot_encoder.sav', 'rb'))

## graph subsetting
temp1 = df["totalPrice"].groupby(df["district"]).agg([np.mean,np.median])
temp2 = df["communityAverage"].groupby(df["district"]).agg([np.mean,np.median])

## graph
def create_figure(metric, color_list):
    return  px.bar(
                x = temp1.index,
                y = temp1[metric],
                labels={'x': 'Districts', 'y':''},
                color_discrete_sequence = ['#D4EBD4']*13
                #color_discrete_sequence =['#D4EBD4']*len(df)
            ).update_layout(
                {
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
                }
            ).update_traces(marker_color=color_list)
app.title = 'House price app'
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
                    className='div-user-controls',
                    style = {'padding':'2.4em','borderRadius':'25px'},
                    children = [
                        html.Div(children = [
                            html.H2(
                                children = [
                                    'House price prediction app',
                                ],
                                style = {'color':title_main, 'fontWeight': 'bold'},
                            ),
                            html.P('''Predicting the sales price of houses in beijing'''),
                            html.P('''Using gradient boosted regression model in sklearn.''')                
                        ]),
                    ]
                ),
            ]
        ),
        html.Div(
            className = 'row',
            style = {'margin':'1em'},
            children = [
                html.Div(
                    className = 'four container columns div-for-charts',
                    style = {'padding':'1.4em','borderRadius':'25px'},
                    children = [
                        html.Div(
                            children = [
                                html.H5(
                                    children = [
                                        'Prices in different districts',
                                    ],
                                    style = {'color':graph_titles}
                                ),
                                dcc.Dropdown(
                                    id='dropdown-plot-1-in',
                                    options=[{'label': i, 'value': i} for i in ['mean','median']],
                                    searchable=False,
                                    clearable = False,
                                    value='mean',
                                    style = {'width':'6em'}
                                ),
                                dcc.Graph(
                                    id = 'dropdown-plot-1-fig',
                                    config={'displayModeBar': False},
                                    animate=True
                                )
                            ] 
                        )
                    ]
                ),
                html.Div(
                    className = 'four container columns div-for-charts',
                    style = {'padding':'1.4em','borderRadius':'25px'},
                    children = [
                        html.Div(
                            children = [
                                
                            ] 
                        )
                    ]
                ),
                html.Div(
                    className = 'four container columns div-for-charts',
                    style = {'padding':'1.4em','borderRadius':'25px'},
                    children = [
                        html.Div(
                            children = [
                                
                            ] 
                        )
                    ]
                ),
                html.Div(
                    className = 'four container columns div-for-charts',
                    style = {'padding':'1.4em','borderRadius':'25px'},
                    children = [
                        html.Div(
                            children = [
                                
                            ] 
                        )
                    ]
                )
                                        
            ]
        ),
        html.Div(
            className = 'row',
            style = {'padding':'2.4em', 'backgroundImage':'linear-gradient(to bottom, #d4ebd4, #ccebda, #c7eae0, #c4e8e6, #c3e6ea)','borderRadius':'15px','margin':'1em'},
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
                                        html.H6(
                                            id = 'popularity-slider-display',
                                            children = [
                                                "Popularity"
                                            ]
                                        ),
                                        dcc.Slider(
                                            id='popularity-slider-in',
                                            min=0,
                                            max=100,
                                            disabled = True,
                                            marks={
                                                '0': {'label': '0', 'style':{'color': '#FF6962'}},
                                                '25': {'label': '25', 'style':{'color': '#FF8989'}},
                                                '50': {'label': '50', 'style':{'color': '#FFB3A5'}},
                                                '75': {'label': '75', 'style':{'color': '#7ABD91'}},
                                                '100': {'label': '100', 'style':{'color': '#5FA777'}}
                                            },
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
                                    searchable=False,
                                    clearable = False,
                                    value='ChaoYang'
                                ),
                                html.H6('Property ownership period'),
                                dcc.Dropdown(
                                    id='dropdown-fiveYearProperty-in',
                                    options=[{'label': i, 'value': i} for i in fiveYearProperty],
                                    searchable=False,
                                    clearable = False,
                                    value='Ownership<5y'
                                ),
                                html.H6('Building structure'),
                                dcc.Dropdown(
                                    id='dropdown-buildingStructure-in',
                                    options=[{'label': i, 'value': i} for i in buildingStructure],
                                    searchable=False,
                                    clearable = False,
                                    value='Brick/Concrete'
                                ),
                                html.H6('Renovation condition'),
                                dcc.Dropdown(
                                    id='dropdown-renovationCondition-in',
                                    options=[{'label': i, 'value': i} for i in renovationCondition],
                                    searchable=False,
                                    clearable = False,
                                    value='Simplicity'
                                ),
                                html.H6('Building type'),
                                dcc.Dropdown(
                                    id='dropdown-buildingType-in',
                                    options=[{'label': i, 'value': i} for i in buildingType],
                                    searchable=False,
                                    clearable = False,
                                    value='Tower'
                                ),
                            ]
                        ),
                        html.Div(
                            className = 'four columns div-user-controls',
                            style = {'padding':'1.4em'},
                            children = [
                                html.Div(
                                    className = 'row',
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
                                        ),
                                    ]
                                ),
                                html.Div(
                                    className = 'row',
                                    style = {'padding':'0.5em'},
                                    children = [
                                        html.Div(
                                            className = 'four columns div-user-controls',
                                            style = {'padding':'1em','margin':'1em'},
                                            children = [
                                                html.H6('''Elevator'''),
                                                dcc.RadioItems(
                                                    id = "radio-elevator-in",
                                                    options=[{'label': i, 'value': i} for i in elevator],
                                                    value='Present'
                                                )
                                            ]
                                        ),
                                        html.Div(
                                            className = 'four columns div-user-controls',
                                            style = {'padding':'1em','margin':'1em'},
                                            children = [
                                                html.H6('''Subway'''),
                                                dcc.RadioItems(
                                                    id = "radio-subway-in",
                                                    options=[{'label': i, 'value': i} for i in subway],
                                                    value='Far'
                                                )         
                                            ]
                                        )  
                                    ]
                                )  
                            ]
                        ),
                    ]
                ),
                html.Div(
                    className = 'row',
                    children = [
                        html.Div(
                            children = [
                                html.P(
                                    '''The predicted total price: ''',
                                    style = {'fontSize':'1em','display':'inline-block', 'marginRight':'0.5em'},
                                ),
                                html.P(
                                    style = {'fontSize':'1.5em','display':'inline-block',  'marginRight':'0.5em'},
                                    id = 'prediction-display',
                                ),
                                html.P(
                                    style = {'fontSize':'1em','display':'inline-block', 'marginRight':'0.5em'},
                                    children = [
                                       '''\u5143 (in millions)'''
                                    ]
                                ),
                            ]
                        ),
                    ]
                )
            ]
        ),
        html.Div(
            className = 'row',
            style = {'paddingTop':'3em','paddingBottom':'3em', 'margin':'1em','borderRadius':'25px'},
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
        html.Div(id = 'popularity-slider-out', style = {'display':'none'}),
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
        html.Div(id = 'radio-elevator-out', style = {'display':'none'}),
        html.Div(id = 'radio-subway-out', style = {'display':'none'}),

        # Dataframe
        html.Div(id = 'dataframe-out', style = {'display':'none'}),
        # prediction output
        html.Div(id = 'prediction-out', style = {'display':'none'})
    ]
)

## input output callback handle
@app.callback(dash.dependencies.Output('dropdown-plot-1-out', 'children'),
              [dash.dependencies.Input('dropdown-plot-1-in', 'value')])
def display_value(value):
    return value
@app.callback(
    Output('dropdown-plot-1-fig', 'figure'), 
    Input('dropdown-plot-1-out', 'children'),
    Input('dropdown-district-out','children')
)
def display_graph(metric, district_name):
    #color_list = generate_discrete_color_list(district_name)
    district_idx = district.index(district_name)
    color_list = ['#D4EBD4']*13
    color_list[district_idx] = '#4A2545'
    time.sleep(0.2)
    figure = create_figure(metric, color_list)
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

## Popularity slider in, out, and display
@app.callback(
    Output('popularity-slider-out', 'children'),
    Input('dataframe-out', 'children'))
def popularity_slider_out(json_file):
    if(json_file!= None and json_file!=''):
        try:
            df = pd.read_json(json_file)
            for key,val in correct_label.items():
                df.iloc[:,key] = df.iloc[:,key].replace(val)
            popularity = pop_measure(df)
            return popularity
        except:
            return "Cannot process request"
@app.callback(
    Output('popularity-slider-in', 'value'),
    Input('popularity-slider-out', 'children')
)
def popularity_slider_in(value):
    return value


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
    return 'Average population: {}'.format(value)

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

## Elevator radio in
@app.callback(
    Output('radio-elevator-out', 'children'),
    Input('radio-elevator-in', 'value')
)
def radio_elevator_out(value):
    return str(value)
## Subway radio in
@app.callback(
    Output('radio-subway-out', 'children'),
    Input('radio-subway-in', 'value')
)
def radio_subway_out(value):
    return str(value)

## Predicting - model working
@app.callback(
    Output('dataframe-out', 'children'),
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
    Input('dropdown-buildingType-out', 'children'),
    Input('radio-elevator-out', 'children'),
    Input('radio-subway-out', 'children')
)
def dataframe_out(tradeTime, square, communityAverage, bathRoom, drawingRoom, kitchen, livingRoom,
 district, fiveYearProperty, buildingStructure, renovationCondition, buildingType, elevator, subway):
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
        'elevator':elevator,
        'fiveYearsProperty':fiveYearProperty,
        'subway':subway,
        'district':district
    }    
    df_new = pd.DataFrame(df_dict,index = [0])
    return df_new.to_json() 

@app.callback(
    Output('prediction-out', 'children'),
    Input('dataframe-out', 'children'))
def prediction_out(json_file):
    if(json_file!= None and json_file!=''):
        try:
            df = pd.read_json(json_file)
            df_enc = encode_df(df)
            pred = prediction(df_enc)
            return pred
        except:
            return "Cannot process request"

@app.callback(
    Output('prediction-display', 'children'),
    Input('prediction-out', 'children'))
def prediction_display(value):
    return str(value)

## Main
if __name__ == '__main__':
    app.run_server(debug = True)