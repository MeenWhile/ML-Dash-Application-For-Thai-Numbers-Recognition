import base64
import datetime
import io
import os

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
import dash_daq as daq

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import cv2
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, recall_score, precision_score,accuracy_score,roc_auc_score

#-------------------------------------------------------------------------------------------------------------------------------------------------

df = None
df0 = None
df1 = None
df2 = None
df3 = None
df4 = None
df5 = None
df6 = None
df7 = None
df8 = None
df9 = None
dfall = None

dfdummy = pd.DataFrame(np.full([3,9],0), columns = ['Model', 'Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC', 'TT (Sec)'])

dummyModel = html.Div([
    dash_table.DataTable(
        dfdummy.to_dict('records'),
        [{'name': i, 'id': i} for i in dfdummy.columns]
    ),
    html.Hr(),
    
    html.H5('Graph for Train set'),
    html.Div([
        dcc.Graph(style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(style={'width': '50%', 'display': 'inline-block'}),
        ]),
    html.Hr(),
    
    html.Div([
        daq.LEDDisplay(
            id='acctrain',
            #label="Default",
            value=0,
            label = "Accuracy Score for Train set",
            style={'width': '33%', 'display': 'inline-block'}
        ),

        daq.LEDDisplay(
            id='pretrain',
            #label="Default",
            value=0,
            label = "Precision Score for Train set",
            style={'width': '33%', 'display': 'inline-block'}
            ),
                
        daq.LEDDisplay(
            id='rectrain',
            #label="Default",
            value=0,
            label = "Recall Score for Train set",
            style={'width': '33%', 'display': 'inline-block'}
            ),
        html.Hr(),
    ]),
    
    html.H5('Graph for Test set'),
    html.Div([
        dcc.Graph(style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(style={'width': '50%', 'display': 'inline-block'}),
        ]),
    html.Hr(),
    
    html.Div([
        daq.LEDDisplay(
            id='acctest',
            #label="Default",
            value=0,
            label = "Accuracy Score for Test set",
            style={'width': '33%', 'display': 'inline-block'}
        ),

        daq.LEDDisplay(
            id='pretest',
            #label="Default",
            value=0,
            label = "Precision Score for Test set",
            style={'width': '33%', 'display': 'inline-block'}
            ),
                
        daq.LEDDisplay(
            id='rectest',
            #label="Default",
            value=0,
            label = "Recall Score for Test set",
            style={'width': '33%', 'display': 'inline-block'}
            ),
    ]),
    ])

results = None
model = None

#-------------------------------------------------------------------------------------------------------------------------------------------------
#For Resize img

def Count_Up(img,img_check):
    count_up = 0
    check_up = False
    for i in range(28):
        for j in range(28):
            if img[i,j] < max(img_check):
                check_up = True
                break
        if check_up == True:
            break
        count_up += 1
    return count_up

def Count_Down(img,img_check):
    count_down = 0
    check_down = False
    for i in range(27,0,-1):
        for j in range(28):
            if img[i,j] < max(img_check):
                check_down = True
                break
        if check_down == True:
            break
        count_down += 1
    return count_down

def Count_Left(img,img_check):
    count_left = 0
    check_left = False
    for i in range(28):
        for j in range(28):
            if img[j,i] < max(img_check):
                check_left = True
                break
        if check_left == True:
            break
        count_left += 1
    return count_left

def Count_Right(img,img_check):
    count_right = 0
    check_right = False
    for i in range(27,0,-1):
        for j in range(27,0,-1):
            if img[j,i] < max(img_check):
                check_right = True
                break
        if check_right == True:
            break
        count_right += 1
    return count_right

def change_position(img,img_check):
    crop_img = img
    dummy = np.full(28,255,dtype='uint8')
    dummy2 = np.full([28,1],255,dtype='uint8')
    
    count_array = []
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    #vertical_change_position
    avg_vertical = (count_array[0] + count_array[1])/2
    while Count_Up(crop_img,img_check) - avg_vertical > 0.5:
        crop_img = crop_img[1:crop_img.shape[0], :]
        crop_img = np.vstack([crop_img,dummy])
    while Count_Down(crop_img,img_check) - avg_vertical > 0.5:
        crop_img = crop_img[0:crop_img.shape[0]-1, :]
        crop_img = np.vstack([dummy,crop_img])
    
    #horizontal_change_position
    avg_horizontal = (count_array[2] + count_array[3])/2
    while Count_Left(crop_img,img_check) - avg_horizontal > 0.5:
        crop_img = crop_img[:,1:crop_img.shape[1]]
        crop_img = np.hstack([crop_img,dummy2])
    while Count_Right(crop_img,img_check) - avg_horizontal > 0.5:
        crop_img = crop_img[:, 0:crop_img.shape[1]-1]
        crop_img = np.hstack([dummy2,crop_img])

    return crop_img

#-------------------------------------------------------------------------------------------------------------------------------------------------

def img_csv_0(contents, filename, date):
    count_array = []
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img_check = img.flatten()
    img_check = [0,240]
    crop_img = change_position(img,img_check)
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    crop_img = crop_img[min(count_array):crop_img.shape[0], :]
    crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
    crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
    crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
    crop_img = cv2.resize(crop_img,(28,28))

    crop_img_flatten = crop_img.flatten()
    lst = [0]
    for k in crop_img_flatten:
        lst.append(k)
    return lst

def img_csv_1(contents, filename, date):
    count_array = []
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img_check = img.flatten()
    img_check = [0,240]
    crop_img = change_position(img,img_check)
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    crop_img = crop_img[min(count_array):crop_img.shape[0], :]
    crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
    crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
    crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
    crop_img = cv2.resize(crop_img,(28,28))

    crop_img_flatten = crop_img.flatten()
    lst = [1]
    for k in crop_img_flatten:
        lst.append(k)
    return lst

def img_csv_2(contents, filename, date):
    count_array = []
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img_check = img.flatten()
    img_check = [0,240]
    crop_img = change_position(img,img_check)
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    crop_img = crop_img[min(count_array):crop_img.shape[0], :]
    crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
    crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
    crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
    crop_img = cv2.resize(crop_img,(28,28))

    crop_img_flatten = crop_img.flatten()
    lst = [2]
    for k in crop_img_flatten:
        lst.append(k)
    return lst

def img_csv_3(contents, filename, date):
    count_array = []
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img_check = img.flatten()
    img_check = [0,240]
    crop_img = change_position(img,img_check)
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    crop_img = crop_img[min(count_array):crop_img.shape[0], :]
    crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
    crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
    crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
    crop_img = cv2.resize(crop_img,(28,28))

    crop_img_flatten = crop_img.flatten()
    lst = [3]
    for k in crop_img_flatten:
        lst.append(k)
    return lst

def img_csv_4(contents, filename, date):
    count_array = []
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img_check = img.flatten()
    img_check = [0,240]
    crop_img = change_position(img,img_check)
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    crop_img = crop_img[min(count_array):crop_img.shape[0], :]
    crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
    crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
    crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
    crop_img = cv2.resize(crop_img,(28,28))

    crop_img_flatten = crop_img.flatten()
    lst = [4]
    for k in crop_img_flatten:
        lst.append(k)
    return lst

def img_csv_5(contents, filename, date):
    count_array = []
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img_check = img.flatten()
    img_check = [0,240]
    crop_img = change_position(img,img_check)
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    crop_img = crop_img[min(count_array):crop_img.shape[0], :]
    crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
    crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
    crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
    crop_img = cv2.resize(crop_img,(28,28))

    crop_img_flatten = crop_img.flatten()
    lst = [5]
    for k in crop_img_flatten:
        lst.append(k)
    return lst

def img_csv_6(contents, filename, date):
    count_array = []
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img_check = img.flatten()
    img_check = [0,240]
    crop_img = change_position(img,img_check)
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    crop_img = crop_img[min(count_array):crop_img.shape[0], :]
    crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
    crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
    crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
    crop_img = cv2.resize(crop_img,(28,28))

    crop_img_flatten = crop_img.flatten()
    lst = [6]
    for k in crop_img_flatten:
        lst.append(k)
    return lst

def img_csv_7(contents, filename, date):
    count_array = []
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img_check = img.flatten()
    img_check = [0,240]
    crop_img = change_position(img,img_check)
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    crop_img = crop_img[min(count_array):crop_img.shape[0], :]
    crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
    crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
    crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
    crop_img = cv2.resize(crop_img,(28,28))

    crop_img_flatten = crop_img.flatten()
    lst = [7]
    for k in crop_img_flatten:
        lst.append(k)
    return lst

def img_csv_8(contents, filename, date):
    count_array = []
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img_check = img.flatten()
    img_check = [0,240]
    crop_img = change_position(img,img_check)
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    crop_img = crop_img[min(count_array):crop_img.shape[0], :]
    crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
    crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
    crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
    crop_img = cv2.resize(crop_img,(28,28))

    crop_img_flatten = crop_img.flatten()
    lst = [8]
    for k in crop_img_flatten:
        lst.append(k)
    return lst

def img_csv_9(contents, filename, date):
    count_array = []
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img_check = img.flatten()
    img_check = [0,240]
    crop_img = change_position(img,img_check)
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    crop_img = crop_img[min(count_array):crop_img.shape[0], :]
    crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
    crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
    crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
    crop_img = cv2.resize(crop_img,(28,28))

    crop_img_flatten = crop_img.flatten()
    lst = [9]
    for k in crop_img_flatten:
        lst.append(k)
    return lst

#------------------------------------------------------------------------------------------

def parse_contents(contents, filename, date):
    global df
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),

        dash_table.DataTable(
            df.head().to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),
    ])

#------------------------------------------------------------------------------------------

predicted = []

def parse_contents2(contents, filename, date):
    
    global model
    lst_columns = []
    lst_number = []
    for l in range(1,28*28+1):
        lst_columns.append('pixel' + str(l))
    count_array = []
    file = filename
    img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    img_check = [0,240]
    crop_img = change_position(img,img_check)
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))

    crop_img = crop_img[min(count_array):crop_img.shape[0], :]
    crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
    crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
    crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
    crop_img = cv2.resize(crop_img,(28,28))

    crop_img_flatten = crop_img.flatten()
    lst = []
    for k in crop_img_flatten:
        lst.append(k)
    lst_number.append(lst)
    predict = predict_model(model,data = pd.DataFrame(lst_number, columns = lst_columns))
    predicted.append( predict.iloc[0,784] )

    return html.Div([
        html.H5(filename),
        html.Img(src=contents),
        html.Div('Predicted Number : ' + str(predict.iloc[0,784])),
        html.Hr(),
    ])

#-------------------------------------------------------------------------------------------------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    
    html.H5(children = 'Data for Training (Image)'),
    html.Div(children = [
        html.H6(children = 'Image: 0'),
        dcc.Upload(
                id='upload-0',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'display': 'inline-block',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
        ),html.Div(id='output-0-upload'),
        ],style={'width': '20%', 'display': 'inline-block'}),

    html.Div(children = [
        html.H6(children = 'Image: 1'),
        dcc.Upload(
                id='upload-1',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'display': 'inline-block',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
        ),html.Div(id='output-1-upload'),
        # html.Div(id='output-1-upload'),
        ],style={'width': '20%', 'display': 'inline-block'}),

    html.Div(children = [
        html.H6(children = 'Image: 2'),
        dcc.Upload(
                id='upload-2',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'display': 'inline-block',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
        ),html.Div(id='output-2-upload'),
        # html.Div(id='output-2-upload'),
        ],style={'width': '20%', 'display': 'inline-block'}),

    html.Div(children = [
        html.H6(children = 'Image: 3'),
        dcc.Upload(
                id='upload-3',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'display': 'inline-block',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
        ),html.Div(id='output-3-upload'),
        # html.Div(id='output-3-upload'),
        ],style={'width': '20%', 'display': 'inline-block'}),

    html.Div(children = [
        html.H6(children = 'Image: 4'),
        dcc.Upload(
                id='upload-4',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'display': 'inline-block',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
        ),html.Div(id='output-4-upload'),
        # html.Div(id='output-4-upload'),
        ],style={'width': '20%', 'display': 'inline-block'}),
    html.Hr(),

    html.Div(children = [
        html.H6(children = 'Image: 5'),
        dcc.Upload(
                id='upload-5',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'display': 'inline-block',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
        ),html.Div(id='output-5-upload'),
        # html.Div(id='output-5-upload'),
        ],style={'width': '20%', 'display': 'inline-block'}),

    html.Div(children = [
        html.H6(children = 'Image: 6'),
        dcc.Upload(
                id='upload-6',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'display': 'inline-block',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
        ),html.Div(id='output-6-upload'),
        # html.Div(id='output-6-upload'),
        ],style={'width': '20%', 'display': 'inline-block'}),

    html.Div(children = [
        html.H6(children = 'Image: 7'),
        dcc.Upload(
                id='upload-7',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'display': 'inline-block',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
        ),html.Div(id='output-7-upload'),
        # html.Div(id='output-7-upload'),
        ],style={'width': '20%', 'display': 'inline-block'}),

    html.Div(children = [
        html.H6(children = 'Image: 8'),
        dcc.Upload(
                id='upload-8',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'display': 'inline-block',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
        ),html.Div(id='output-8-upload'),
        # html.Div(id='output-8-upload'),
        ],style={'width': '20%', 'display': 'inline-block'}),

    html.Div(children = [
        html.H6(children = 'Image: 9'),
        dcc.Upload(
                id='upload-9',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'display': 'inline-block',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
        ),html.Div(id='output-9-upload'),
        # html.Div(id='output-9-upload'),
        ],style={'width': '20%', 'display': 'inline-block'}),

    html.Hr(),
    
    #-------------------------------------------------------------------------------------
    
    html.H5('Data for Training (CSV)'),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'paper_bgcolor': 'lightblue'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
    
    #-------------------------------------------------------------------------------------
    
    html.Button('Submit', id='button-example-1'),
    html.Hr(),
    
    html.Div(children=[
        daq.LEDDisplay(
            id='recordt',
            #label="Default",
            value=0,
            label = "Number of Data",
            style={'width': '33%', 'display': 'inline-block'}
            ),

        daq.LEDDisplay(
            id='trainsett',
            #label="Default",
            value=0,
            label = "Data for Modeling",
            style={'width': '33%', 'display': 'inline-block'}
            ),

        daq.LEDDisplay(
            id='testsett',
            #label="Default",
            value=0,
            label = "Unseen Data For Predictions",
            style={'width': '33%', 'display': 'inline-block'}
            ),
    ]),

    html.Br(),
    
    html.Div(id='compare-model'),

    
    html.Hr(),
    
#----------------------------------------------------------------------
    
    html.H5('Image for Testing'),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
    html.Hr(),  
])

#-------------------------------------------------------------------------------------------------------------------------------------------------

@app.callback(Output('output-0-upload', 'children'),
              Input('upload-0', 'contents'),
              State('upload-0', 'filename'),
              State('upload-0', 'last_modified'))
def update_csv0(list_of_contents, list_of_names, list_of_dates):
    global df0
    if list_of_contents is not None:
        lst_columns = ['y']
        lst_number = []
        for l in range(1,28*28+1):
            lst_columns.append('pixel' + str(l))
        for c,n,d in zip(list_of_contents, list_of_names, list_of_dates):
            lst_number.append(img_csv_0(c, n, d))
        
        df0 = pd.DataFrame(lst_number, columns = lst_columns)
        return html.Div([
            html.Div('Number of image 0 : ' + str(df0.shape[0]))
            ])
    else:
        return html.Div([
            html.Div('Number of image 0 : ')
            ])

@app.callback(Output('output-1-upload', 'children'),
              Input('upload-1', 'contents'),
              State('upload-1', 'filename'),
              State('upload-1', 'last_modified'))
def update_csv1(list_of_contents, list_of_names, list_of_dates):
    global df1
    if list_of_contents is not None:
        lst_columns = ['y']
        lst_number = []
        for l in range(1,28*28+1):
            lst_columns.append('pixel' + str(l))
        for c,n,d in zip(list_of_contents, list_of_names, list_of_dates):
            lst_number.append(img_csv_1(c, n, d))
        
        df1 = pd.DataFrame(lst_number, columns = lst_columns)
        return html.Div([
            html.Div('Number of image 1 : ' + str(df1.shape[0]))
            ])
    else:
        return html.Div([
            html.Div('Number of image 1 : ')
            ])

@app.callback(Output('output-2-upload', 'children'),
              Input('upload-2', 'contents'),
              State('upload-2', 'filename'),
              State('upload-2', 'last_modified'))
def update_csv2(list_of_contents, list_of_names, list_of_dates):
    global df2
    if list_of_contents is not None:
        lst_columns = ['y']
        lst_number = []
        for l in range(1,28*28+1):
            lst_columns.append('pixel' + str(l))
        for c,n,d in zip(list_of_contents, list_of_names, list_of_dates):
            lst_number.append(img_csv_2(c, n, d))
        
        df2 = pd.DataFrame(lst_number, columns = lst_columns)
        return html.Div([
            html.Div('Number of image 2 : ' + str(df2.shape[0]))
            ])
    else:
        return html.Div([
            html.Div('Number of image 2 : ')
            ])

@app.callback(Output('output-3-upload', 'children'),
              Input('upload-3', 'contents'),
              State('upload-3', 'filename'),
              State('upload-3', 'last_modified'))
def update_csv3(list_of_contents, list_of_names, list_of_dates):
    global df3
    if list_of_contents is not None:
        lst_columns = ['y']
        lst_number = []
        for l in range(1,28*28+1):
            lst_columns.append('pixel' + str(l))
        for c,n,d in zip(list_of_contents, list_of_names, list_of_dates):
            lst_number.append(img_csv_3(c, n, d))
        
        df3 = pd.DataFrame(lst_number, columns = lst_columns)
        return html.Div([
            html.Div('Number of image 3 : ' + str(df3.shape[0]))
            ])
    else:
        return html.Div([
            html.Div('Number of image 3 : ')
            ])

@app.callback(Output('output-4-upload', 'children'),
              Input('upload-4', 'contents'),
              State('upload-4', 'filename'),
              State('upload-4', 'last_modified'))
def update_csv4(list_of_contents, list_of_names, list_of_dates):
    global df4
    if list_of_contents is not None:
        lst_columns = ['y']
        lst_number = []
        for l in range(1,28*28+1):
            lst_columns.append('pixel' + str(l))
        for c,n,d in zip(list_of_contents, list_of_names, list_of_dates):
            lst_number.append(img_csv_4(c, n, d))
        
        df4 = pd.DataFrame(lst_number, columns = lst_columns)
        return html.Div([
            html.Div('Number of image 4 : ' + str(df4.shape[0]))
            ])
    else:
        return html.Div([
            html.Div('Number of image 4 : ')
            ])

@app.callback(Output('output-5-upload', 'children'),
              Input('upload-5', 'contents'),
              State('upload-5', 'filename'),
              State('upload-5', 'last_modified'))
def update_csv5(list_of_contents, list_of_names, list_of_dates):
    global df5
    if list_of_contents is not None:
        lst_columns = ['y']
        lst_number = []
        for l in range(1,28*28+1):
            lst_columns.append('pixel' + str(l))
        for c,n,d in zip(list_of_contents, list_of_names, list_of_dates):
            lst_number.append(img_csv_5(c, n, d))
        
        df5 = pd.DataFrame(lst_number, columns = lst_columns)
        return html.Div([
            html.Div('Number of image 5 : ' + str(df5.shape[0]))
            ])
    else:
        return html.Div([
            html.Div('Number of image 5 : ')
            ])

@app.callback(Output('output-6-upload', 'children'),
              Input('upload-6', 'contents'),
              State('upload-6', 'filename'),
              State('upload-6', 'last_modified'))
def update_csv6(list_of_contents, list_of_names, list_of_dates):
    global df6
    if list_of_contents is not None:
        lst_columns = ['y']
        lst_number = []
        for l in range(1,28*28+1):
            lst_columns.append('pixel' + str(l))
        for c,n,d in zip(list_of_contents, list_of_names, list_of_dates):
            lst_number.append(img_csv_6(c, n, d))
        
        df6 = pd.DataFrame(lst_number, columns = lst_columns)
        return html.Div([
            html.Div('Number of image 6 : ' + str(df6.shape[0]))
            ])
    else:
        return html.Div([
            html.Div('Number of image 6 : ')
            ])

@app.callback(Output('output-7-upload', 'children'),
              Input('upload-7', 'contents'),
              State('upload-7', 'filename'),
              State('upload-7', 'last_modified'))
def update_csv7(list_of_contents, list_of_names, list_of_dates):
    global df7
    if list_of_contents is not None:
        lst_columns = ['y']
        lst_number = []
        for l in range(1,28*28+1):
            lst_columns.append('pixel' + str(l))
        for c,n,d in zip(list_of_contents, list_of_names, list_of_dates):
            lst_number.append(img_csv_7(c, n, d))
        
        df7 = pd.DataFrame(lst_number, columns = lst_columns)
        return html.Div([
            html.Div('Number of image 7 : ' + str(df7.shape[0]))
            ])
    else:
        return html.Div([
            html.Div('Number of image 7 : ')
            ])

@app.callback(Output('output-8-upload', 'children'),
              Input('upload-8', 'contents'),
              State('upload-8', 'filename'),
              State('upload-8', 'last_modified'))
def update_csv8(list_of_contents, list_of_names, list_of_dates):
    global df8
    if list_of_contents is not None:
        lst_columns = ['y']
        lst_number = []
        for l in range(1,28*28+1):
            lst_columns.append('pixel' + str(l))
        for c,n,d in zip(list_of_contents, list_of_names, list_of_dates):
            lst_number.append(img_csv_8(c, n, d))
        
        df8 = pd.DataFrame(lst_number, columns = lst_columns)
        return html.Div([
            html.Div('Number of image 8 : ' + str(df8.shape[0]))
            ])
    else:
        return html.Div([
            html.Div('Number of image 8 : ')
            ])

@app.callback(Output('output-9-upload', 'children'),
              Input('upload-9', 'contents'),
              State('upload-9', 'filename'),
              State('upload-9', 'last_modified'))
def update_csv9(list_of_contents, list_of_names, list_of_dates):
    global df9
    if list_of_contents is not None:
        lst_columns = ['y']
        lst_number = []
        for l in range(1,28*28+1):
            lst_columns.append('pixel' + str(l))
        for c,n,d in zip(list_of_contents, list_of_names, list_of_dates):
            lst_number.append(img_csv_9(c, n, d))
        
        df9 = pd.DataFrame(lst_number, columns = lst_columns)
        return html.Div([
            html.Div('Number of image 9 : ' + str(df9.shape[0]))
            ])
    else:
        return html.Div([
            html.Div('Number of image 9 : ')
            ])

#------------------------------------------------------------------------------

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

#------------------------------------------------------------------------------
    
@app.callback([Output('recordt', 'value'),
               Output('trainsett', 'value'),
               Output('testsett', 'value'),
               Output('compare-model', 'children')],
              Input('button-example-1', 'n_clicks'))
def update_output3(n_clicks):
    
    global dfall
    global results
    global model
    global dummyModel

    if ((df0 is not None) & (df1 is not None) & (df2 is not None) & (df3 is not None) & (df4 is not None) & (df5 is not None) & (df6 is not None) & (df7 is not None) & (df8 is not None) & (df9 is not None)) or (df is not None):
        if (df0 is not None) & (df1 is not None) & (df2 is not None) & (df3 is not None) & (df4 is not None) & (df5 is not None) & (df6 is not None) & (df7 is not None) & (df8 is not None) & (df9 is not None):
            dfall = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9], ignore_index=True)
        elif (df is not None):
            dfall = df
        
        X = dfall.iloc[:,1:]
        y = dfall['y']
        trainX, testX, trainy, testy = train_test_split(X, y, train_size= 80/100, random_state=42)
        
        exp_clf101 = setup(data = trainX, target = trainy)
        best = compare_models(include = ['lr','et','rf'])
        result = pull()
        results = pd.DataFrame(result)
        model = create_model(best)
        
        y_scores = model.predict_proba(trainX)
        y_onehot = pd.get_dummies(trainy, columns=model.classes_)

        fig_ROC = go.Figure()
        fig_ROC.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
            )
    
        for i in range(y_scores.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = y_scores[:, i]
        
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
        
            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        
        fig_ROC.update_layout(
            title_text='<i><b>ROC Curve</b></i>',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
        
        actual_labels = np.array(trainy)
        predicted_labels = model.predict(trainX)
        
        z = confusion_matrix(actual_labels, predicted_labels)
        
        # change each element of z to type string for annotations
        z_text = [[str(y) for y in x] for x in z]
        
        # set up figure 
        label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        fig_cfm = ff.create_annotated_heatmap(z, x=label, y=label, annotation_text=z_text, colorscale='Blues')
        
        # add title
        fig_cfm.update_layout(title_text='<i><b>Confusion matrix</b></i>\n',
                          #xaxis = dict(title='x'),
                          #yaxis = dict(title='x')
                         )
        
        # add custom xaxis title
        fig_cfm.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.5,
                                y=-0.1,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
        
        # add custom yaxis title
        fig_cfm.add_annotation(dict(font=dict(color="black",size=14),
                                x=-0.1,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        
        # adjust margins to make room for yaxis title
        fig_cfm.update_layout(margin=dict(t=50, l=200))
        
        # add colorbar
        fig_cfm['data'][0]['showscale'] = True
        
        #FOR TEST
        y_scores_test = model.predict_proba(testX)
        y_onehot_test = pd.get_dummies(testy, columns=model.classes_)

        fig_ROC_test = go.Figure()
        fig_ROC_test.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
            )
    
        for i in range(y_scores.shape[1]):
            y_true_test = y_onehot_test.iloc[:, i]
            y_score_test = y_scores_test[:, i]
        
            fpr_test, tpr_test, _ = roc_curve(y_true_test, y_score_test)
            auc_score_test = roc_auc_score(y_true_test, y_score_test)
        
            name_test = f"{y_onehot.columns[i]} (AUC={auc_score_test:.2f})"
            fig_ROC_test.add_trace(go.Scatter(x=fpr_test, y=tpr_test, name=name_test, mode='lines'))
        
        fig_ROC_test.update_layout(
            title_text='<i><b>ROC Curve</b></i>',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
        
        actual_labels_test = np.array(testy)
        predicted_labels_test = model.predict(testX)
        
        z_test = confusion_matrix(actual_labels_test, predicted_labels_test)
        
        # change each element of z to type string for annotations
        z_text_test = [[str(y) for y in x] for x in z_test]
        
        # set up figure 
        label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        fig_cfm_test = ff.create_annotated_heatmap(z_test, x=label, y=label, annotation_text=z_text_test, colorscale='Blues')
        
        # add title
        fig_cfm_test.update_layout(title_text='<i><b>Confusion matrix</b></i>\n',
                          #xaxis = dict(title='x'),
                          #yaxis = dict(title='x')
                         )
        
        # add custom xaxis title
        fig_cfm_test.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.5,
                                y=-0.1,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
        
        # add custom yaxis title
        fig_cfm_test.add_annotation(dict(font=dict(color="black",size=14),
                                x=-0.1,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        
        # adjust margins to make room for yaxis title
        fig_cfm_test.update_layout(margin=dict(t=50, l=200))
        
        # add colorbar
        fig_cfm_test['data'][0]['showscale'] = True
        
        acctrain = accuracy_score(np.array(trainy), model.predict(trainX))
        acctrain = round(acctrain, 4)
        pretrain = precision_score(np.array(trainy), model.predict(trainX), average='weighted')
        pretrain = round(pretrain, 4)
        rectrain = recall_score(np.array(trainy), model.predict(trainX), average='weighted')
        rectrain = round(rectrain, 4)

        acctest = accuracy_score(np.array(testy), model.predict(testX))
        acctest = round(acctest, 4)
        pretest = precision_score(np.array(testy), model.predict(testX), average='weighted')
        pretest = round(pretest, 4)
        rectest = recall_score(np.array(testy), model.predict(testX), average='weighted')
        rectest = round(rectest, 4)

        bestModel = html.Div([
            dash_table.DataTable(
                results.to_dict('records'),
                [{'name': i, 'id': i} for i in results.columns]
            ),
            html.Hr(),
            
            html.H5('Graph for Train set'),
            html.Div([
                dcc.Graph(
                    figure=fig_ROC,
                    style={'width': '50%', 'display': 'inline-block'}
                    ),
                dcc.Graph(
                    figure=fig_cfm,
                    style={'width': '50%', 'display': 'inline-block'}
                    ),
                ]),
            html.Hr(),
            
            html.Div([
                daq.LEDDisplay(
                    id='acctrain',
                    #label="Default",
                    value=acctrain,
                    label = "Accuracy Score for Train set",
                    style={'width': '33%', 'display': 'inline-block'}
                ),
        
                daq.LEDDisplay(
                    id='pretrain',
                    #label="Default",
                    value=pretrain,
                    label = "Precision Score for Train set",
                    style={'width': '33%', 'display': 'inline-block'}
                    ),
                        
                daq.LEDDisplay(
                    id='rectrain',
                    #label="Default",
                    value=rectrain,
                    label = "Recall Score for Train set",
                    style={'width': '33%', 'display': 'inline-block'}
                    ),
                html.Hr(),
            ]),
            
            html.H5('Graph for Test set'),
            html.Div([
                dcc.Graph(
                    figure=fig_ROC_test,
                    style={'width': '50%', 'display': 'inline-block'}
                    ),
                dcc.Graph(
                    figure=fig_cfm_test,
                    style={'width': '50%', 'display': 'inline-block'}
                    ),
                ]),
            html.Hr(),
            
            html.Div([
                daq.LEDDisplay(
                    id='acctest',
                    #label="Default",
                    value=acctest,
                    label = "Accuracy Score for Test set",
                    style={'width': '33%', 'display': 'inline-block'}
                ),
        
                daq.LEDDisplay(
                    id='pretest',
                    #label="Default",
                    value=pretest,
                    label = "Precision Score for Test set",
                    style={'width': '33%', 'display': 'inline-block'}
                    ),
                        
                daq.LEDDisplay(
                    id='rectest',
                    #label="Default",
                    value=rectest,
                    label = "Recall Score for Test set",
                    style={'width': '33%', 'display': 'inline-block'}
                    ),
            ]),
            ])
        
        return X.shape[0],trainX.shape[0],testX.shape[0],bestModel
    
    else:
        return 0,0,0,dummyModel

#----------------------------------------------------------------------

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output2(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents2(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

#-------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False) #, use_reloader=False
