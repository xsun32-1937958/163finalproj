#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:15:08 2022

@author: hsy
"""
import os
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

directory_csv = 'csv_files'
filenames_csv = os.listdir(directory_csv)
filenames_csv.sort()
df_list = []

# Converting the csv files in the directory into dataframes
for filename in filenames_csv:
        path = os.path.join(directory_csv, filename)
        company = filename[0:-4]
        with open(path) as f:
            df = pd.read_csv(path)
            if len(df) != 0:
                df['name'] = company
                df_list.append(df)

meta_data = df_list[-1]

# get the names and index of companies that are in the IT industry
IT_data = meta_data[meta_data['Industry'] == 'IT']
IT_series = IT_data['Symbol']


# get individual dataframe for companies that are in the IT industry
HCLTECH = df_list[IT_series[IT_series == 'HCLTECH'].index[0]]
INFY = df_list[IT_series[IT_series == 'INFY'].index[0]]
TCS = df_list[IT_series[IT_series == 'TCS'].index[0]]
TECHM = df_list[IT_series[IT_series == 'TECHM'].index[0]]
WIPRO = df_list[IT_series[IT_series == 'WIPRO'].index[0]]

# filter data for pre-Covid 19 time
def before_covid(df):
    before_covid = df['Date'] <= '2019-12-31'
    df = df.loc[before_covid]
    return df

HCLTECH = before_covid(HCLTECH)

# Dropping unuseful columns and rows
HCLTECH = HCLTECH[['Date', 'Open', 'High', 'Low', 'Last', \
                   'Close', 'VWAP', 'Volume', 'Turnover', 'Trades']]

# First research question: trend of stock HCLTECH's close price and volume over
# the years
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=HCLTECH['Date'],
                         y=HCLTECH['Close'],
                         name='Close Price',
                         mode="lines"),
              secondary_y=True)

fig.add_trace(go.Bar(x=HCLTECH['Date'],
                     y=HCLTECH['Volume'],
                     name='Volume'),
              secondary_y=False)

fig.update_xaxes(title_text='Year')

# Set y-axes titles and plot title
fig.update_yaxes(title_text='Close Price', secondary_y=False)
fig.update_yaxes(title_text='Volume', secondary_y=True)
fig.update_layout(title={'text': "HCLTECH's Close Price & Volume Over Time",
                         'y':0.9,
                         'x':0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'})
plot(fig)

# Find the largest single-day increase and single-day decrease in close price
