#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:15:08 2022

@author: hsy
"""
import os
import pandas as pd

directory_csv = '/Users/hsy/Desktop/CSE163/163finalproj/csv_files'
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

# Dropping unuseful columns and rows
HCLTECH = HCLTECH[['Date', 'Open', 'High', 'Low', 'Last', \
                   'Close', 'VWAP', 'Volume', 'Turnover', 'Trades']]
