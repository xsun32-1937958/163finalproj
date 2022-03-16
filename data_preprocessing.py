'''
CSE 163 AH
Final Project (Data Processing)
Shiyu Han, Zhiheng Liu, Selina Sun
This file utilizes the Indian stock market dataset and conducts preliminary
data processing for later data analysis. The Pandas library is used here.
'''
import os
import pandas as pd


def csv_to_df():
    '''
    This function converts the csv files in the directory into dataframes
    and returns a list that contains all the df files
    '''
    directory_csv = 'csv_files'
    all_50 = pd.read_csv('./csv_files/NIFTY50_all.csv')
    filenames_csv = os.listdir(directory_csv)
    filenames_csv.sort()
    df_list = []
    for filename in filenames_csv:
        path = os.path.join(directory_csv, filename)
        company = filename[0:-4]
        with open(path):
            df = pd.read_csv(path)
            if len(df) != 0:
                df['name'] = company
                df_list.append(df)
    meta_data = df_list[-1]
    return meta_data, df_list, all_50


def get_IT_df(meta_data, df_list):
    '''
    This function gets individual dataframe for companies that are in
    the IT industry and returns each IT stock in dataframe
    '''
    IT_data = meta_data[meta_data['Industry'] == 'IT']
    IT_series = IT_data['Symbol']
    HCLTECH = df_list[IT_series[IT_series == 'HCLTECH'].index[0]]
    INFY = df_list[IT_series[IT_series == 'INFY'].index[0]-1]
    TCS = df_list[IT_series[IT_series == 'TCS'].index[0]]
    TECHM = df_list[IT_series[IT_series == 'TECHM'].index[0]]
    WIPRO = df_list[IT_series[IT_series == 'WIPRO'].index[0]]
    # get combined or individual dataframe
    IT_df_temp = [HCLTECH, INFY, TCS, TECHM, WIPRO]
    IT_df = pd.concat(IT_df_temp)
    HCLTECH = set_time(HCLTECH)
    INFY = set_time(INFY)
    TCS = set_time(TCS)
    TECHM = set_time(TECHM)
    WIPRO = set_time(WIPRO)
    IT_df = set_time(IT_df)
    # Dropping unuseful columns and rows for HCLTECH
    HCLTECH = HCLTECH[['Date', 'Open', 'High', 'Low', 'Last', 'Prev Close',
                       'Close', 'VWAP', 'Volume', 'Turnover', 'Trades']]
    return HCLTECH, INFY, TCS, TECHM, WIPRO, IT_df


def set_time(df):
    '''
    This function filters and returns data for year 2010-2018 (covid-19 imapct
    eliminated)
    '''
    time_mask = (df['Date'] <= '2018-12-31') & (df['Date'] >= '2010-1-1')
    df = df[time_mask]
    return df


def main():
    meta_data, df_list, all_50 = csv_to_df()

    # get the separate dataframe for each stock in the IT industry
    HCLTECH, INFY, TCS, TECHM, WIPRO, IT_df = get_IT_df(meta_data, df_list)


if __name__ == '__main__':
    main()
