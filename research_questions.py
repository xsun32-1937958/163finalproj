'''
CSE 163 AH
Final Project (Research Questions)
Shiyu Han, Zhiheng Liu, Selina Sun
This file utilizes the Indian stock market dataset and conducts data analysis
via data visualizations and machine learning models. The main purpose is to
examine trend of stock prices within the Indian IT industry and to create
machine learning models to predict stock prices for a particular stock HCLTECH.
Libraries Plotly, Sklearn, Tensorflow, Numpy and Math are used.
'''
import data_preprocessing as prep
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math


def plot_cp_vol_HCLTECH(HCLTECH):
    '''
    This function addresses the First research question:
    it plots trend of stock HCLTECH's close price and volume over the years
    '''
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=HCLTECH['Date'],
                         y=HCLTECH['Volume'],
                         name='Volume'),
                  secondary_y=True)

    fig.add_trace(go.Scatter(x=HCLTECH['Date'],
                             y=HCLTECH['Close'],
                             name='Close Price',
                             mode="lines"),
                  secondary_y=False)

    fig.update_xaxes(title_text='Year')

    # Set y-axes titles and plot title
    fig.update_yaxes(title_text='Close Price', secondary_y=False)
    fig.update_yaxes(title_text='Volume', secondary_y=True)
    fig.update_layout(title={
        'text': "HCLTECH's Close Price & Volume Over Time",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        yaxis_title='Close Price',
        xaxis_title='Time')
    plot(fig)


def calc_max_daily_change(HCLTECH):
    '''
    This function finds the largest single-day increase and single-day
    decrease in close price and returns max increase, decrease, and the
    corresponding date
    '''
    HCLTECH['Daily_DIFF'] = HCLTECH['Close']-HCLTECH['Prev Close']
    MAX_daily_increase = max(HCLTECH['Daily_DIFF'])
    MIN_daily_increase = min(HCLTECH['Daily_DIFF'])
    MAX_index = HCLTECH['Daily_DIFF'].nlargest(1).index[0]
    MIN_index = HCLTECH['Daily_DIFF'].nsmallest(1).index[0]
    MAX_Date = HCLTECH.loc[MAX_index, 'Date']
    MIN_Date = HCLTECH.loc[MIN_index, 'Date']
    return MAX_daily_increase, MAX_Date, MIN_daily_increase, MIN_Date


def plot_IT(IT_df):
    '''
    This function plots the stock close price for the entire IT industry
    '''
    # find average of 5 stocks' close price over time
    avg_prices = IT_df.groupby('Date')['Close'].mean()

    fig2 = px.line(IT_df, x='Date', y='Close', color='Symbol')

    fig2.add_trace(go.Scatter(x=avg_prices.index,
                              y=avg_prices,
                              name='IT Industry',
                              mode='lines',
                              line=dict(color='firebrick')))

    fig2.update_layout(title={
        'text': "Stock Prices in IT Industry (2016-2019)",
        'y': 0.94,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        yaxis_title='Close Price',
        xaxis_title='Time',
        legend_title='Stock')
    plot(fig2)


def plot_all(all_50, meta_data):
    '''
    This function plots and compares the IT industry with other industries
    '''
    all_50 = prep.set_time(all_50)
    all_50 = all_50.merge(meta_data, left_on='Symbol', right_on='Symbol')
    avgs_all = all_50.groupby(['Industry', 'Date'])['Close'].mean()
    avgs_all = avgs_all.to_frame().reset_index()
    fig3 = px.line(avgs_all, x='Date', y='Close', color='Industry')

    fig3.update_layout(title={'text': "Stock Prices in India (2016-2019)",
                              'y': 0.94,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    plot(fig3)


def ml_prep(HCLTECH):
    '''
    This function resets the index so that the data is clear and saves the
    other data. It returns the reset dataset and the transforming scaler
    '''
    print(HCLTECH)
    HCLTECH = HCLTECH.reset_index()['Close']
    print(HCLTECH)
    scaler = MinMaxScaler(feature_range=(0, 1))
    HCLTECH = scaler.fit_transform(np.array(HCLTECH).reshape(-1, 1))
    return HCLTECH, scaler


def ml_split(HCLTECH):
    '''
    This function divides data into train data and test data by 7:3 ratio
    '''
    training_size = int(len(HCLTECH)*0.7)
    train_data, test_data = HCLTECH[0:training_size, :], \
        HCLTECH[training_size:len(HCLTECH), :1]
    return train_data, test_data


def get_train_test(train_data, test_data, time_step):
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train, Y_train, X_test, Y_test


def create_dataset(dataset, time_step=1):
    '''
    this function splits the data into X, Y
    In the 0th iteration, the first time_step elements goes as the first
    record and the (time_step+1) elements will be put up in the X.
    The time_step elements will be put up in the Y.
    '''
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def lstm_build(less_layers):
    '''
    This function uses a sequential model and add the layers of the LSTM model
    '''
    model = Sequential()
    if not less_layers:
        model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, input_shape=(100, 1)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_absolute_error'])
    return model


def LSTM_ML(X_train, Y_train, X_test, Y_test, scaler, less_layers):
    '''
    This function builds and trains the LSTM model
    Then it predicts the test values and performs the scaler inverse transform
    tp evalue model for obtaining error and prints the Root Mean Squared Error
    finally it returns the prediced test values and the untransformed test
    values
    '''
    model = lstm_build(less_layers)
    model.fit(X_train,
              Y_train,
              epochs=10,
              validation_data=(X_test, Y_test),
              verbose=1)
    test_predicted = model.predict(X_test)
    score = model.evaluate(X_test, Y_test, batch_size=1, verbose=0)
    test_predicted = scaler.inverse_transform(test_predicted)
    Y_test = Y_test.reshape(1, -1)
    Y_test_untransformed = scaler.inverse_transform(Y_test)
    Mean_Sqaured_Error = score[0]
    Root_Mean_Squared_Error = math.sqrt(Mean_Sqaured_Error)
    print(Root_Mean_Squared_Error)
    return test_predicted, Y_test_untransformed


def plot_predict_cp(test_predicted, Y_test_untransformed, title):
    '''
    This function plots the true v.s. predicted close prices for test dataset
    '''
    fig4 = px.line(test_predicted)
    fig4['data'][0]['showlegend'] = True
    fig4['data'][0]['name'] = 'Close Prices Predicted'
    fig4.add_trace(go.Scatter(y=Y_test_untransformed[0],
                              mode='lines',
                              line=dict(color='firebrick'),
                              name='Close Prices'))
    fig4.update_layout(title={'text': "LSTM Prediction of Stock Prices",
                              'y': 0.94,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                       yaxis_title='Close Price',
                       xaxis_title='Time Step',
                       legend_title=None,
                       )
    plot(fig4)


def main():
    # gather all processed data
    meta_data, df_list, all_50 = prep.csv_to_df()
    HCLTECH, INFY, TCS, TECHM, WIPRO, IT_df = prep.get_IT_df(meta_data,
                                                             df_list)

    # First Research Question:
    # plot_cp_vol_HCLTECH(HCLTECH)
    MAX_daily_increase, MAX_Date, \
        MIN_daily_increase, MIN_Date = calc_max_daily_change(HCLTECH)
    print(MAX_Date)
    print(MAX_daily_increase)
    print(MIN_Date)
    print(MIN_daily_increase)

    # Second Research Question:
    # plot_IT(IT_df)
    # plot_all(all_50, meta_data)

    # Third Research Question:
    HCLTECH, scaler = ml_prep(HCLTECH)
    train_data, test_data = ml_split(HCLTECH)
    time_step = 100
    X_train, Y_train, \
        X_test, Y_test = get_train_test(train_data, test_data, time_step)
    # first lstm model with 4 layers
    title = "LSTM Prediction of Stock Prices (4 layers)"
    # test_predicted, Y_test_untransformed = LSTM_ML(X_train,
    #                                               Y_train,
    #                                               X_test,
    #                                               Y_test,
    #                                               scaler,
    #                                               False)
    # plot_predict_cp(test_predicted, Y_test_untransformed, title)
    # second lstm model with 2 layers
    title = "LSTM Prediction of Stock Prices (2 layers)"
    test_predicted_2, Y_test_untransformed_2 = LSTM_ML(X_train,
                                                       Y_train,
                                                       X_test,
                                                       Y_test,
                                                       scaler,
                                                       True)
    plot_predict_cp(test_predicted_2, Y_test_untransformed_2, title)


if __name__ == '__main__':
    main()
