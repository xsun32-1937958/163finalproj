'''
CSE 163 AH
Final Project (Testing)
Shiyu Han, Zhiheng Liu, Selina Sun
This file serves as a test for the machine learning model developed. Another
IT stock TCS is explored and predicted here.
'''
import data_preprocessing as prep
import research_questions as res


def main():
    # gather all processed data
    meta_data, df_list, all_50 = prep.csv_to_df()
    HCLTECH, INFY, TCS, TECHM, WIPRO, IT_df = prep.get_IT_df(meta_data,
                                                             df_list)
    # reproduce the model for stock
    TCS, scaler = res.ml_prep(TCS)
    train_data, test_data = res.ml_split(TCS)
    time_step = 100
    X_train, Y_train, \
        X_test, Y_test = res.get_train_test(train_data, test_data, time_step)
    # first lstm model with 4 layers
    title = "LSTM Prediction of Stock Prices (4 layers)"
    test_predicted, Y_test_untransformed = res.LSTM_ML(X_train,
                                                       Y_train,
                                                       X_test,
                                                       Y_test,
                                                       scaler,
                                                       False)
    res.plot_predict_cp(test_predicted, Y_test_untransformed, title)
    # second lstm model with 2 layers
    title = "LSTM Prediction of Stock Prices (2 layers)"
    test_predicted_2, Y_test_untransformed_2 = res.LSTM_ML(X_train,
                                                           Y_train,
                                                           X_test,
                                                           Y_test,
                                                           scaler,
                                                           True)
    res.plot_predict_cp(test_predicted_2, Y_test_untransformed_2, title)


if __name__ == '__main__':
    main()
