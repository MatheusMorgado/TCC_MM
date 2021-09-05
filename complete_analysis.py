import numpy as np
import pandas_datareader as web
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from matplotlib.pyplot import figure
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow import keras


'''from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout'''

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM
GRU  = keras.layers.GRU
SGD  = keras.optimizers.SGD
Dropout = keras.layers.Dropout

WINDOW         = 60
TRAINING_START = '2015-01-01'
TRAINING_END   = '2018-12-31'
TESTING_START  = '2019-01-01'
TESTING_END    = '2019-12-31'


def stack_data(data_frame, window=WINDOW, test=False):

    if test:
        x_test = []
        for i in range(window, len(data_frame)):
            x_test.append(data_frame[i - window:i, :])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], -1))

        return {'x_test': x_test}

    else:
         x_train = []
         y_train = []
         for i in range(window, len(data_frame)):
             x_train.append(data_frame[i - window:i, :])
             y_train.append(data_frame[i, 0:1])
         x_train, y_train = np.array(x_train), np.array(y_train)
         x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], -1))

         return {'x_train': x_train, 'y_train': y_train}


def merge_data_frames(primary, secondary, left_on='Date', right_on='Date', suffixes=('', '_bovespa')):
    merged_data = primary.merge(secondary, left_on=left_on, right_on=right_on, suffixes=suffixes)
    return merged_data


def fetch_dataframe(name="^BVSP", start=TRAINING_START, end=TESTING_END, data_source="yahoo"):
    return web.DataReader(name, data_source=data_source, start=start, end=end)


def filters():
    return {
        "4_features": ['Close', 'Volume', 'Close_bovespa', 'Volume_bovespa'],

        "25_features": ['Close', 'Volume', 'Close_bovespa', 'Volume_bovespa', 'MA3', 'MA7', 'MA20', 'MA30', 'MA60',
                        'CMA3',
                        'ST3', 'ST7', 'ST20', 'ST30', 'ST60', 'UPPER3', 'LOWER3', 'UPPER7', 'LOWER7', 'UPPER20',
                        'LOWER20',
                        'UPPER30', 'LOWER30', 'UPPER60', 'LOWER60'],

        "17_features": ['Close', 'Volume', 'Close_bovespa', 'Volume_bovespa', 'MA3', 'MA30', 'MA60', 'CMA3', 'ST3',
                        'ST30',
                        'ST60', 'UPPER3', 'LOWER3', 'UPPER30', 'LOWER30', 'UPPER60', 'LOWER60'],

        "9_features": ['Close', 'Volume', 'Close_bovespa', 'Volume_bovespa', 'MA3', 'CMA3', 'ST3', 'UPPER3', 'LOWER3'],

        "1_features": ["Close"]
    }


def get_filter(name):
    return filters().get(name)

def all_datasets():
    return {
        'ABEV3.SA': fetch_dataframe(name='ABEV3.SA'),
        'BBDC4.SA': fetch_dataframe(name='BBDC4.SA'),
        'ITUB4.SA': fetch_dataframe(name='ITUB4.SA'),
        'VALE3.SA': fetch_dataframe(name='VALE3.SA')
    }

def get_dataset(name):
    all_data = all_datasets()
    return all_data.get(name)


def apply_new_features(df):

    df['MA3'] = df['Close'].rolling(window=3, min_periods=0).mean()
    df['MA7'] = df['Close'].rolling(window=7, min_periods=0).mean()
    df['MA20'] = df['Close'].rolling(window=20, min_periods=0).mean()
    df['MA30'] = df['Close'].rolling(window=30, min_periods=0).mean()
    df['MA60'] = df['Close'].rolling(window=60, min_periods=0).mean()

    df['MA50'] = df['Close'].rolling(window=50, min_periods=0).mean()
    df['MA200'] = df['Close'].rolling(window=200, min_periods=0).mean()

    cma3 = df['Close'].expanding(min_periods=3).mean()
    df['CMA3'] = cma3.fillna(df['MA3'])

    df['ST3'] = df['Close'].rolling(window=3, min_periods=0).std(ddof=0)
    df['ST7'] = df['Close'].rolling(window=7, min_periods=0).std(ddof=0)
    df['ST20'] = df['Close'].rolling(window=20, min_periods=0).std(ddof=0)
    df['ST30'] = df['Close'].rolling(window=30, min_periods=0).std(ddof=0)
    df['ST60'] = df['Close'].rolling(window=60, min_periods=0).std(ddof=0)

    df['UPPER3'] = df['MA3'] + (df['ST3'] * 2)
    df['LOWER3'] = df['MA3'] - (df['ST3'] * 2)

    df['UPPER7'] = df['MA7'] + (df['ST7'] * 2)
    df['LOWER7'] = df['MA7'] - (df['ST7'] * 2)

    df['UPPER20'] = df['MA20'] + (df['ST20'] * 2)
    df['LOWER20'] = df['MA20'] - (df['ST20'] * 2)

    df['UPPER30'] = df['MA30'] + (df['ST30'] * 2)
    df['LOWER30'] = df['MA30'] - (df['ST30'] * 2)

    df['UPPER60'] = df['MA60'] + (df['ST60'] * 2)
    df['LOWER60'] = df['MA60'] - (df['ST60'] * 2)

    return df


def build_lst_model(input_data):
    assert type(input_data) == type({})

    x_train_shape = input_data.get('x_train').shape

    regressor = Sequential()

    regressor.add(LSTM(units=100, return_sequences=True, input_shape=(x_train_shape[1], x_train_shape[2])))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=80, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.summary()

    return regressor


def build_gru_model(input_data):

    x_train_shape = input_data.get('x_train').shape


    regressor = Sequential()
    regressor.add(
        GRU(units=50, return_sequences=True, input_shape=(x_train_shape[1], x_train_shape[2]), activation='tanh'))
    regressor.add(Dropout(0.2))

    regressor.add(
        GRU(units=50, return_sequences=True, input_shape=(x_train_shape[1], x_train_shape[2]), activation='tanh'))
    regressor.add(Dropout(0.2))

    regressor.add(
        GRU(units=50, return_sequences=True, input_shape=(x_train_shape[1], x_train_shape[2]), activation='tanh'))
    regressor.add(Dropout(0.2))

    regressor.add(GRU(units=50, activation='tanh'))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))
    regressor.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')
    regressor.summary()

    return regressor


def fit_model(model, data_to_fit, epochs=50, batch_size=1):
    return model.fit(data_to_fit.get('x_train'), data_to_fit.get('y_train'), epochs=epochs, batch_size=batch_size)


def time_slice(data_frame, start=TRAINING_START, end=TESTING_END):
    return data_frame.loc[start:end]


def scale_dataframe(data_frame, low=0, high=1):
    sc = MinMaxScaler(feature_range=(low, high))
    return sc.fit_transform(data_frame)


def stocks():
    return {
        '^BVSP': 'Indice Bovespa',
        'ABEV3.SA': 'AMBEV',
        'BBDC4.SA': 'Banco Bradesco',
        'ITUB4.SA': 'Itau Unibanco',
        'VALE3.SA': 'Vale'
    }


def real_stock_name(name):
    return stocks().get(name)


def apply_filter(df, filter_name):
    return df.filter(filter_name)


bovespa_data = fetch_dataframe()

for i in range(len(all_datasets())):
    stock_name = list(all_datasets().keys())[i]
    print(stock_name)
    raw_data = get_dataset(stock_name)
    merged_data = merge_data_frames(raw_data, bovespa_data)
    data_with_features = apply_new_features(merged_data)

    for columns_filter in filters().keys():
        filtered_data = apply_filter(data_with_features, filters().get(columns_filter))
        sliced_data_frame = time_slice(filtered_data, end=TRAINING_END)

        filtered_values = sliced_data_frame.values

        scaled_values_dataset = scale_dataframe(filtered_values)

        stacked_data = stack_data(scaled_values_dataset)

        lstm_net = build_lst_model(stacked_data)
        gru_net = build_gru_model(stacked_data)

        trained_lstm_net = fit_model(lstm_net, stacked_data, epochs=50, batch_size=1)
        trained_gru_net = fit_model(gru_net, stacked_data, epochs=25, batch_size=32)

        lstm_path = 'LSTM\\' + real_stock_name(stock_name) + '\\' + str(columns_filter)
        gru_path  =  'GRU\\' + real_stock_name(stock_name) + '\\' + str(columns_filter)

        lstm_net.save(lstm_path)
        gru_net.save(gru_path)


        # modelagem = keras.models.load_model(path)
        #
        # test_data_values = time_slice(filtered_data, start=TESTING_START, end=TESTING_END).values
        # test_data_scaled = scale_dataframe(test_data_values)
        # stacked_test_data = stack_data(test_data_scaled, test=True)
        



# get_filters().value()
# ab    = merge_data_frames(ambev_data, bovespa_data)
# abb   = merge_data_frames(ab, bradesco_data, suffixes=('', '_bradesco'))
# abbi  = merge_data_frames(abb, itub_data, suffixes=('', '_itub'))
# abbiv = merge_data_frames(abbi, vale_data, suffixes=('', '_vale'))
