import numpy as np
import pandas_datareader as web
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
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

# %%

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM
Dropout = keras.layers.Dropout

# %%

STOCK_NAME = "ABEV3.SA"
df = web.DataReader(STOCK_NAME, data_source="yahoo", start="2015-01-01", end="2019-12-31")
df

# %%

df.shape

# %%

df_bovespa = web.DataReader("^BVSP", data_source="yahoo", start="2015-01-01", end="2019-12-31")
# df_bovespa.reset_index(inplace=True,drop=False)
df_bovespa.tail

# %%

df_bovespa.shape

# %%

df = df.merge(df_bovespa, left_on='Date', right_on='Date', suffixes=('', '_bovespa'))
df.describe()

# %%

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

# %%

# %%

features_4 = ['Close', 'Volume', 'Close_bovespa', 'Volume_bovespa']
features_25 = ['Close', 'Volume', 'Close_bovespa', 'Volume_bovespa', 'MA3', 'MA7', 'MA20', 'MA30', 'MA60', 'CMA3', 'ST3', 'ST7', 'ST20', 'ST30', 'ST60', 'UPPER3', 'LOWER3', 'UPPER7', 'LOWER7', 'UPPER20', 'LOWER20', 'UPPER30', 'LOWER30', 'UPPER60', 'LOWER60']
features_17 = ['Close', 'Volume', 'Close_bovespa', 'Volume_bovespa', 'MA3', 'MA30', 'MA60', 'CMA3', 'ST3', 'ST30',
               'ST60', 'UPPER3', 'LOWER3', 'UPPER30', 'LOWER30', 'UPPER60', 'LOWER60']
features_9 = ['Close', 'Volume', 'Close_bovespa', 'Volume_bovespa', 'MA3', 'CMA3', 'ST3', 'UPPER3', 'LOWER3']
dolar = ['High','Low','Open', 'Close', 'Close_bovespa', 'High_bovespa', 'Low_bovespa', 'Open_bovespa', 'MA50', 'MA7', 'MA200']
features_1 = ["Close"]
models_array = []
features_array = [features_1 , features_4, features_17, features_9, features_25]

for feature in features_array:
    plot_titles = STOCK_NAME[:-3] + ' ' + str(len(feature)) + " Features"
    data = df.filter(feature)
    dataset = data.values
    # training_data_len = math.ceil(len(dataset) * .7)

    training_data_len = len(data.loc['2015-01-01':'2018-12-31'])
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_data = sc.fit_transform(dataset)

    len(scaled_data)

    train_data = scaled_data[0:training_data_len, :]
    print(train_data)
    window = 60
    X_train = []
    y_train = []
    for i in range(window, len(train_data)):
        X_train.append(train_data[i - window:i, :])
        y_train.append(train_data[i, 0:1])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], -1))

    # Inicializar a RNN
    regressor = Sequential()

    # Adicionar a primeira camada LSTM e Dropout
    regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.2))

    # Adicionar a segunda camada LSTM e Dropout
    regressor.add(LSTM(units=80, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Adicionar a terceira camada LSTM e Dropout
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    # camada de saída
    regressor.add(Dense(units=1))

    # Compilar a rede
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    # regressor.compile(loss = 'mean_squared_error')

    # Visualizar a rede
    regressor.summary()

    history = regressor.fit(X_train, y_train, epochs=50, batch_size=1)

    plt.plot(history.history['loss'], label='loss rate')
    plt.title(plot_titles)
    plt.legend()

    test_data = scaled_data[training_data_len - window:, :]
    # print(len(scaled_data))
    # print(len(test_data))
    X_test = []
    y_test = dataset[training_data_len:, 0:1]

    for i in range(window, len(test_data)):
        #     print(i)
        X_test.append(test_data[i - window:i, :])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], -1))
    # print(X_test.shape, X_train.shape)

    # %%

    predicted = regressor.predict(X_test)
    volume = dataset[training_data_len:, 1:]
    predicted = np.column_stack((predicted, volume))
    # print(dataset[:,1:])
    print(predicted.shape)
    predicted = sc.inverse_transform(predicted)
    print(predicted)

    # %%

    allTargetData = np.vstack((dataset[:training_data_len, 0:1], dataset[training_data_len:, 0:1]))
    training_predicted = regressor.predict(X_train)
    volume = dataset[:len(X_train), 1:]
    training_predicted = np.column_stack((training_predicted, volume))
    training_predicted = sc.inverse_transform(training_predicted)
    allForecastedData = np.vstack((dataset[0:window, 0:1], training_predicted[:, 0:1], predicted[:, 0:1]))
    # date = df['DATA']
    date = df.index

    figure(figsize=(20, 6), dpi=80)
    plt.plot(date, allForecastedData, color='red', label='Previsto')
    plt.plot(date, allTargetData, color='blue', label='Real')
    plt.title('Previsão de série temporal - ' + plot_titles)
    plt.xlabel('Tempo')
    plt.ylabel('Cotação (R$)')
    plt.legend()
    plt.show()

    # %%


    rmse = math.sqrt(mean_squared_error(dataset[training_data_len:, 0:1], predicted[:, 0:1]))
    print('RMSE: ', str(rmse).replace('.', ','))

    mse = mean_squared_error(dataset[training_data_len:, 0:1], predicted[:, 0:1])
    print('MSE: ', str(mse).replace('.', ','))

    mape = np.mean(np.abs((dataset[training_data_len:, 0:1] - predicted[:, 0:1]) / dataset[training_data_len:, 0:1])) * 100
    print('MAPE:  ', str(mape).replace('.', ','), '%')

    r2 = r2_score(predicted[:, 0:1], dataset[training_data_len:, 0:1])
    print('R2: ', str(r2).replace('.', ','))

    fig, ax = plt.subplots(1, 1)

    data = [[rmse], [mse], [mape], [r2]]
    row_labels = ["RMSE", "MSE", "MAPE", "R2"]
    plt.rcParams["figure.figsize"] = [4.0, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=data, rowLabels=row_labels, colLabels=["Metrics"], loc="center")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    plt.title(plot_titles)
    plt.show()
