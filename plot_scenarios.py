import pandas as pd

from tcc_library import *

bovespa_data = fetch_dataframe()

for i in range(len(all_datasets())):
    stock_name = list(all_datasets().keys())[i]
    print(stock_name)
    raw_data = get_dataset(stock_name)
    merged_data = merge_data_frames(raw_data, bovespa_data)
    data_with_features = apply_new_features(merged_data)

    for columns_filter in filters().keys():
        filtered_data = apply_filter(data_with_features, filters().get(columns_filter))

        sc = MinMaxScaler(feature_range=(0, 1))
        scaled_values = sc.fit_transform(filtered_data.values)

        scaled_dataframe = pd.DataFrame(scaled_values, index=filtered_data.index, columns=filtered_data.columns)

        test_data = time_slice(scaled_dataframe, start=TESTING_START, end=TESTING_END, test=True)
        training_data  = time_slice(scaled_dataframe, start=TRAINING_START, end=TRAINING_END)

        stacked_test_data = stack_data(test_data.values, test=True)
        stacked_training_data = stack_data(training_data.values)

        x_test = stacked_test_data.get('x_test')
        x_train = stacked_training_data.get('x_train')

        lstm_path = 'LSTM\\' + real_stock_name(stock_name) + '\\' + str(columns_filter)
        gru_path  =  'GRU\\' + real_stock_name(stock_name) + '\\' + str(columns_filter)
        path = real_stock_name(stock_name) +' '+str(columns_filter)
        lstm_net = keras.models.load_model(lstm_path)
        # gru_net = keras.models.load_model(gru_path)

        predicted_lstm = lstm_net.predict(x_test)

        # predicted_gru  = gru_net.predict(x_test)

        training_data_len = len(stacked_training_data.get('x_train'))
        testing_data_len = len(stacked_test_data.get('x_test'))

        predicted_lstm_scaled_out = sc.inverse_transform(
            np.column_stack(
                (predicted_lstm, test_data[test_data.columns[1:]].values[60:, :])
            )
        )

        training_lstm_predicted_scaled = lstm_net.predict(x_train)
        training_lstm_predicted = sc.inverse_transform(np.column_stack(
            (training_lstm_predicted_scaled, training_data[training_data.columns[1:]].values[60:, :])
            )
        )

        # predicted_gru_scaled_out = sc.inverse_transform(
        #     np.column_stack(
        #           (predicted_gru, test_data[test_data.columns[1:]].values[60:, :])
        #     )
        # )
        # training_gru_predicted_scaled = gru_net.predict(x_train)
        #
        # training_gru_predicted = sc.inverse_transform(
        #     np.column_stack(
        #         (training_gru_predicted_scaled, training_data[training_data.columns[1:]].values[60:, :])
        #     )
        # )


        all_lstm_forecasted_data = np.vstack((filtered_data.values[0:60, 0:1], training_lstm_predicted[:, 0:1], predicted_lstm_scaled_out[:, 0:1]))

        # all_gru_forecasted_data = np.vstack((filtered_data.values[0:60, 0:1], training_gru_predicted[:, 0:1], predicted_gru_scaled_out[:, 0:1]))

        all_target_data = filtered_data['Close'].values
        date = filtered_data.index

        # plot_double_price(date, y_real=all_target_data, y_pred_lstm=all_lstm_forecasted_data,
        #            y_pred_gru=all_gru_forecasted_data,
        #            title='Forecasting\\'+path.replace("\\", ' ')
        #            )

        # plot_price(date, y_real=all_target_data, y_pred=all_lstm_forecasted_data, title=lstm_path.replace('\\', ' '))
        # plot_price(date, y_real=all_target_data, y_pred=all_gru_forecasted_data, title=gru_path.replace('\\', ' '))

        test_mape = test_data['Close'].values[60:]

        mape_gru = np.mean(np.abs((test_mape - predicted_gru_scaled_out[:, 0:1]) / test_mape)) * 100
        mape_lstm = np.mean(np.abs((test_mape - predicted_lstm_scaled_out[:, 0:1]) / test_mape)) * 100
        #
        mape_plot(lstm_mape=mape_lstm, gru_mape=mape_gru, plot_titles=path)