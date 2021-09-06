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


