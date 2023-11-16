def resultado_analise(var,com_indicador):
    import numpy as np
    import pandas as pd
    import datetime
    import matplotlib.pyplot as plt
    from pandas_datareader import data as wb
    import yfinance as yf
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.impute import SimpleImputer
    from statsmodels.tsa.arima.model import ARIMA
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    from sklearn.preprocessing import MinMaxScaler

    yf.pdr_override()

    start_date = "2022-11-15"
    end_date = "2023-11-15"
    ticker = var


    # Coleta de dados
    data = wb.get_data_yahoo(ticker, start=start_date, end=end_date)

    scaled_data = data['Close'].values.reshape(-1, 1)

    # Divisão dos dados em treinamento (80%) e previsão (20%)
    split_index = int(len(scaled_data) * 0.8)
    test_size = int(len(scaled_data) * 0.2)

    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index:]
    forecast_data = scaled_data[split_index:]

    # Cálculo dos indicadores
    def calculate_indicators(data):
        rsi_period = 25
        sma_window = 50
        mfi_period = 25
        stoch_period = 14

        data['RSI'] = calculate_rsi(data['Close'], period=rsi_period)
        data['SMA_50'] = calculate_sma(data, window=sma_window)
        data['MFI'] = calculate_mfi(data, period=mfi_period)
        stoch_k, stoch_d = calculate_stochastic(data, period=stoch_period)
        data['Stoch_K'] = stoch_k
        data['Stoch_D'] = stoch_d

        return data

    def calculate_rsi(data, period=25):
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = gain[:period].mean()
        avg_loss = loss[:period].mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_sma(data, window=25):
        return data['Close'].rolling(window=window).mean()

    def calculate_mfi(data, period=25):
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        raw_money_flow = typical_price * data['Volume']
        money_flow = np.where(data['Close'] > data['Close'].shift(1), raw_money_flow, 0)
        money_flow_negative = np.where(data['Close'] < data['Close'].shift(1), raw_money_flow, 0)
        money_ratio = np.sum(money_flow, axis=0) / np.sum(money_flow_negative, axis=0)
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

    def calculate_stochastic(data, period=14):
        low_min = data['Low'].rolling(window=period).min()
        high_max = data['High'].rolling(window=period).max()
        stoch_k = 100 * (data['Close'] - low_min) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=3).mean()
        return stoch_k, stoch_d

    data = calculate_indicators(data)

    train_rsi = data['RSI'][:split_index]
    test_rsi = data['RSI'][split_index:]

    train_sma = data['SMA_50'][:split_index]
    test_sma = data['SMA_50'][split_index:]

    train_mfi = data['MFI'][:split_index]
    test_mfi = data['MFI'][split_index:]

    train_stoch_k = data['Stoch_K'][:split_index]
    train_stoch_d = data['Stoch_D'][:split_index]
    test_stoch_k = data['Stoch_K'][split_index:]
    test_stoch_d = data['Stoch_D'][split_index:]

    # Preparação dos dados de treinamento e previsão com indicadores
    x_train = np.column_stack((train_data[:-1], train_rsi[:-1], train_sma[:-1], train_mfi[:-1], train_stoch_k[:-1], train_stoch_d[:-1]))
    y_train = train_data[1:]

    x_test = np.column_stack((test_data[:-1], test_rsi[:-1], test_sma[:-1], test_mfi[:-1], test_stoch_k[:-1], test_stoch_d[:-1]))
    y_test = test_data[1:]

    # Impute os valores ausentes nos dados de treinamento e teste
    imputer = SimpleImputer(strategy='mean')
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)

    # Crie e treine o modelo de regressão linear com os dados imputados (com indicadores)
    model = LinearRegression()
    model.fit(x_train_imputed, y_train)

    # Faça previsões para os dados de previsão (com indicadores)
    predicted_prediction_with_indicators = model.predict(x_test_imputed)

    # Calcule os erros para as previsões (com indicadores)
    mse_prediction_with_indicators_test = mean_squared_error(y_test, predicted_prediction_with_indicators)
    rmse_prediction_with_indicators_test = np.sqrt(mse_prediction_with_indicators_test)
    mae_prediction_with_indicators_test = mean_absolute_error(y_test, predicted_prediction_with_indicators)
    # Calcule os erros para as previsões (com indicadores)
    errors = np.abs(y_test - predicted_prediction_with_indicators)
    mape_prediction_with_indicators_test = np.mean(errors / y_test) * 100



    # Preparação dos dados de treinamento e previsão sem indicadores
    x_train_no_indicators = train_data[:-1]
    x_test_no_indicators = test_data[:-1]

    # Crie e treine o modelo de regressão linear com os dados sem indicadores
    model_no_indicators = LinearRegression()
    model_no_indicators.fit(x_train_no_indicators, y_train)

    # Faça previsões para os dados de previsão sem indicadores
    predicted_prediction_no_indicators = model_no_indicators.predict(x_test_no_indicators)

    mape_prediction_no_indicators_test = np.mean(np.abs((y_test - predicted_prediction_no_indicators) / y_test)) * 100
    # Calcule os erros para as previsões sem indicadores
    mse_prediction_no_indicators_test = mean_squared_error(y_test, predicted_prediction_no_indicators)
    rmse_prediction_no_indicators_test = np.sqrt(mse_prediction_no_indicators_test)
    mae_prediction_no_indicators_test = mean_absolute_error(y_test, predicted_prediction_no_indicators)

    print("Métricas de Erro (Com e Sem Indicadores - RSI, SMA, MFI e Estocástico):")
    print("Com Indicadores:")
    print(f"Erro Médio Quadrático (MSE): {mse_prediction_with_indicators_test:.2f}")
    print(f"Erro Quadrático Médio Raiz (RMSE): {rmse_prediction_with_indicators_test:.2f}")
    print(f"Erro Médio Absoluto (MAE): {mae_prediction_with_indicators_test:.2f}")
    print(f"Erro Percentual Médio Absoluto (MAPE): {mape_prediction_with_indicators_test:.2f}%")

    print("\nSem Indicadores:")
    print(f"Erro Médio Quadrático (MSE): {mse_prediction_no_indicators_test:.2f}")
    print(f"Erro Quadrático Médio Raiz (RMSE): {rmse_prediction_no_indicators_test:.2f}")
    print(f"Erro Médio Absoluto (MAE): {mae_prediction_no_indicators_test:.2f}")
    print(f"Erro Percentual Médio Absoluto (MAPE): {mape_prediction_no_indicators_test:.2f}%")


    # Plot the results for the prediction data
    plt.figure(figsize=(6, 6))
    plt.title("Preço das Ações da Apple - Comparação de Previsão com e sem Indicadores (RSI, SMA, MFI e Estocástico)")
    plt.xlabel("Tempo")
    plt.ylabel("Preço das Ações da Apple")

    # Linha de preço real
    plt.plot(data.index[split_index+1:], y_test, label="Real (Previsão)", color="green")

    # Linha de previsão com indicadores de momentum (RSI, SMA, MFI e Estocástico)
    plt.plot(data.index[split_index+1:], predicted_prediction_with_indicators, label="Previsão (RSI, SMA, MFI e Estocástico)", color="blue")

    # Linha de previsão sem indicadores
    plt.plot(data.index[split_index+1:], predicted_prediction_no_indicators, label="Previsão (Sem Indicadores)", color="orange")

    plt.legend()
    plt.grid(True)

    # plt.show()
    # Obtenha a data do último dia nos dados de teste
    last_test_date = datetime.datetime.now()

    # Crie uma lista de datas para os próximos 10 dias a partir da data do último dia nos dados de teste
    forecasted_dates = [last_test_date + datetime.timedelta(days=i) for i in range(1, 11)]

    # Inverta a ordem da lista de datas
    forecasted_dates = forecasted_dates[::-1]

    # Faça previsões para os próximos 10 dias com base nos novos dados
    forecasted_prices = model.predict(x_test_imputed)
    forecasted_prices_list = forecasted_prices[-5:].flatten().tolist()
    # Faça previsões para os próximos 10 dias com base nos dados de preço de fechamento sem indicadores
    forecasted_prices_no_indicators = model_no_indicators.predict(x_test_no_indicators)
    forecasted_prices_list_no_indicators = forecasted_prices_no_indicators[-5:].flatten().tolist()
    #================================================================ARIMA

    # Ajuste o modelo ARIMA com indicadores
    model_arima = ARIMA(train_data, order=(5, 1, 0))
    model_arima_fit = model_arima.fit()

    # Faça previsões usando o modelo ARIMA nos dados de teste com indicadores
    predicted_prediction_arima_with_indicators = model_arima_fit.forecast(steps=len(test_data))

    # Calcule os erros para as previsões do ARIMA com indicadores
    mse_prediction_arima_with_indicators_test = mean_squared_error(test_data, predicted_prediction_arima_with_indicators)
    rmse_prediction_arima_with_indicators_test = np.sqrt(mse_prediction_arima_with_indicators_test)
    mae_prediction_arima_with_indicators_test = mean_absolute_error(test_data, predicted_prediction_arima_with_indicators)
    mape_prediction_arima_with_indicators_test = np.mean(np.abs((test_data - predicted_prediction_arima_with_indicators) / test_data)) * 100

    # Exibir os resultados dos erros para o ARIMA com indicadores
    print("\nARIMA com Indicadores:")
    print(f"Erro Médio Quadrático (MSE): {mse_prediction_arima_with_indicators_test:.2f}")
    print(f"Erro Quadrático Médio Raiz (RMSE): {rmse_prediction_arima_with_indicators_test:.2f}")
    print(f"Erro Médio Absoluto (MAE): {mae_prediction_arima_with_indicators_test:.2f}")
    print(f"Erro Percentual Médio Absoluto (MAPE): {mape_prediction_arima_with_indicators_test:.2f}%")


    # Ajuste o modelo ARIMA sem indicadores
    model_arima_no_indicators = ARIMA(test_data, order=(5, 1, 0))
    model_arima_no_indicators_fit = model_arima_no_indicators.fit()
    # Faça previsões usando o modelo ARIMA nos dados de teste sem indicadores
    predicted_prediction_arima_no_indicators = model_arima_no_indicators_fit.forecast(steps=len(test_data))

    # Calcule os erros para as previsões do ARIMA sem indicadores
    mse_prediction_arima_no_indicators_test = mean_squared_error(forecast_data, predicted_prediction_arima_no_indicators)
    rmse_prediction_arima_no_indicators_test = np.sqrt(mse_prediction_arima_no_indicators_test)
    mae_prediction_arima_no_indicators_test = mean_absolute_error(forecast_data, predicted_prediction_arima_no_indicators)
    mape_prediction_arima_no_indicators_test = np.mean(np.abs((forecast_data - predicted_prediction_arima_no_indicators) / forecast_data)) * 100

    # Exibir os resultados dos erros para o ARIMA sem indicadores
    print("\nARIMA sem Indicadores:")
    print(f"Erro Médio Quadrático (MSE): {mse_prediction_arima_no_indicators_test:.2f}")
    print(f"Erro Quadrático Médio Raiz (RMSE): {rmse_prediction_arima_no_indicators_test:.2f}")
    print(f"Erro Médio Absoluto (MAE): {mae_prediction_arima_no_indicators_test:.2f}")
    print(f"Erro Percentual Médio Absoluto (MAPE): {mape_prediction_arima_no_indicators_test:.2f}%")

    # Plot the results for the ARIMA prediction data
    plt.figure(figsize=(6, 6))
    plt.title(f"Preço das Ações da Apple - Comparação de Previsão com e sem Indicadores (ARIMA)")
    plt.xlabel("Tempo")
    plt.ylabel(f"Preço das Ações da Apple")

    # Linha de preço real
    plt.plot(data.index[split_index:], forecast_data.flatten(), label="Real (Previsão)", color="green")

    # Linha de previsão com indicadores de ARIMA
    plt.plot(data.index[split_index:split_index+len(predicted_prediction_arima_with_indicators)], predicted_prediction_arima_with_indicators.flatten(), label="Previsão (ARIMA com indicador)", color="blue")

    # Linha de previsão sem indicadores de ARIMA
    plt.plot(data.index[split_index:split_index+len(predicted_prediction_arima_no_indicators)], predicted_prediction_arima_no_indicators.flatten(), label="Previsão (ARIMA sem indicador)", color="red")

    plt.legend(["Real", "ARIMA Com indicador" , "ARIMA sem indicador"])
    plt.grid(True)

    # plt.show()

    #================================================================LSTM

    # Prepare Data

    scaler = MinMaxScaler(feature_range=(0,1)) # Transforma os dados para valores entre 0,1
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1)) # Transforma os dados de fechamento das ações

    prediction_days = 60

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Determinar o índice de divisão (80% para treinamento, 20% para teste)
    split_index = int(len(x_train) * 0.8)

    # Dividir os dados em conjuntos de treinamento e teste
    x_train = x_train[:split_index]
    y_train = y_train[:split_index]

    x_test = x_train[split_index:]
    y_test = y_train[split_index:]

    # Build the Model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Prediction of the next closing price

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50, batch_size=32)


    test_data = wb.get_data_yahoo(ticker,start=start_date, end=end_date)

    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)

    # Make Predictions on Test Data

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot the Test Predictions for the last 60 days
    plt.figure(figsize=(6, 6))
    last_60_dates = test_data.index[-60:]
    last_60_actual_prices = actual_prices[-60:]
    last_60_predicted_prices = predicted_prices[-60:]

    # Traduza os rótulos para o português
    plt.title(f"Preço das Ações da Apple - LSTM (Últimos 60 Dias) em USD")
    plt.xlabel("Tempo")
    plt.ylabel(f"Preço das Ações Apple em USD")

    # Plot dos preços em USD
    plt.plot(last_60_dates, last_60_actual_prices, color="black", label=f"Preço Real em USD")
    plt.plot(last_60_dates, last_60_predicted_prices, color="red", label=f"Preço Previsto em USD")

    # Adicione um rótulo ao eixo y em USD
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

    # Traduza a legenda
    plt.legend(["Preço Real em USD", "Preço Previsto em USD"])
    plt.grid(True)

    # Mostra o gráfico
    # plt.show()

    # Calcular o Erro Quadrático Médio Raiz (RMSE)
    rmse = np.sqrt(np.mean((actual_prices - predicted_prices)**2))

    # Calcular o Erro Percentual Médio Absoluto (MAPE)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

    # Calcular o Erro Quadrático Médio (MSE)
    mse = np.mean((actual_prices - predicted_prices)**2)

    # Calcular o Erro Médio Absoluto (MAE)
    mae = np.mean(np.abs(actual_prices - predicted_prices))

    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape}%')
    print("oxi",mape_prediction_with_indicators_test)
    print("oxi2",mape_prediction_no_indicators_test)
    forecasted_prices_list = forecasted_prices_list[::-1]
    forecasted_prices_list_no_indicators = forecasted_prices_list_no_indicators[::-1]
    if com_indicador:
        resultados_finais = {
            'MSE': round(mse_prediction_with_indicators_test,2),
            'MAE': round(mae_prediction_with_indicators_test,2),
            'RMSE': round(rmse_prediction_with_indicators_test,2),
            'MAPE': round(mape_prediction_with_indicators_test,2),
            'lista_predicao': forecasted_prices_list,
        }
    else:
        resultados_finais = {
            'MSE': round(mse_prediction_no_indicators_test,2),
            'MAE': round(mae_prediction_no_indicators_test,2),
            'RMSE': round(rmse_prediction_no_indicators_test,2),
            'MAPE': round(mape_prediction_no_indicators_test,2),
            'lista_predicao': forecasted_prices_list_no_indicators,
        }
    return resultados_finais