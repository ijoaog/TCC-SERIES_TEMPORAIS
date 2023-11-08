import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

yf.pdr_override()

start_date = "2022-11-08"
end_date = "2023-11-08"
ticker = 'AAPL'

# Coleta de dados
data = wb.get_data_yahoo(ticker, start=start_date, end=end_date)
print(data)
scaled_data = data['Close'].values.reshape(-1, 1)

# Cálculo do RSI (índice de período de 14 dias) ======================= RSI -  MOMENTUM
def calculate_rsi(data, period=14):
    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data['Close'], period=14)

# Cálculo da Média Móvel Simples (SMA) de 50 dias ============== BASEADO EM TENDENCIAS
def calculate_sma(data, window=50):
    return data['Close'].rolling(window=window).mean()

data['SMA_50'] = calculate_sma(data, window=50)

# Cálculo do Money Flow Index (MFI) de 14 dias - BASEADO EM VOLUME
def calculate_mfi(data, period=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    raw_money_flow = typical_price * data['Volume']
    money_flow = np.where(data['Close'] > data['Close'].shift(1), raw_money_flow, 0)
    money_flow_negative = np.where(data['Close'] < data['Close'].shift(1), raw_money_flow, 0)
    money_ratio = np.sum(money_flow, axis=0) / np.sum(money_flow_negative, axis=0)
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

data['MFI'] = calculate_mfi(data, period=14)

# Cálculo do Estocástico (Estochastic Oscillator) - BASEADO EM OSCILADORES
def calculate_stochastic(data, period=14):
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    stoch_k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=3).mean()
    return stoch_k, stoch_d

data['Stoch_K'], data['Stoch_D'] = calculate_stochastic(data, period=14)

# Divisão dos dados em treinamento (80%) e previsão (20%)
split_index = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:split_index], scaled_data[split_index:]
train_rsi, test_rsi = data['RSI'][:split_index], data['RSI'][split_index:]
train_sma, test_sma = data['SMA_50'][:split_index], data['SMA_50'][split_index:]
train_mfi, test_mfi = data['MFI'][:split_index], data['MFI'][split_index:]
train_stoch_k, train_stoch_d = data['Stoch_K'][:split_index], data['Stoch_D'][:split_index]
test_stoch_k, test_stoch_d = data['Stoch_K'][split_index:], data['Stoch_D'][split_index:]

# Preparação dos dados de treinamento e previsão com indicadores (RSI, SMA, MFI e Estocástico)
x_train, y_train, rsi_train, sma_train, mfi_train, stoch_k_train, stoch_d_train = [], [], [], [], [], [], []
x_test, y_test, rsi_test, sma_test, mfi_test, stoch_k_test, stoch_d_test = [], [], [], [], [], [], []

for i in range(1, len(train_data)):
    x_train.append([train_data[i - 1][0], train_rsi.iloc[i - 1], train_sma.iloc[i - 1], train_mfi.iloc[i - 1], train_stoch_k.iloc[i - 1], train_stoch_d.iloc[i - 1]])
    y_train.append(train_data[i][0])
    rsi_train.append(train_rsi.iloc[i - 1])
    sma_train.append(train_sma.iloc[i - 1])
    mfi_train.append(train_mfi.iloc[i - 1])
    stoch_k_train.append(train_stoch_k.iloc[i - 1])
    stoch_d_train.append(train_stoch_d.iloc[i - 1])

for i in range(1, len(test_data)):
    x_test.append([test_data[i - 1][0], test_rsi.iloc[i - 1], test_sma.iloc[i - 1], test_mfi.iloc[i - 1], test_stoch_k.iloc[i - 1], test_stoch_d.iloc[i - 1]])
    y_test.append(test_data[i][0])
    rsi_test.append(test_rsi.iloc[i - 1])
    sma_test.append(test_sma.iloc[i - 1])
    mfi_test.append(test_mfi.iloc[i - 1])
    stoch_k_test.append(test_stoch_k.iloc[i - 1])
    stoch_d_test.append(test_stoch_d.iloc[i - 1])

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
mse_prediction_with_indicators = mean_squared_error(y_test, predicted_prediction_with_indicators)
rmse_prediction_with_indicators = np.sqrt(mse_prediction_with_indicators)
mae_prediction_with_indicators = mean_absolute_error(y_test, predicted_prediction_with_indicators)

# Preparação dos dados de treinamento e previsão sem indicadores
x_train_no_indicators = train_data[:-1]
x_test_no_indicators = test_data[:-1]

# Crie e treine o modelo de regressão linear com os dados sem indicadores
model_no_indicators = LinearRegression()
model_no_indicators.fit(x_train_no_indicators, y_train)

# Faça previsões para os dados de previsão sem indicadores
predicted_prediction_no_indicators = model_no_indicators.predict(x_test_no_indicators)

# Calcule os erros para as previsões sem indicadores
mse_prediction_no_indicators = mean_squared_error(y_test, predicted_prediction_no_indicators)
rmse_prediction_no_indicators = np.sqrt(mse_prediction_no_indicators)
mae_prediction_no_indicators = mean_absolute_error(y_test, predicted_prediction_no_indicators)

# Plot the results for the prediction data
plt.figure(figsize=(12, 6))
plt.title(f"Preço das Ações da Apple - Comparação de Previsão com e sem Indicadores (RSI, SMA, MFI e Estocástico)")
plt.xlabel("Tempo")
plt.ylabel(f"Preço das Ações da Apple")

# Linha de preço real
plt.plot(data.index[split_index+1:], y_test, label="Real (Previsão)", color="green")

# Linha de previsão com indicadores de momentum (RSI, SMA, MFI e Estocástico)
plt.plot(data.index[split_index+1:], predicted_prediction_with_indicators, label="Previsão (RSI, SMA, MFI e Estocástico)", color="blue")

# Linha de previsão sem indicadores
plt.plot(data.index[split_index+1:], predicted_prediction_no_indicators, label="Previsão (Sem Indicadores)", color="orange")

plt.legend(["Real", "RSI, SMA, MFI e Estocástico", "Sem Indicadores"])
plt.grid(True)

plt.show()

print("Métricas de Erro (Com e Sem Indicadores - RSI, SMA, MFI e Estocástico):")
print("Com Indicadores:")
print(f"Erro Médio Quadrático (MSE): {mse_prediction_with_indicators:.2f}")
print(f"Erro Quadrático Médio Raiz (RMSE): {rmse_prediction_with_indicators:.2f}")
print(f"Erro Médio Absoluto (MAE): {mae_prediction_with_indicators:.2f}")

print("\nSem Indicadores:")
print(f"Erro Médio Quadrático (MSE): {mse_prediction_no_indicators:.2f}")
print(f"Erro Quadrático Médio Raiz (RMSE): {rmse_prediction_no_indicators:.2f}")
print(f"Erro Médio Absoluto (MAE): {mae_prediction_no_indicators:.2f}")

#==========================PREVISAO 10 DIAS A FRENTE

# Obtenha a data do último dia nos dados de teste
last_test_date = datetime.datetime.now()

# Crie uma lista de datas para os próximos 10 dias a partir da data do último dia nos dados de teste
forecasted_dates = [last_test_date + datetime.timedelta(days=i) for i in range(1, 11)]

# Inverta a ordem da lista de datas
forecasted_dates = forecasted_dates[::-1]

# Faça previsões para os próximos 10 dias com base nos novos dados
forecasted_prices = model.predict(x_test_imputed)

# Imprima as datas e valores de preço previsto para os próximos 10 dias (com indicadores)
print("Preços previstos para os próximos 10 dias (com indicadores):")
for date, price in zip(forecasted_dates, forecasted_prices[-10:]):
    print(f"Data: {date.date()}, Preço previsto: {price:.2f}")

# Faça previsões para os próximos 10 dias com base nos dados de preço de fechamento sem indicadores
forecasted_prices_no_indicators = model_no_indicators.predict(x_test_no_indicators)

# Imprima as datas e valores de preço previsto para os próximos 10 dias (sem indicadores)
print("Preços previstos para os próximos 10 dias (sem indicadores):")
for date, price in zip(forecasted_dates, forecasted_prices_no_indicators[-10:]):
    print(f"Data: {date.date()}, Preço previsto: {price:.2f}")

