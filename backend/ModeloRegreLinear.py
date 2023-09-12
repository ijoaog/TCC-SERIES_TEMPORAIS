import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import jax.numpy as jnp  # Use jnp para o pacote JAX
from sklearn.metrics import mean_absolute_percentage_error

# Obter os dados do yfinance
data = yf.download('AAPL', start='2018-06-09', end='2023-06-09')

# Preparação dos dados
train_size = int(len(data) * 0.8)  # 80% para treinamento, 20% para teste
train_data = data['Close'][:train_size]
test_data = data['Close'][train_size:]

# Criar features e target
X_train = train_data.index.values.astype(int).reshape(-1, 1)  # Convertendo o índice em valores inteiros
y_train = train_data.values
X_test = test_data.index.values.astype(int).reshape(-1, 1)  # Convertendo o índice em valores inteiros
y_test = test_data.values

# Criação e treinamento do modelo de regressão
model = LinearRegression()
model.fit(X_train, y_train)

# Previsões com o modelo treinado
predictions = model.predict(X_test)

def mse(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return jnp.mean(jnp.abs(y_true - y_pred))

y_true = jnp.array(y_test)
y_pred = jnp.array(predictions)

mse_value = mse(y_true, y_pred)
mae_value = mae(y_true, y_pred)
rmse = jnp.sqrt(mse_value)
mape = mean_absolute_percentage_error(y_true, y_pred)

print('Modelo Regressão Linear')
print('Erro médio quadrático (MSE):', mse_value)
print('Erro médio absoluto (MAE):', mae_value)
print('Raiz quadrada do erro médio quadrático (RMSE):', rmse)
print('Erro Percentual Médio Absoluto (MAPE):', mape)
