import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data
from models import train_arima, train_prophet, train_lstm, prepare_lstm_data
from evaluation import evaluate_model, time_series_cross_validation
import pandas as pd
import numpy as np

# ---------- Carregar os Dados ----------
# Carrega os dados de consumo de energia elétrica
data = load_data()

# Agregar os dados para frequência diária (média por dia)
daily_data = data.resample('D').mean()

# Define a frequência explícita no índice
daily_data = daily_data.asfreq('D')

# Remove quaisquer valores ausentes após a agregação
daily_data.dropna(inplace=True)

# ---------- Divisão dos Dados ----------
# Divide os dados em treino (histórico) e teste (últimos 30 dias)
train = daily_data.iloc[:-30]  # Dados de treino
test = daily_data.iloc[-30:]   # Dados de teste

# ---------- Pré-processamento e Escalonamento ----------
# Escala os dados para o intervalo [0, 1]
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[['Global_active_power']].values)
test_scaled = scaler.transform(test[['Global_active_power']].values)

# Verifica valores inválidos após o escalonamento
if np.any(np.isnan(train_scaled)) or np.any(np.isinf(train_scaled)):
    raise ValueError("Os dados de treino contêm valores inválidos (NaN ou Inf).")
if np.any(np.isnan(test_scaled)) or np.any(np.isinf(test_scaled)):
    raise ValueError("Os dados de teste contêm valores inválidos (NaN ou Inf).")

# Preparação dos dados para o modelo LSTM
lookback = 10  # Tamanho da janela deslizante
X_train, y_train = prepare_lstm_data(train_scaled.flatten(), lookback)
X_test, y_test = prepare_lstm_data(test_scaled.flatten(), lookback)

# Verifica valores inválidos nos dados preparados
if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
    raise ValueError("Os dados preparados para treino contêm valores inválidos.")
if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
    raise ValueError("As saídas de treino contêm valores inválidos.")

# ---------- Treinamento e Previsões ----------
# Treina o modelo ARIMA
arima_model = train_arima(train['Global_active_power'])
arima_pred = arima_model.forecast(steps=30)

# Treina o modelo Prophet
prophet_model = train_prophet(daily_data[['Global_active_power']])
prophet_forecast = prophet_model.predict(test.reset_index().rename(columns={'Datetime': 'ds'}))
prophet_pred = prophet_forecast['yhat'].values

# Treina o modelo LSTM
lstm_model = train_lstm(train_scaled.flatten(), lookback)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
lstm_pred = lstm_model.predict(X_test).flatten()

# Verifica valores inválidos na previsão
if np.any(np.isnan(lstm_pred)):
    raise ValueError("As previsões do LSTM contêm valores inválidos.")

# ---------- Avaliação dos Modelos ----------
# Avalia o desempenho de cada modelo usando RMSE e MAE
arima_rmse, arima_mae = evaluate_model(test['Global_active_power'], arima_pred)
prophet_rmse, prophet_mae = evaluate_model(test['Global_active_power'], prophet_pred)
lstm_rmse, lstm_mae = evaluate_model(y_test, lstm_pred)

# Exibe os resultados das métricas no console
print(f"ARIMA -> RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}")
print(f"Prophet -> RMSE: {prophet_rmse:.2f}, MAE: {prophet_mae:.2f}")
print(f"LSTM -> RMSE: {lstm_rmse:.2f}, MAE: {lstm_mae:.2f}")

# ---------- Gráficos Comparativos ----------
# Cria o diretório 'results/' caso não exista
os.makedirs("results", exist_ok=True)

# Gera um gráfico comparando os valores reais e previstos
plt.figure(figsize=(10, 6))
plt.plot(test.index[-len(lstm_pred):], y_test, label='Real')
plt.plot(test.index[-len(lstm_pred):], arima_pred[:len(lstm_pred)], label='ARIMA')
plt.plot(test.index[-len(lstm_pred):], prophet_pred[:len(lstm_pred)], label='Prophet')
plt.plot(test.index[-len(lstm_pred):], lstm_pred, label='LSTM')
plt.title('Comparação de Previsões')
plt.xlabel('Data')
plt.ylabel('Potência Ativa Global (kW)')
plt.legend()
plt.grid()
plt.savefig("results/comparacao_previsoes.png")
plt.close()

# ---------- Exportação de Resultados ----------
# Salva as previsões de cada modelo e os valores reais em um arquivo CSV
pd.DataFrame({
    "Real": y_test,
    "ARIMA": arima_pred[:len(lstm_pred)],
    "Prophet": prophet_pred[:len(lstm_pred)],
    "LSTM": lstm_pred
}).to_csv("results/previsoes.csv", index=False)

# ---------- Validação Cruzada ----------
# Executa a validação cruzada para ARIMA
arima_errors = time_series_cross_validation(daily_data[['Global_active_power']], train_arima)
print("ARIMA - Validação Cruzada (RMSE, MAE):", arima_errors)

# Executa a validação cruzada para Prophet
prophet_errors = time_series_cross_validation(daily_data[['Global_active_power']], train_prophet)
print("Prophet - Validação Cruzada (RMSE, MAE):", prophet_errors)
