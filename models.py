from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Modelo ARIMA
def train_arima(data):
    """
    Função para treinar um modelo ARIMA com os dados fornecidos.
    Os parâmetros do modelo podem ser ajustados conforme necessário
    para melhorar o desempenho.
    """
    model = ARIMA(data, order=(5, 1, 0))  # Parâmetros iniciais para ARIMA
    model_fit = model.fit()
    return model_fit

# Modelo Prophet
def train_prophet(data):
    """
    Treina um modelo Prophet para prever séries temporais.
    Os dados são convertidos para o formato esperado pelo Prophet.
    """
    # Ajusta o formato dos dados para o Prophet (colunas 'ds' e 'y')
    df = data.reset_index().rename(columns={'Datetime': 'ds', 'Global_active_power': 'y'})
    model = Prophet()
    model.fit(df)
    return model

# Funções para LSTM
def prepare_lstm_data(data, lookback=10):
    """
    Prepara os dados para uso no modelo LSTM.
    Cria uma janela deslizante de tamanho 'lookback' para capturar
    as relações temporais nos dados.
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])  # Janela deslizante como entrada
        y.append(data[i+lookback])   # Valor seguinte como saída
    return np.array(X), np.array(y)

def train_lstm(train_data, lookback=10):
    """
    Treina um modelo LSTM usando os dados de treino.
    O modelo é configurado com duas camadas LSTM e uma camada densa.
    """
    # Prepara os dados de treino
    X_train, y_train = prepare_lstm_data(train_data, lookback)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Ajusta para formato 3D

    # Define a arquitetura do modelo LSTM
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(50, return_sequences=False),
        Dense(1)  # Saída com um único valor
    ])
    model.compile(optimizer='adam', loss='mse')  # Configura otimizador e função de perda
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)  # Treina o modelo
    return model
