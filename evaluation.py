from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def evaluate_model(y_true, y_pred):
    """
    Avalia o desempenho de um modelo com base em duas métricas:
    1. RMSE (Root Mean Squared Error)
    2. MAE (Mean Absolute Error)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

def time_series_cross_validation(data, model_func, n_splits=5):
    """
    Realiza validação cruzada para séries temporais, lidando com modelos diferentes como ARIMA e Prophet.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = []

    for train_idx, test_idx in tscv.split(data):
        train, test = data.iloc[train_idx], data.iloc[test_idx]

        # Ajuste para Prophet
        if model_func.__name__ == "train_prophet":
            # Salva os valores reais antes de renomear as colunas
            y_true = test['Global_active_power'].values

            # Ajusta o formato dos dados para Prophet
            train = train.reset_index().rename(columns={'Datetime': 'ds', 'Global_active_power': 'y'})
            test = test.reset_index().rename(columns={'Datetime': 'ds', 'Global_active_power': 'y'})
            model = model_func(train)
            predictions = model.predict(test)['yhat'].values
        else:
            # Para ARIMA e outros modelos que suportam .forecast()
            model = model_func(train)
            predictions = model.forecast(len(test))
            y_true = test['Global_active_power'].values

        # Calcula as métricas para o conjunto de teste
        rmse, mae = evaluate_model(y_true, predictions)
        errors.append((rmse, mae))

    return errors
