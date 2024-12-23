import pandas as pd
import os

def load_data(file_path=None):
    """
    Carrega e processa os dados de consumo elétrico para análise.
    """
    # Define o caminho padrão, caso não seja fornecido
    if file_path is None:
        file_path = os.path.join("data", "household_power_consumption.txt")
    
    # Lê o arquivo com separador específico e trata valores ausentes
    data = pd.read_csv(file_path, sep=';', na_values=['?'])
    
    # Combina as colunas 'Date' e 'Time' para criar o índice de datetime
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], dayfirst=True)
    data.drop(columns=['Date', 'Time'], inplace=True)
    data.set_index('Datetime', inplace=True)
    
    # Renomeia as colunas para facilitar o entendimento
    data.columns = [
        'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity', 'Sub_metering_1', 
        'Sub_metering_2', 'Sub_metering_3'
    ]
    
    # Remove quaisquer linhas com valores ausentes
    data.dropna(inplace=True)
    
    # Converte colunas numéricas para o tipo correto
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 
        'Voltage', 'Global_intensity', 'Sub_metering_1', 
        'Sub_metering_2', 'Sub_metering_3'
    ]
    data[numeric_cols] = data[numeric_cols].astype(float)
    
    return data
