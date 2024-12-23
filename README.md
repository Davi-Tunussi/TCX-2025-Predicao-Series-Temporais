### Projeto 3 - TCX 2025: Sistema Preditivo de Séries Temporais (Machine Learning)

### Introdução

Este projeto tem como objetivo desenvolver um sistema preditivo que estima o consumo de energia elétrica com base em dados históricos. O sistema foi projetado para explorar diferentes modelos de séries temporais, incluindo ARIMA, Prophet e LSTM, e avaliar o desempenho de cada um em termos de previsão para os próximos 30 dias.

Os dados utilizados foram obtidos do UCI Machine Learning Repository - Energy Consumption Dataset. Eles abrangem o período de 16/12/2006 a 26/11/2010, com medições realizadas em intervalos de 1 minuto.

### Estrutura do Projeto

O projeto está organizado nos seguintes arquivos e diretórios:

Projeto_3_TCX_MachineLearning/ |-- data/ | |-- household_power_consumption.txt # Arquivo de dados |-- results/ | |-- previsoes.csv # Resultados das previsões | |-- comparacao_previsoes.png # Gráfico comparativo |-- data_loader.py # Carregamento e pré-processamento dos dados |-- models.py # Implementação dos modelos ARIMA, Prophet e LSTM |-- evaluation.py # Avaliação de métricas e validação cruzada |-- main.py # Execução principal do projeto |-- requirements.txt # Dependências do projeto |-- README.md # Documentação do projeto

### Modelos Utilizados

1. ARIMA
O modelo ARIMA é eficiente para séries temporais univariadas com padrões sazonais e não sazonais. Ele foi configurado com os seguintes parâmetros:
Ordem: (5, 1, 0)

2. Prophet
O Prophet, desenvolvido pelo Facebook, é adequado para séries temporais com padrões sazonais e feriados. Ele foi treinado com os dados convertidos para o formato esperado (ds e y).

3. LSTM
O LSTM é uma rede neural recorrente (RNN) projetada para aprender dependências de longo prazo. Ele foi configurado com:
Janela deslizante: 10 dias
- Neurônios: 50 em cada camada LSTM
- Épocas: 20
- Função de perda: Mean Squared Error (MSE)

### Resultados e Avaliação

Métricas de Desempenho

Os modelos foram avaliados usando RMSE (Root Mean Squared Error) e MAE (Mean Absolute Error). Os resultados para o conjunto de teste foram:

MODELO    |  RMSE   |   MAE
ARIMA     |  0.30   |   0.25
PROPHET   |  0.28   |   0.20
LSTM      |  0.07   |   0.05

O LSTM apresentou o melhor desempenho, com RMSE e MAE significativamente menores, destacando sua capacidade de capturar padrões complexos nos dados.

### Validação Cruzada

Foi realizada uma validação cruzada com 5 divisões (folds) para os modelos ARIMA e Prophet. Os resultados médios foram:
MODELO    |  RMSE (médio)   |   MAE (médio) 
ARIMA     |      0.51       |     0.43
PROPHET   |      0.57       |     0.48

### Resultados Numéricos e G´rafico Comparativo

Os resultados detalhados das previsões estão salvos na pasta results/

### Como Executar o Projeto

1. Requisitos
Certifique-se de que possui as dependências listadas no arquivo requirements.txt. Instale-as com o comando:
pip install -r requirements.txt

2. Execução
Execute o arquivo principal para treinar os modelos e gerar previsões:
python main.py

3. Resultados

Os resultados serão salvos no diretório results/:
previsoes.csv: Contém os valores reais e as previsões dos modelos.
comparacao_previsoes.png: Gráfico comparativo das previsões.

### Conclusão

O projeto demonstrou que o modelo LSTM é o mais adequado para este conjunto de dados, superando ARIMA e Prophet em todas as métricas. O LSTM mostrou-se mais capaz de capturar padrões complexos e fornecer previsões precisas.

Este trabalho destaca a importância de selecionar o modelo certo para séries temporais, dependendo das características dos dados e do objetivo da previsão.

### Projeto desenvolvido por Davi Tunussi como parte do desafio TCX 2025.
