## Dados

Os ficheiros CSV em `data/raw/` não estão versionados devido ao seu tamanho.

Para executar o projeto, colocar os seguintes ficheiros em `data/raw/`:
- Producao.csv
- Marcações.csv
- Slotsdisponiveis.csv


## Projeçoes Mensais

    1. OBJETIVO

Este módulo implementa a previsão mensal do número de inspeções por centro de inspeção, com base em séries temporais históricas. As projeções são realizadas ao nível individual de cada centro, permitindo analisar padrões sazonais e suportar planeamento operacional.

    2. DADOS DE INPUT resultam da agregação mensal dos registos de produção (Producao.csv), contendo as colunas:

CFGCENTROID – identificador do centro
MONTH – mês de referência
y – número de inspeções realizadas no mês

Os dados são armazenados em formato Parquet para eficiência de processamento.

    3. METODOLOGIA

A modelação baseia-se no modelo Holt-Winters aditivo, adequado a séries temporais com sazonalidade anual. O modelo é ajustado individualmente a cada centro, captando padrões de nível, tendência e sazonalidade mensal.

Foram também testados modelos baseline e SARIMA, sendo o Holt-Winters selecionado com base no desempenho em validação temporal.

    4. VALIDAÇÃO

A qualidade das previsões foi avaliada através de backtesting one-step-ahead, treinando o modelo até ao penúltimo mês e prevendo o último mês observado.
O erro médio absoluto obtido foi aproximadamente 2–3% do volume mensal médio por centro, indicando elevada capacidade preditiva no curto prazo.

    5. OUTPUTS

O módulo gera:
    previsões mensais até 12 meses para todos os centros (forecast_monthly_all_centers.parquet)
    gráficos de histórico vs previsão para centros individuais
    métricas de erro (MAE e RMSE) no backtesting

    6. EXECUÇÃO

    Previsões mensais: python src/models_monthly/backtest_all_centers.py
    Backtesting: python src/models_monthly/backtest_all_centers.py

