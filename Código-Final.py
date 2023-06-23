import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/afgstofel/TbFinal 01/test.csv')
df1 = pd.read_csv('C:/Users/afgstofel/TbFinal 01/train.csv')
df.head()
variaveis = df.columns
print(variaveis)
tipos = df.dtypes
print(tipos)
valores_faltantes = df.isnull().sum()
print(valores_faltantes)
df.fillna(df.mean(), inplace=True)
df['LotFrontage'].plot.box()
plt.show()
estatisticas_descritivas = df.describe()
print(estatisticas_descritivas)
Q1 = estatisticas_descritivas.loc['25%']
Q3 = estatisticas_descritivas.loc['75%']
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
outliers = df[(df < limite_inferior) | (df > limite_superior)].dropna(how='all')
print(outliers)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Carregar o conjunto de dados
data = {
    'Id': [1, 2, 3, 4, 5],
    'MSSubClass': [20, 20, 60, 60, 120],
    'LotFrontage': [80.0, 81.0, 74.0, 78.0, 43.0],
    'LotArea': [11622, 14267, 13830, 9978, 5005],
    'OverallQual': [5, 6, 7, 8, 9],
    'OverallCond': [5, 5, 6, 6, 6],
    'YearBuilt': [1978, 1996, 2001, 1998, 2010],
    'YearRemodAdd': [1998, 1997, 2002, 1998, 2010],
    'MasVnrArea': [0, 162, 0, 350, 0],
    'BsmtFinSF1': [0, 798, 0, 980, 0],
    'SaleType': ['WD', 'WD', 'New', 'New', 'WD'],
    'SaleCondition': ['Normal', 'Normal', 'Abnormal', 'Abnormal', 'Normal'],
    'SalePrice': [208500, 181500, 223500, 250000, 330000]
}

df = pd.DataFrame(data)

# Selecionar apenas as colunas numéricas e categóricas
numeric_columns = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                   'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1']
categorical_columns = ['SaleType', 'SaleCondition']

# Normalizar as colunas numéricas
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Codificar as variáveis categóricas
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(df[categorical_columns])
encoded_columns = encoder.get_feature_names_out(categorical_columns)
df_encoded = pd.DataFrame(encoded_features, columns=encoded_columns, index=df.index)
df = pd.concat([df, df_encoded], axis=1)
df.drop(columns=categorical_columns, inplace=True)

# Criar novas features
df['TotalArea'] = df['LotArea'] + df['MasVnrArea'] + df['BsmtFinSF1']
df['Age'] = 2023 - df['YearBuilt']

# Exibir o DataFrame após as transformações
print(df)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Carregar o conjunto de dados
data = {
    'Id': [1, 2, 3, 4, 5],
    'MSSubClass': [20, 20, 60, 60, 120],
    'LotFrontage': [80.0, 81.0, 74.0, 78.0, 43.0],
    'LotArea': [11622, 14267, 13830, 9978, 5005],
    'OverallQual': [5, 6, 7, 8, 9],
    'OverallCond': [5, 5, 6, 6, 6],
    'YearBuilt': [1978, 1996, 2001, 1998, 2010],
    'YearRemodAdd': [1998, 1997, 2002, 1998, 2010],
    'MasVnrArea': [0, 162, 0, 350, 0],
    'BsmtFinSF1': [0, 798, 0, 980, 0],
    'SaleType': ['WD', 'WD', 'New', 'New', 'WD'],
    'SaleCondition': ['Normal', 'Normal', 'Abnormal', 'Abnormal', 'Normal'],
    'SalePrice': [208500, 181500, 223500, 250000, 330000]
}

df = pd.DataFrame(data)

# Selecionar apenas as colunas numéricas e categóricas
numeric_columns = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                   'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1']
categorical_columns = ['SaleType', 'SaleCondition']

# Normalizar as colunas numéricas
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Codificar as variáveis categóricas
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(df[categorical_columns])
encoded_columns = encoder.get_feature_names_out(categorical_columns)
df_encoded = pd.DataFrame(encoded_features, columns=encoded_columns, index=df.index)
df = pd.concat([df, df_encoded], axis=1)
df.drop(columns=categorical_columns, inplace=True)

# Criar novas features
df['TotalArea'] = df['LotArea'] + df['MasVnrArea'] + df['BsmtFinSF1']
df['Age'] = 2023 - df['YearBuilt']

# Exibir o DataFrame após as transformações
print(df)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Carregar o conjunto de dados (após as transformações)
data = {
    'Id': [1, 2, 3, 4, 5],
    'MSSubClass': [20, 20, 60, 60, 120],
    'LotFrontage': [80.0, 81.0, 74.0, 78.0, 43.0],
    'LotArea': [11622, 14267, 13830, 9978, 5005],
    'OverallQual': [5, 6, 7, 8, 9],
    'OverallCond': [5, 5, 6, 6, 6],
    'YearBuilt': [1978, 1996, 2001, 1998, 2010],
    'YearRemodAdd': [1998, 1997, 2002, 1998, 2010],
    'MasVnrArea': [0, 162, 0, 350, 0],
    'BsmtFinSF1': [0, 798, 0, 980, 0],
    'SalePrice': [208500, 181500, 223500, 250000, 330000]
}

df = pd.DataFrame(data)

# Dividir o conjunto de dados em atributos de entrada (X) e variável de destino (y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Dividir o conjunto de dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados de entrada
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar e treinar o modelo KNN
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test_scaled)

# Calcular a métrica de erro (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)
# Carregar os conjuntos de dados
df_test = pd.read_csv('C:/Users/afgstofel/TbFinal 01/test.csv')
df_train = pd.read_csv('C:/Users/afgstofel/TbFinal 01/train.csv')

# Adicionar coluna 'SalePrice' com valores nulos no conjunto de test
df_test['SalePrice'] = None

# Concatenar os conjuntos de dados
df = pd.concat([df_train, df_test], ignore_index=True)

# Exibir o conjunto de dados
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Carregar o conjunto de dados de preços de casas
df = pd.read_csv('C:/Users/afgstofel/TbFinal 01/test.csv')
df1 = pd.read_csv('C:/Users/afgstofel/TbFinal 01/train.csv')

# Separar as variáveis independentes (features) e a variável dependente (target)
X = df.drop('preco', axis=1)
y = df['preco']

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo 1: Regressão Linear
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Modelo 2: Árvore de Decisão
model_dt = DecisionTreeRegressor()
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

# Métricas de avaliação - Regressão Linear
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Métricas de avaliação - Árvore de Decisão
mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Comparação dos modelos
print("Regressão Linear:")
print("MSE:", mse_lr)
print("MAE:", mae_lr)
print("R2:", r2_lr)
print("\nÁrvore de Decisão:")
print("MSE:", mse_dt)
print("MAE:", mae_dt)
print("R2:", r2_dt)