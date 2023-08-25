import pandas as pd
import csv

ipca = pd.read_csv('../data/ipca.csv')
ipca['Data'] = pd.to_datetime(ipca['Data'])

file = '../data/2013_2022.csv'
dtype = {
    'Matricula': str,
    'CNPJ': str
}
df = pd.read_csv(file, dtype=dtype)
df['Data'] = pd.to_datetime(df["Ano"].astype(str) 
                            + (df["Mes"].astype(str)).str.zfill(2)
                            + '01')
df = df[df['Tipo'] == 'I - HOSPEDAGEM, ALIMENTAÇÃO E DESPESAS DE LOCOMOÇÃO']
df = df[['Data', 'CNPJ', 'Valor']]
df = df[df['Data'].dt.year > 2017 ]

dados = pd.merge(left=df, right=ipca, on='Data', how="inner")

dados['Valor_ref'] = ipca[ipca['Data'] == '2022-12-01']['Valor'].values[0]
dados['Valor_corrigido'] = round((dados['Valor_ref'] / dados['Valor_y']) * dados['Valor_x'], 2)
dados = dados[['Data', 'CNPJ', 'Valor_corrigido']]
dados.to_csv('../data/2018_2022_corrigido.csv', encoding='utf-8', index=False, sep=',', quoting=csv.QUOTE_NONNUMERIC)


