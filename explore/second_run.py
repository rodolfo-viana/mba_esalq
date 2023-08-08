import sys
sys.path.insert(0, "..")
from src.kmeans import KMeans
from collections import defaultdict
from itertools import groupby
import numpy as np
import pandas as pd

df = pd.read_csv("../data/2013_2022_corrigido.csv")
df['CNPJ'] = df['CNPJ'].astype(str)
df = df[df['Data'].str.startswith('2022')]
df = df[['CNPJ', 'Valor_corrigido']]

results = defaultdict()
kmeans_class_obj = KMeans(feature='Valor_corrigido')

sorted_data = sorted(zip(df['CNPJ'], df['Valor_corrigido']), key=lambda x: x[0])
for company, group in groupby(sorted_data, key=KMeans.keyfunc):
    values = np.array([item[1] for item in group])
    if len(values) > 20:
        kmeans_class_obj.fit(values)
        anomalies_kmeans = kmeans_class_obj.detect(values)
        results[company] = anomalies_kmeans.tolist()

display_results = dict(results)
for k, v in display_results.items():
    print(f"{k}: {v}")
