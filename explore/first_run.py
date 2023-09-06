import sys
sys.path.insert(0, "..")
from src.kmeans import KMeans
from src.scores import Score
from collections import defaultdict
from itertools import groupby
import numpy as np
import pandas as pd

# Read dataset
df = pd.read_csv("../data/2018_2022_corrigido.csv")
# Remove null CNPJ
df = df[df['CNPJ'].notnull()]
# Convert CNPJ to str
df['CNPJ'] = df['CNPJ'].astype(str)
# Remove unnecessary columns
df = df[['CNPJ', 'Valor_corrigido']]

results = defaultdict()
kmeans_class_obj = KMeans()
sils = list()
dbs = list()

sorted_data = sorted(zip(df['CNPJ'], df['Valor_corrigido']), key=lambda x: x[0])
for company, group in groupby(sorted_data, key=lambda x: x[0]):
    values = np.array([item[1] for item in group])
    if len(values) > 20:
        kmeans_class_obj.fit(values.reshape(-1, 1))
        anomalies_kmeans = kmeans_class_obj.detect(values.reshape(-1, 1))
        labels = kmeans_class_obj.get_labels(values.reshape(-1, 1))
        sils.append(Score.silhouette(values.reshape(-1, 1), labels))
        dbs.append(Score.daviesbouldin(values.reshape(-1, 1), labels))

#display_results = dict(results)
#for k, v in display_results.items():
#    print(f"{k}: {v}")

print(sum(sils) / len(sils))
print(sum(dbs) / len(dbs))

