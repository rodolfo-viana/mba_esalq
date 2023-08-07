import sys
sys.path.insert(0, "..")
from src.kmeans import KMeans
from collections import defaultdict
from itertools import groupby
import numpy as np
import pandas as pd

df = pd.read_csv("../data/sample.csv")

results = defaultdict()
kmeans_class_obj = KMeans()

sorted_data = sorted(zip(df['company_name'], df['price']), key=lambda x: x[0])
for company, group in groupby(sorted_data, key=KMeans.keyfunc):
    prices = np.array([item[1] for item in group])
    if len(prices) > 2:
        kmeans_class_obj.fit(prices)
        anomalies_kmeans = kmeans_class_obj.detect_anomalies(prices)
        results[company] = anomalies_kmeans.tolist()

display_results = dict(results)
for k, v in display_results.items():
    print(f"{k}: {v}")