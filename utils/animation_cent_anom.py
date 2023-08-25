import sys
from typing import Tuple
from collections import defaultdict
from itertools import groupby
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
sys.path.insert(0, "..")
from src.kmeans import KMeans


class AnimatedKMeans(KMeans):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centroid_history = []

    def _single_run(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        centroids = self._kpp_init(data, self.k)
        self.centroid_history.append(centroids.copy())
        for _ in range(self.max_iters):
            dist = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(dist, axis=1)
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.k)])
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                break
            centroids = new_centroids
            self.centroid_history.append(centroids.copy())
        inertia = np.sum([np.linalg.norm(data[i] - centroids[labels[i]])**2 for i in range(len(data))])
        return centroids, labels, inertia

df = pd.read_csv("../data/2018_2022_corrigido.csv")
df = df[df['Data'].str.contains('2021|2022', na=False)]
df = df[df['CNPJ'].notnull()]
df['CNPJ'] = df['CNPJ'].astype(str)
df = df[['CNPJ', 'Valor_corrigido']]

results = defaultdict()
kmeans_class_objs = []
sorted_data = sorted(zip(df['CNPJ'], df['Valor_corrigido']), key=lambda x: x[0])
selected_cnpjs = []
selected_values = []
counter = 0
for company, group in groupby(sorted_data, key=lambda x: x[0]):
    values = np.array([item[1] for item in group])
    if len(values) > 20:
        kmeans_obj = AnimatedKMeans()
        kmeans_obj.fit(values.reshape(-1, 1))
        anomalies_kmeans = kmeans_obj.detect(values.reshape(-1, 1))
        results[company] = anomalies_kmeans.tolist()
        selected_cnpjs.append(company)
        selected_values.append(values)
        kmeans_class_objs.append(kmeans_obj)
        counter += 1
        if counter == 12:
            break

max_iterations = max([len(kmeans_obj.centroid_history) for kmeans_obj in kmeans_class_objs])
for kmeans_obj in kmeans_class_objs:
    while len(kmeans_obj.centroid_history) < max_iterations:
        kmeans_obj.centroid_history.append(kmeans_obj.centroid_history[-1])

for kmeans_obj in kmeans_class_objs:
    for _ in range(20):
        kmeans_obj.centroid_history.append(kmeans_obj.centroid_history[-1])

def animate_multiple_centroids(i):
    ax.clear()
    for idx, (values, kmeans_obj) in enumerate(zip(selected_values, kmeans_class_objs)):
        ax.scatter(values, [idx + 1] * len(values), alpha=0.6, edgecolors="w", linewidth=0.5, label=selected_cnpjs[idx], marker=".")
        centroids = kmeans_obj.centroid_history[i]
        ax.scatter(centroids, [idx + 1] * len(centroids), c='red', marker='+', s=100)
        
        if i >= len(kmeans_obj.centroid_history) - 20:
            anomalies = results[selected_cnpjs[idx]]
            ax.scatter(anomalies, [idx + 1] * len(anomalies), c='blue', marker='s', s=50, alpha=0.7)
            
    ax.set_title(fr"$k=2$, iteration {i}")
    ax.set_yticks(range(1, len(selected_cnpjs) + 1))
    ax.set_xlabel("values")
    ax.set_ylabel("companies")

plt.rcParams["font.family"] = "monospace"
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Centroids movement in custom k-means algorithm for univariate data', fontweight='bold')
ani_multiple = animation.FuncAnimation(fig, animate_multiple_centroids, frames=len(kmeans_class_objs[0].centroid_history), repeat=True)
plt.close(fig)
ani_multiple.save("../assets/anomalies_animation2.mp4", writer='ffmpeg', fps=10)