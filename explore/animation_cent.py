from typing import Tuple, List
from collections import defaultdict
from itertools import groupby
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
sys.path.insert(0, "..")
from src.kmeans import KMeans

class AnimatedKMeans(KMeans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centroids_history = []

    def _single_run(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        centroids = self._kpp_init(data, self.k)
        self.centroids_history.append(centroids.copy())
        
        for _ in range(self.max_iters):
            dist = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(dist, axis=1)
            new_centroids = np.array(
                [data[labels == i].mean(axis=0) for i in range(self.k)])
            
            # Record the centroids for animation
            self.centroids_history.append(new_centroids.copy())

            # Check for convergence
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                break

            centroids = new_centroids

        inertia = np.sum(
            [np.linalg.norm(data[i] - centroids[labels[i]])**2 for i in range(len(data))])
        return centroids, labels, inertia

    def reset(self):
        """Reset the centroids history."""
        self.centroids_history = []

df = pd.read_csv("../data/2018_2022_corrigido.csv")
df = df[df['CNPJ'].notnull()]
df['CNPJ'] = df['CNPJ'].astype(str)
df = df[['CNPJ', 'Valor_corrigido']]

# Running clustering and anomaly detection
results = defaultdict()
kmeans_class_obj = AnimatedKMeans()
sorted_data = sorted(zip(df['CNPJ'], df['Valor_corrigido']), key=lambda x: x[0])
for company, group in groupby(sorted_data, key=lambda x: x[0]):
    values = np.array([item[1] for item in group])
    if len(values) > 20:
        kmeans_class_obj.reset()
        kmeans_class_obj.fit(values.reshape(-1, 1))
        anomalies_kmeans = kmeans_class_obj.detect(values.reshape(-1, 1))
        flat_anomalies = [item[0] for item in anomalies_kmeans]
        results[company] = {
            "anomalies": flat_anomalies,
            "centroid_history": kmeans_class_obj.centroids_history
        }

# Animation for a selected company
company_data = list(results.items())[0]
company_name, data_dict = company_data
anomalies = data_dict['anomalies']
centroid_history = data_dict['centroid_history']

plt.ioff()

# Create the figure and axis for the animation
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the data points
ax.scatter(values, [1] * len(values), alpha=0.6, edgecolors="w", linewidth=0.5)
centroid_lines = [ax.plot(centroid, 1, 'ro')[0] for centroid in centroid_history[0]]
anomaly_points = ax.scatter([], [], color='red', s=100, edgecolors='black', label='Anomalies')

def update(num):
    # Update title based on iteration number
    if num < len(centroid_history):
        for line, centroid in zip(centroid_lines, centroid_history[num]):
            line.set_data(centroid, 1)
        anomaly_points.set_offsets([])
        plt.title(f'KMeans Centroids Movement (Iteration {num}) for Company: {company_name}')
    elif num == len(centroid_history) and anomalies:
        anomaly_coords = [(anomaly, 1) for anomaly in anomalies]
        anomaly_points.set_offsets(anomaly_coords)
        plt.title(f'KMeans Anomalies Detected for Company: {company_name}')
    return centroid_lines + [anomaly_points]

ani = animation.FuncAnimation(fig, update, frames=len(centroid_history) + 2, repeat=False, blit=True)

plt.xlabel('Valor_corrigido')
plt.yticks([])
plt.legend()
plt.tight_layout()

# Display the animation and turn interactive mode back on
plt.show()
plt.ion()