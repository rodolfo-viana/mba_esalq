from typing import Tuple
import math
from collections import defaultdict
from itertools import groupby
import numpy as np
import pandas as pd


class KMeans:
    """
    KMeans clustering class with enhanced convergence criteria.

    Attributes:
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations for KMeans.
        tolerance (float): Convergence tolerance based on centroid movement.
        n_init (int): Number of times the k-means algorithm will be run with different centroid seeds.
        threshold_percentile (int): Percentile for anomaly detection.
    """

    # TODO: Think of a way to pick `k` dinamically (elbow method, perhaps?)
    def __init__(self,
                 k: int = 2,
                 max_iters: int = 100,
                 tolerance: float = 1e-4,
                 n_init: int = 30,
                 threshold_percentile: int = 95):
        """
        Initialize KMeans with specified parameters.
        """
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.n_init = n_init
        self.threshold_percentile = threshold_percentile

    @staticmethod
    def _kpp_init(data: np.ndarray, k: int) -> np.ndarray:
        """
        Initialize the centroids using the k-means++ method.

        Parameters:
            data (array-like): Input data.
            k (int): Number of desired centroids.

        Returns:
            array-like: Initialized centroids.
        """
        centroids = [data[np.random.choice(len(data))]]
        for _ in range(1, k):
            squared_distances = np.array(
                [min([math.pow(c-x, 2) for c in centroids]) for x in data])
            probs = squared_distances / squared_distances.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break
            centroids.append(data[i])
        return np.array(centroids)

    def _single_iter(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run a single iteration of the KMeans algorithm.

        Parameters:
            data (array-like): Input data.

        Returns:
            array-like: Best centroids after running KMeans.
            array-like: Cluster assignments for each data point.
            float: Total distance of data points from their assigned centroids.
        """
        centroids = self._kpp_init(data, self.k)
        previous_centroids = np.zeros_like(centroids)

        for _ in range(self.max_iters):
            distances = np.abs(data[:, np.newaxis] - centroids)
            clusters = np.argmin(distances, axis=1)
            new_centroids = np.array(
                [data[clusters == i].mean() for i in range(self.k)])

            centroid_shift = np.linalg.norm(new_centroids - previous_centroids)
            if centroid_shift < self.tolerance:
                break

            previous_centroids = centroids.copy()
            centroids = new_centroids

        current_distance = np.sum(
            [np.abs(data[clusters == i] - centroids[i]).sum() for i in range(self.k)])
        return centroids, clusters, current_distance

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the KMeans algorithm to the data.

        Parameters:
            data (array-like): Input data.
        """
        best_centroids = None
        best_clusters = None
        min_distance = np.inf

        for _ in range(self.n_init):
            centroids, clusters, distance = self._single_iter(data)
            if distance < min_distance:
                min_distance = distance
                best_centroids = centroids
                best_clusters = clusters

        self.centroids = best_centroids
        self.clusters = best_clusters

    def detect_anomalies(self, data: np.ndarray) -> np.ndarray:
        """
        Detect anomalies in the data based on the distance to the nearest centroid.

        Parameters:
            data (array-like): Input data.

        Returns:
            array-like: Detected anomalies.
        """
        distances = np.array([math.fabs(price - self.centroids[cluster])
                             for price, cluster in zip(data, self.clusters)])
        threshold = np.percentile(distances, self.threshold_percentile)
        anomalies = data[distances > threshold]
        return anomalies

    @staticmethod
    def keyfunc(x: Tuple[str, float]) -> str:
        """
        Key function for itertools.groupby. Assumes x is a tuple where the first element is the key.
        """
        return x[0]
