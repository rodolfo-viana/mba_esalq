from typing import Tuple
import numpy as np


class KMeans:
    """
    k-means clustering class with enhanced convergence criteria.

    Attributes:
        k (int): Number of clusters.
        max_iters (int): Maximum number of iterations for k-means.
        tol (float): Convergence tolerance based on centroid movement.
        n_init (int): Number of times the algorithm will be run with different centroid seeds.
        threshold (int): Percentile for anomaly detection.
        centroids (np.ndarray): Centroids for the clusters.
    """

    def __init__(self,
                 k: int = 2,
                 max_iters: int = 100,
                 tol: float = 1e-4,
                 n_init: int = 30,
                 threshold: int = 95,
                 centroids: np.ndarray = None
                 ):
        """
        Initialize k-means with specified parameters.
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init
        self.threshold = threshold
        self.centroids = centroids

    @staticmethod
    def _kpp_init(data: np.ndarray, k: int) -> np.ndarray:
        """
        Initialize the centroids using the k-means++ method.

        Args:
            data (np.ndarray): Input data.
            k (int): Number of desired centroids.

        Returns:
            centroids (np.ndarray): Initialized centroids.
        """
        centroids = [data[np.random.choice(len(data))]]
        for _ in range(1, k):
            squared_dist = np.array(
                [np.min([np.linalg.norm(c - x) ** 2 for c in centroids]) for x in data])
            probs = squared_dist / squared_dist.sum()
            centroid = data[np.argmax(probs)]
            centroids.append(centroid)
        return np.array(centroids)

    def _single_run(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform a single run of the k-means algorithm.

        Args:
            data (np.ndarray): Input data.

        Returns:
            centroids (np.ndarray): Best centroids after running k-means.
            labels (np.ndarray): Cluster assignments for each data point.
            inertia (float): Total distance of data points from their assigned centroids.
        """
        centroids = self._kpp_init(data, self.k)
        for _ in range(self.max_iters):
            dist = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(dist, axis=1)
            new_centroids = np.array(
                [data[labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                break

            centroids = new_centroids

        # Calculate inertia (sum of squared distances to the nearest centroid)
        inertia = np.sum(
            [np.linalg.norm(data[i] - centroids[labels[i]])**2 for i in range(len(data))])
        return centroids, labels, inertia

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the k-means algorithm to the data.

        Args:
            data (np.ndarray): Input data.
        """
        min_inertia = float('inf')
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            centroids, labels, inertia = self._single_run(data)
            if inertia < min_inertia:
                min_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels = best_labels

    def detect(self, data: np.ndarray) -> np.ndarray:
        """
        Detect anomalies in the data based on the distance to the nearest centroid.

        Args:
            data (np.ndarray): Input data.

        Returns:
            anomalies (np.ndarray): Detected anomalies.
        """
        dist = np.min(np.linalg.norm(
            data[:, np.newaxis] - self.centroids, axis=2), axis=1)
        threshold = np.percentile(dist, self.threshold)
        anomalies = data[dist > threshold]
        return anomalies
