from typing import Tuple
import numpy as np


class KMeans:
    """
    k-means com critérios de convergência aprimorados.

    Atributos:
        k (int): Número de clusters.
        max_iters (int): Número máximo de iterações para o k-means.
        tol (float): Tolerância de convergência baseada no movimento do centroide.
        n_init (int): Número de vezes que o algoritmo será executado com diferentes seeds de centroides.
        threshold (int): Percentil para detecção de anomalias.
        centroids (np.ndarray): Centroides para os clusters.
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
        Inicialização com parâmetros especificados.
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
        Inicializa os centroides usando o método k-means++.

        Argumentos:
            data (np.ndarray): Dados de entrada.
            k (int): Número de centroides desejados.

        Retorna:
            centroids (np.ndarray): Centroides inicializados.
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
        Realiza execução única do algoritmo k-means.

        Argumentos:
            data (np.ndarray): Dados de entrada.

        Retorna:
            centroids (np.ndarray): Melhores centroides após a execução do k-means.
            labels (np.ndarray): Atribuições de cluster para cada ponto de dado.
            inertia (float): Distância total dos pontos de dados a partir de seus centroides atribuídos.
        """
        centroids = self._kpp_init(data, self.k)
        for _ in range(self.max_iters):
            dist = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(dist, axis=1)
            new_centroids = np.array(
                [data[labels == i].mean(axis=0) for i in range(self.k)])
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                break
            centroids = new_centroids
        inertia = np.sum(
            [np.linalg.norm(data[i] - centroids[labels[i]])**2 for i in range(len(data))])
        
        return centroids, labels, inertia

    def fit(self, data: np.ndarray) -> None:
        """
        Ajusta o algoritmo k-means aos dados.

        Args:
            data (np.ndarray): Dados de entrada.
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
        Detecta anomalias nos dados com base na distância ao centroide mais próximo.

        Argumentos:
            data (np.ndarray): Dados de entrada.

        Retorna:
            anomalies (np.ndarray): Anomalias detectadas.
        """
        dist = np.min(np.linalg.norm(
            data[:, np.newaxis] - self.centroids, axis=2), axis=1)
        threshold = np.percentile(dist, self.threshold)
        anomalies = data[dist > threshold]
        return anomalies

    def get_labels(self, data: np.ndarray) -> np.ndarray:
        """
        Atribui cada ponto de dado ao centroide mais próximo para determinar seu cluster.

        Argumentos:
            data (np.ndarray): Conjunto de dados.

        Retorna:
            np.ndarray: Array de labels de cluster correspondentes a cada ponto de dado.
        """
        dist = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(dist, axis=1)
        return labels
