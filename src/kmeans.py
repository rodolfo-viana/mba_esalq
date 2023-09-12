from typing import Tuple
import numpy as np
import pandas as pd


class KMeans:
    """
    k-means com critérios de convergência aprimorados.

    Atributos:
        k (int): Número de clusters.
        max_iters (int): Número máximo de iterações para o k-means.
        tol (float): Tolerância de convergência baseada no movimento do
            centroide.
        n_init (int): Número de vezes que o algoritmo será executado com
            diferentes seeds de centroides.
        threshold (int): Percentil para detecção de anomalias.
        centroids (np.ndarray): Centroides para os clusters.
    """

    def __init__(
        self,
        k: int = 2,
        max_iters: int = 100,
        tol: float = 1e-4,
        n_init: int = 30,
        threshold: int = 95,
        centroids: np.ndarray = None,
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
                [np.min([np.linalg.norm(c - x) ** 2 for c in centroids]) for x in data]
            )
            probs = squared_dist / squared_dist.sum()
            centroid = data[np.argmax(probs)]
            centroids.append(centroid)
        return np.array(centroids)

    def get_optimal_k(self, data: np.ndarray, k_max: int = 10) -> int:
        """
        Determina o número ideal de clusters k com o método Silhouette.

        Argumentos:
            data (np.ndarray): Dados sobre os quais o número ideal de k
                será determinado.
            k_max (int, opcional): Valor máximo de k. Valor padrão: 10.

        Retorna:
            optimal_k (int): Número ideal de clusters.
        """
        max_silhouette = -1
        optimal_k = 2

        def get_score(data, labels):
            """
            Calcula a média de Silhouette score dadas as labels de
            clusterização do conjunto de dados.

            Argumentos:
                data (np.ndarray): Dados de entrada.
                labels (np.ndarray): Labels da clusterização para os
                    pontos de dados.

            Retorna:
                (float): Silhouette score médio.
            """
            unique_labels = np.unique(labels)
            silhouettes = []
            for i, label in enumerate(labels):
                points_within_cluster = data[labels == label]
                avg_dist_within_cluster = np.mean(
                    np.linalg.norm(points_within_cluster - data[i], axis=1)
                )
                min_avg_dists = [
                    np.mean(
                        np.linalg.norm(data[labels == other_label] - data[i], axis=1)
                    )
                    for other_label in unique_labels
                    if other_label != label
                ]
                silhouette_value = (
                    np.min(min_avg_dists) - avg_dist_within_cluster
                ) / max(avg_dist_within_cluster, np.min(min_avg_dists))
                silhouettes.append(silhouette_value)
            return np.mean(silhouettes)

        for k in range(2, k_max + 1):
            self.k = k
            self.fit(data)
            silhouette_avg = get_score(data, self.labels)
            if silhouette_avg > max_silhouette:
                max_silhouette = silhouette_avg
                optimal_k = k
        return optimal_k

    def _single_run(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Realiza execução única do algoritmo k-means.

        Argumentos:
            data (np.ndarray): Dados de entrada.

        Retorna:
            centroids (np.ndarray): Melhores centroides após a execução
                do k-means.
            labels (np.ndarray): Atribuições de cluster para cada ponto
                de dado.
            inertia (float): Distância total dos pontos de dados a
                partir de seus centroides atribuídos.
        """
        centroids = self._kpp_init(data, self.k)
        for _ in range(self.max_iters):
            dist = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(dist, axis=1)
            new_centroids = np.array(
                [data[labels == i].mean(axis=0) for i in range(self.k)]
            )
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                break
            centroids = new_centroids
        inertia = np.sum(
            [
                np.linalg.norm(data[i] - centroids[labels[i]]) ** 2
                for i in range(len(data))
            ]
        )
        return centroids, labels, inertia

    def fit(self, data: np.ndarray) -> None:
        """
        Ajusta o algoritmo k-means aos dados.

        Argumento:
            data (np.ndarray): Dados de entrada.
        """
        min_inertia = float("inf")
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
        Detecta anomalias nos dados com base na distância ao centroide
        mais próximo.

        Argumentos:
            data (np.ndarray): Dados de entrada.

        Retorna:
            anomalies (np.ndarray): Anomalias detectadas.
        """
        dist = np.min(
            np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2), axis=1
        )
        threshold = np.percentile(dist, self.threshold)
        anomalies = data[dist > threshold]
        return anomalies


def run_kmeans(group: pd.DataFrame, num_var: str) -> pd.DataFrame:
    """
    Executa o algoritmo de K-Means em CNPJ único e adiciona informações
    sobre quais valores são observados como anomalias e quantos k foram
    usados para obter tal resultado.

    Argumentos:
        group (pd.DataFrame): Grupo de CNPJs elegíveis para o algoritmo.
        num_var (str): Nome da variável numérica sobre a qual o
            algoritmo será rodado.

    Retorna:
        (pd.DataFrame): Dataframe com os dados originais e as colunas
            'Anomalia' (com valor 0 para não anomalia e 1 para anomalia)
            e 'k' (o número ideal de clusters usados pelo K-Means).
    """
    kmeans = KMeans()
    data = group[num_var].values.reshape(-1, 1)
    k_optimal = kmeans.get_optimal_k(data)
    kmeans.k = k_optimal
    kmeans.fit(data)
    anomalies = kmeans.detect(data).flatten()
    group["Anomalia"] = group[num_var].isin(anomalies).astype(int)
    group["k"] = k_optimal
    return group
