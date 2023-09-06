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
        # selciona o primeiro centroide randomicamente
        centroids = [data[np.random.choice(len(data))]]
        # looping para escolher os k-1 centroides restantes
        for _ in range(1, k):
            # calcula a distância ao quadrado mínima de cada ponto de dado em relação aos centroides já selecionados
            squared_dist = np.array(
                [np.min([np.linalg.norm(c - x) ** 2 for c in centroids]) for x in data])
            # calcula a distribuição de probabilidades
            probs = squared_dist / squared_dist.sum()
            # seleciona o ponto de dados com maior probabilidade para ser próximo centroide
            centroid = data[np.argmax(probs)]
            # adiciona à lista de centroides
            centroids.append(centroid)
        # retorna os centroides inicializados
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
        # looping para o número máximo de iterações
        for _ in range(self.max_iters):
            # calcula a distância euclidiana entre cada ponto de dado e cada centroide
            dist = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            # atribui cada ponto de dado ao centroide mais próximo
            labels = np.argmin(dist, axis=1)
            # recalcula os centroides com base na média dos pontos de dados em cada cluster
            new_centroids = np.array(
                [data[labels == i].mean(axis=0) for i in range(self.k)])
            # observa a convergência e encerra o looping se a mudança de centroides estiver abaixo da tolerância
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                break
            # atualiza os centroides
            centroids = new_centroids
        # calcula a distância total entre os pontos de dados e os centroides a eles atribuídos
        inertia = np.sum(
            [np.linalg.norm(data[i] - centroids[labels[i]])**2 for i in range(len(data))])
        # retorna os centroides finais, as labels atribuídas e a inércia
        return centroids, labels, inertia

    def fit(self, data: np.ndarray) -> None:
        """
        Ajusta o algoritmo k-means aos dados.

        Args:
            data (np.ndarray): Dados de entrada.
        """
        # ajusta a inércia mínima inicial a valor infinito
        min_inertia = float('inf')
        # atribuiu o valor None ao melhores centroides e labels
        best_centroids = None
        best_labels = None
        # looping para o número de inicializações
        for _ in range(self.n_init):
            # executa `_single_run`
            centroids, labels, inertia = self._single_run(data)
            # observa se a execução atual tem inércia menor do que a melhor inércia
            if inertia < min_inertia:
                # em caso positivo, atualiza inércia, centroides e labels
                min_inertia = inertia
                best_centroids = centroids
                best_labels = labels
        # atribuiu novos melhores centroides e labels à classe `KMeans`
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
        # calcula a distância mínima de cada ponto de dado em relação a seu centroide
        dist = np.min(np.linalg.norm(
            data[:, np.newaxis] - self.centroids, axis=2), axis=1)
        # determina o limite da distância com base no percentil de KMeans
        threshold = np.percentile(dist, self.threshold)
        # identifica pontos de dados com distâncias maiores do que o limite
        anomalies = data[dist > threshold]
        # retorna os valores anômalos
        return anomalies

    def get_labels(self, data: np.ndarray) -> np.ndarray:
        """
        Atribui cada ponto de dado ao centroide mais próximo para determinar seu cluster.

        Argumentos:
            data (np.ndarray): Conjunto de dados.

        Retorna:
            np.ndarray: Array de labels de cluster correspondentes a cada ponto de dado.
        """
        # calcula a distância de cada ponto de dado em relação aos centroides
        dist = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        # atribuiu cada ponto ao centroide mais próximo
        labels = np.argmin(dist, axis=1)
        # retorna as labels atribuídas
        return labels
