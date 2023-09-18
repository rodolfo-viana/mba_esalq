from typing import Tuple
import numpy as np


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
        # seleciona ponto aleatório como centroide
        centroids = [data[np.random.choice(len(data))]]

        # itera sobre centroides restantes
        for _ in range(1, k):
            # calcula o quadrado da distância entre cada ponto e o
            # centroide mais próximo
            squared_dist = np.array(
                [np.min([np.linalg.norm(c - x) ** 2 for c in centroids]) for x in data]
            )
            # calcula a probabilidade de selecionar cada ponto de dado
            # como novo centroide
            probs = squared_dist / squared_dist.sum()
            # escolhe o ponto com maior probabilidade como novo
            # centroide
            centroid = data[np.argmax(probs)]
            # adiciona novo centroide à lista de centroides
            centroids.append(centroid)
        return np.array(centroids)

    def get_optimal_k(self, data: np.ndarray, k_max: int = 10) -> int:
        """
        Aplica método Elbow para obter o número de clusters ideal.

        Argumentos:
            data (np.ndarray): Dados usados no algoritmo K-Means.
            k_max (int): Número máximo de clusters. Valor-padrão: 10.

        Retorna:
            optimal_k (int): Número de clusters ideal.
        """
        # lista para armazenar inércia de cada k
        sum_sq = []
        # itera sobre intervalo de 1 a 10
        for k in range(1, k_max + 1):
            # ajusta o número de clusters para a iteração atual
            self.k = k
            # ajusta os dados ao algoritmo
            self.fit(data)
            # calcula a inércia
            inertia = np.sum(
                [
                    np.linalg.norm(data[i] - self.centroids[self.labels[i]]) ** 2
                    for i in range(len(data))
                ]
            )
            # adiciona a inércia à lista
            sum_sq.append(inertia)
        # calcula a diferença dos valores de inércia para encontrar o
        # cotovelo
        diffs = np.diff(sum_sq, 2)
        # escolhe k ideal a partir da menor diferença
        optimal_k = np.argmin(diffs) + 1
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
        # inicializa centoides
        centroids = self._kpp_init(data, self.k)

        # itera sobre max_iters:
        for _ in range(self.max_iters):
            # calcula a distância entre cada ponto e cada centroide
            dist = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            # atribui cada ponto ao centroide mais próximo
            labels = np.argmin(dist, axis=1)
            # calcula os novos centroides com base na atribuição recente
            new_centroids = np.array(
                [data[labels == i].mean(axis=0) for i in range(self.k)]
            )
            # observa se a mudança no centroide está abaixo da
            # tolerância
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                # interrompe a iteração
                break
            # sobrescreve lista de centroides
            centroids = new_centroids
        # calcula a inércia
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
        # atribui valor infinito à inércia mínima
        min_inertia = float("inf")
        # atribui None aos melhores centroides
        best_centroids = None
        # atribui None às melhores labels
        best_labels = None

        # itera sobre quantidade de execuções de K-Means
        for _ in range(self.n_init):
            # obtém valores de centroides, labels, inécia
            centroids, labels, inertia = self._single_run(data)
            # observa se a execução atual tem menor inércia
            if inertia < min_inertia:
                # atualiza inércia mínima
                min_inertia = inertia
                # atualiza melhores centroides
                best_centroids = centroids
                # atualiza melhores labels
                best_labels = labels

        # ajusta os valores de centroides para os melhores valores
        # encontrados
        self.centroids = best_centroids
        # ajusta os valores de labels para os melhores valores
        # encontrados
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
        # calcula a distância entre cada ponto e o centroide mais
        # próximo
        dist = np.min(
            np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2), axis=1
        )
        # ajusta o limite com base no percentil inserido
        threshold = np.percentile(dist, self.threshold)
        # considera anomalias os pontos cujas distâncias são maiores que
        # o limite
        anomalies = data[dist > threshold]
        return anomalies

    def get_labels(self, data: np.ndarray) -> np.ndarray:
        """
        Atribui cada ponto de dado ao centroide mais próximo para
        determinar seu cluster.

        Argumento:
            data (np.ndarray): Conjunto de dados.

        Retorna:
            labels (np.ndarray): Array de labels de cluster
                correspondentes a cada ponto de dado.
        """
        # calcula a distância de cada ponto a cada centroide
        dist = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        # atribui cada ponto ao centroide mais próximo
        labels = np.argmin(dist, axis=1)
        return labels


class Score:
    """
    Cálculo de scoring para algoritmo de clusterização.
    """

    @staticmethod
    def silhouette(data: np.ndarray, labels: np.ndarray) -> float:
        """
        Calcula o score do método da silhueta.

        Argumentos:
            data (np.ndarray): Dados de entrada.
            labels (np.ndarray): Atribuições de cluster para cada ponto
                de dado.

        Retorna:
            float: valor do método da silhueta.
        """
        # obtém labels únicas
        unique_labels = np.unique(labels)
        # lista para armazenar valores do método da silhueta
        silhouette_vals = []
        # itera sobre pontos de dados
        for index, label in enumerate(labels):
            # obtém pontos que estão no mesmo cluster
            same_cluster = data[labels == label]
            # calcula a distância média a outros pontos no mesmo cluster
            a = np.mean(np.linalg.norm(same_cluster - data[index], axis=1))
            # extrai pontos de outros clusters
            other_clusters = [
                data[labels == other_label]
                for other_label in unique_labels
                if other_label != label
            ]
            # calcula a distância média para pontos em outros clusters
            b_vals = [
                np.mean(np.linalg.norm(cluster - data[index], axis=1))
                for cluster in other_clusters
            ]
            # obtém os menores valores
            b = min(b_vals)
            # calcula o valor da silhueta
            silhouette_vals.append((b - a) / max(a, b))
        # retorna a silhueta média para todos os pontos
        return np.mean(silhouette_vals)

    @staticmethod
    def daviesbouldin(data: np.ndarray, labels: np.ndarray) -> float:
        """
        Calcula o índice de Davies-Bouldin.

        Argumentos:
            data (np.ndarray): Dados de entrada.
            labels (np.ndarray): Atribuições de cluster para cada ponto
                de dado.

        Retorna:
            float: valor de Davies-Bouldin calculado.
        """
        # obtém labels únicas
        unique_labels = np.unique(labels)
        # calcula o centroide para cada cluster
        centroids = np.array(
            [data[labels == label].mean(axis=0) for label in unique_labels]
        )
        # calcula a distância média dentro de cada cluster
        avg_dist_within_cluster = np.array(
            [
                np.mean(
                    np.linalg.norm(data[labels == label] - centroids[label], axis=1)
                )
                for label in unique_labels
            ]
        )
        # calcula a distância entre centroides
        centroid_dist = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)
        # ajusta valores diagonais para infinito
        np.fill_diagonal(centroid_dist, float("inf"))
        # calcula a razão entre a soma das distâncias médias e a
        # distância entre centroides
        cluster_ratios = (
            avg_dist_within_cluster[:, np.newaxis] + avg_dist_within_cluster
        ) / centroid_dist
        # obtém a maior razão para cada cluster
        max_cluster_ratios = np.max(cluster_ratios, axis=1)
        # retorna a média das maiores razões
        return np.mean(max_cluster_ratios)
