import numpy as np

class Score:
    """
    Cálculo de scoring para algoritmo de clusterização.
    """

    @staticmethod
    def silhouette(data: np.ndarray, labels: np.ndarray) -> float:
        """
        Calcula o Silhouette Score.

        Argumentos:
            data (np.ndarray): Dados de entrada.
            labels (np.ndarray): Atribuições de cluster para cada ponto de dado.

        Retorna:
            float: Silhouette Score calculado.
        """
        unique_labels = np.unique(labels)
        silhouette_vals = []

        for index, label in enumerate(labels):
            same_cluster = data[labels == label]
            a = np.mean(np.linalg.norm(same_cluster - data[index], axis=1))
            other_clusters = [data[labels == other_label]
                              for other_label in unique_labels if other_label != label]
            b_vals = [np.mean(np.linalg.norm(cluster - data[index], axis=1))
                      for cluster in other_clusters]
            b = min(b_vals)
            silhouette_vals.append((b - a) / max(a, b))

        return np.mean(silhouette_vals)

    @staticmethod
    def daviesbouldin(data: np.ndarray, labels: np.ndarray) -> float:
        """
        Calcula o Davies-Bouldin Score.

        Argumentos:
            data (np.ndarray): Dados de entrada.
            labels (np.ndarray): Atribuições de cluster para cada ponto de dado.

        Returns:
            float: Davies-Bouldin Score calculado.
        """
        unique_labels = np.unique(labels)
        centroids = np.array([data[labels == label].mean(axis=0)
                             for label in unique_labels])
        avg_dist_within_cluster = np.array([np.mean(np.linalg.norm(
            data[labels == label] - centroids[label], axis=1)) for label in unique_labels])
        centroid_dist = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)
        np.fill_diagonal(centroid_dist, float('inf'))

        cluster_ratios = (avg_dist_within_cluster[:, np.newaxis] + avg_dist_within_cluster) / centroid_dist
        max_cluster_ratios = np.max(cluster_ratios, axis=1)
        return np.mean(max_cluster_ratios)