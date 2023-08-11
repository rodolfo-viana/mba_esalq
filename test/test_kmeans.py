import unittest
import sys
sys.path.insert(0, "..")
import numpy as np
from src.kmeans import KMeans


class TestKMeans(unittest.TestCase):

    def setUp(self):
        self.data = np.array([
            [1, 12],
            [5, 16],
            [1, 8],
            [5, 2],
            [8, 20],
            [9, 9]
        ])

    def test_initialization(self):
        kmeans = KMeans(k=3, max_iters=50, tol=1e-3, n_init=10, threshold=90)
        self.assertEqual(kmeans.k, 3)
        self.assertEqual(kmeans.max_iters, 50)
        self.assertEqual(kmeans.tol, 1e-3)
        self.assertEqual(kmeans.n_init, 10)
        self.assertEqual(kmeans.threshold, 90)

    def test_kpp_init(self):
        kmeans = KMeans()
        centroids = kmeans._kpp_init(self.data, 2)
        self.assertEqual(centroids.shape, (2, 2))
        self.assertNotEqual(tuple(centroids[0]), tuple(centroids[1]))

    def test_single_run(self):
        kmeans = KMeans()
        centroids, labels, inertia = kmeans._single_run(self.data)
        self.assertEqual(centroids.shape, (2, 2))
        self.assertEqual(labels.shape, (len(self.data),))
        self.assertTrue(isinstance(inertia, float))

    def test_fit(self):
        kmeans = KMeans()
        kmeans.fit(self.data)
        self.assertIsNotNone(kmeans.centroids)
        self.assertEqual(kmeans.centroids.shape, (2, 2))
        self.assertEqual(len(kmeans.labels), len(self.data))

    def test_detect(self):
        kmeans = KMeans()
        kmeans.fit(self.data)
        anomalies = kmeans.detect(self.data)
        self.assertTrue(isinstance(anomalies, np.ndarray))
        self.assertIn(5, anomalies)
        self.assertNotIn(1, anomalies)


if __name__ == "__main__":
    unittest.main()