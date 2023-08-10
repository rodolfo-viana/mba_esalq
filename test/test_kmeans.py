import sys
sys.path.append('../src')
from src.kmeans import KMeans
import unittest
import numpy as np

class TestKMeans(unittest.TestCase):

    def setUp(self):
        """Set up a sample dataset and KMeans object for testing."""
        self.data = np.array([10, 12, 10, 11, 18, 55, 56, 57])
        self.kmeans = KMeans()

    def test_initialization(self):
        """Test initialization of the KMeans object with default and custom parameters."""
        self.assertEqual(self.kmeans.k, 2)
        self.assertEqual(self.kmeans.max_iters, 100)
        self.assertEqual(self.kmeans.tol, 1e-4)
        
        custom_kmeans = KMeans(k=3, max_iters=50, tol=1e-3)
        self.assertEqual(custom_kmeans.k, 3)
        self.assertEqual(custom_kmeans.max_iters, 50)
        self.assertEqual(custom_kmeans.tol, 1e-3)

    def test_kpp_init(self):
        """Test centroid initialization using the k-means++ method."""
        centroids = self.kmeans._kpp_init(self.data, 2)
        self.assertEqual(centroids.shape, (2,))

    def test_fit(self):
        """Test fitting the KMeans algorithm to sample data."""
        self.kmeans.fit(self.data)
        self.assertIsNotNone(self.kmeans.centroids)
        self.assertIsNotNone(self.kmeans.clusters)
        self.assertEqual(self.kmeans.centroids.shape, (2,))

    def test_detect(self):
        """Test detecting anomalies based on distance to centroids."""
        self.kmeans.fit(self.data)
        anomalies = self.kmeans.detect(self.data)
        # Check if any anomalies were detected
        self.assertTrue(len(anomalies) > 0)
        # Check if the detected anomaly is 18
        self.assertEqual(anomalies[0], 18)

    def tearDown(self):
        """Tear down the test setup."""
        self.data = None
        self.kmeans = None


