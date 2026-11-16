import unittest

import numpy as np

from annotate_sarcgraph_with_regions import edge_key, region_evidence


class RegionEvidenceTests(unittest.TestCase):
    def test_full_region_support(self):
        probabilities = np.zeros((3, 64, 64)); probabilities[1] = 1
        evidence = region_evidence([20, 10], [20, 50], probabilities, 1)
        self.assertEqual(evidence['region_status'], 'region_supported')
        self.assertEqual(evidence['corridor_coverage'], 1)

    def test_absent_region_is_not_anatomical_rejection(self):
        probabilities = np.zeros((3, 64, 64)); probabilities[0] = 1
        evidence = region_evidence([20, 10], [20, 50], probabilities, 1)
        self.assertEqual(evidence['region_status'], 'weak_region_evidence')
        self.assertNotIn('delete', evidence)

    def test_reversed_edge_same_evidence(self):
        rng = np.random.default_rng(42)
        probabilities = rng.random((3, 64, 64))
        a = region_evidence([20, 10], [20, 50], probabilities, 1)
        b = region_evidence([20, 50], [20, 10], probabilities, 1)
        self.assertEqual(a['region_status'], b['region_status'])
        self.assertAlmostEqual(a['region_support_score'], b['region_support_score'])

    def test_duplicate_center_refused(self):
        with self.assertRaises(ValueError):
            region_evidence([1, 1], [1, 1], np.ones((3, 8, 8)), 1)

    def test_undirected_identity(self):
        self.assertEqual(edge_key(1, 5), edge_key(5, 1))


if __name__ == '__main__':
    unittest.main()
