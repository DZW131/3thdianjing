import unittest

import numpy as np

from pilot_fused_sarcomere_regions import audit_pair, common_strip, prepare_regions, raster


class FusedRegionTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[20., 30.], [80., 30.]])
        self.b = np.array([[20., 90.], [80., 90.]])
        labels = np.zeros((100, 150), np.uint8)
        labels[20:81, 10:141] = 1
        self.fields = prepare_regions(labels, closing_radius=0)

    def audit(self, segments):
        raw, region, components, skeleton = self.fields
        return audit_pair(self.a, self.b, segments, {0, 1}, raw, components, skeleton, 1)

    def test_supported_synthetic_cell(self):
        result, vertices, path = self.audit([self.a, self.b])
        self.assertEqual(result['status'], 'region_supported_hypothesis')
        self.assertEqual(result['boundary_midpoint_distance_px'], 60)
        self.assertTrue(np.isfinite(path).all())
        self.assertEqual(vertices.shape, (4, 2))

    def test_intervening_z_rejected(self):
        third = np.array([[20., 60.], [80., 60.]])
        result, _, _ = self.audit([self.a, self.b, third])
        self.assertEqual(result['reason'], 'intervening_z')

    def test_no_region_cannot_pass(self):
        self.fields = prepare_regions(np.zeros((100, 150), np.uint8), closing_radius=0)
        result, _, _ = self.audit([self.a, self.b])
        self.assertEqual(result['reason'], 'no_region')

    def test_crossed_z_geometry_rejected(self):
        horizontal = np.array([[50., 10.], [50., 100.]])
        self.assertIsNone(common_strip(self.a, horizontal))

    def test_reversed_endpoints_same_geometry(self):
        forward = common_strip(self.a, self.b)
        reversed_b = common_strip(self.a, self.b[::-1])
        np.testing.assert_allclose(forward[0], reversed_b[0])
        np.testing.assert_allclose(forward[1], reversed_b[1])

    def test_small_components_removed(self):
        labels = np.zeros((30, 30), np.uint8)
        labels[2, 2] = 1
        _, region, components, _ = prepare_regions(labels, closing_radius=0)
        self.assertFalse(region.any())
        self.assertEqual(components.max(), 0)

    def test_row_column_polygon(self):
        vertices = np.array([[2, 10], [5, 10], [5, 20], [2, 20]])
        mask = raster(vertices, (30, 40))
        self.assertTrue(mask[3, 15])
        self.assertFalse(mask[15, 3])


if __name__ == '__main__':
    unittest.main()
