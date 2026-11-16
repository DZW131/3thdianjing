import unittest

import numpy as np
import torch

from pilot_myofibril_region_learning import RegionUNet, confusion, normalize, predict, scores, starts


class RegionLearningTests(unittest.TestCase):
    def test_odd_shape_forward(self):
        torch.set_num_threads(2)
        model = RegionUNet().eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(1, 1, 65, 79)).shape), (1, 3, 65, 79))

    def test_tile_coverage(self):
        for length in [1, 127, 128, 129, 256, 620]:
            covered = np.zeros(max(length, 128), dtype=bool)
            for start in starts(max(length, 128), 128, 96):
                covered[start:start+128] = True
            self.assertTrue(covered.all())

    def test_prediction_small_image(self):
        torch.set_num_threads(2)
        prob = predict(RegionUNet(), np.zeros((17, 29), dtype=np.float32))
        self.assertEqual(prob.shape, (3, 17, 29))
        np.testing.assert_allclose(prob.sum(0), 1, atol=1e-6)

    def test_confusion_ignores_conflicts(self):
        target = np.array([[0, 1], [2, 255]], dtype=np.uint8)
        pred = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        matrix = confusion(target, pred)
        np.testing.assert_array_equal(matrix, np.eye(3, dtype=int))
        self.assertEqual(scores(matrix)['myofibril']['dice'], 1)

    def test_constant_normalization(self):
        out = normalize(np.full((3, 4), 10))
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((out == 0).all())

    def test_normalization_dtype_and_range(self):
        out = normalize(np.arange(30, dtype=np.uint16).reshape(5, 6))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), 0)
        self.assertLessEqual(float(out.max()), 1)


if __name__ == '__main__':
    unittest.main()
