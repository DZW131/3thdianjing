import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from utils.metrics import Evaluator
from utils.checkpoint import load_checkpoint
from utils.jijie.postprocess import postprocess_mito_mask


class HandoffTests(unittest.TestCase):
    def test_ignore_and_confusion_metrics(self):
        e = Evaluator(3)
        e.add_batch(np.array([[1, 1, 2, 255]]), np.array([[1, 2, 2, 1]]))
        self.assertEqual(e.confusion_matrix.sum(), 3)
        np.testing.assert_allclose(e.per_class_iou()[1:], [0.5, 0.5])
        np.testing.assert_allclose(e.per_class_dice()[1:], [2/3, 2/3])

    def test_checkpoint_roundtrip(self):
        m = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pth"
            torch.save({"state_dict": m.state_dict(), "epoch": 1}, path)
            n = torch.nn.Linear(2, 2)
            load_checkpoint(path, n)
            self.assertTrue(torch.equal(m.weight, n.weight))

    def test_postprocess_empty(self):
        mask = np.zeros((16, 16), dtype=np.int32)
        result, stats = postprocess_mito_mask(mask, use_texture_refine=False)
        np.testing.assert_array_equal(result, mask)
        self.assertEqual(stats["unified_count"], 0)

    def test_weight_inventory(self):
        rows = json.loads(Path("docs/weights_inventory.json").read_text())
        self.assertEqual(len(rows), len({r["path"] for r in rows}))
        for row in rows:
            self.assertEqual(len(row["sha256"]), 64)
            self.assertGreater(row["size_bytes"], 0)
            self.assertFalse(Path(row["path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
