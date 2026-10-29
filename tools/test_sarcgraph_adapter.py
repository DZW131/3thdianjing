import unittest
import numpy as np
import pandas as pd
from analyze_sarcomere_sarcgraph import objects_from_mask, direction_audit
from sarcgraph_geometry import merge_fragments, bundle_audit
from sarcgraph_normal_neighbors import normal_neighbors


class AdapterTests(unittest.TestCase):
    def test_nearest_normal_chain(self):
        m=np.zeros((80,180),dtype=np.uint8)
        for col in (10,60,110):
            m[20:60,col:col+3]=10
        t=objects_from_mask(m,10,4)
        pairs=normal_neighbors(t,min_length=20,max_length=150)
        selected=pairs[pairs.normal_neighbor_selected]
        self.assertEqual(set(zip(selected.z1,selected.z2)),{(0,1),(1,2)})

    def test_normal_rejects_along_z(self):
        m=np.zeros((120,30),dtype=np.uint8)
        m[10:30,10:13]=10
        m[70:90,10:13]=10
        self.assertTrue(normal_neighbors(objects_from_mask(m,10,4)).empty)

    def test_fragment_merge(self):
        m = np.zeros((60,80),dtype=np.uint8)
        m[10:20,10:12]=10
        m[23:33,10:12]=10
        m[10:33,40:42]=10
        t=objects_from_mask(m,10,4)
        merged,proposals,mapping=merge_fragments(t)
        self.assertEqual(len(merged),2)
        self.assertEqual(int(proposals.accepted.sum()),1)
        self.assertEqual(len(mapping),3)
        self.assertEqual(merged.area_px.sum(),t.area_px.sum())

    def test_ambiguous_merge_rejected(self):
        m=np.zeros((60,40),dtype=np.uint8)
        for start in (5,18,31):
            m[start:start+10,10:12]=10
        t=objects_from_mask(m,10,4)
        merged,proposals,_=merge_fragments(t)
        self.assertEqual(len(merged),3)
        self.assertFalse(proposals.accepted.any())

    def test_bundle_overlap(self):
        m=np.zeros((100,100),dtype=np.uint8)
        m[10:30,10:12]=10
        m[10:30,60:62]=10
        m[40:60,60:62]=10
        t=objects_from_mask(m,10,4)
        pairs=pd.DataFrame([dict(z1=0,z2=1),dict(z1=0,z2=2)])
        a=bundle_audit(t,direction_audit(t,pairs,90))
        self.assertTrue(a.bundle_geometry_pass.iloc[0])
        self.assertFalse(a.bundle_geometry_pass.iloc[1])

    def test_direction(self):
        m = np.zeros((80,80),dtype=np.uint8)
        m[10:30,5:8] = 10
        m[10:30,45:48] = 10
        m[50:70,5:8] = 10
        t = objects_from_mask(m,10,4)
        a = direction_audit(t,pd.DataFrame([dict(z1=0,z2=1),dict(z1=0,z2=2)]))
        self.assertTrue(a.direction_pass.iloc[0])
        self.assertFalse(a.direction_pass.iloc[1])

    def test_empty(self):
        self.assertTrue(objects_from_mask(np.zeros((20,20)),10,4).empty)

    def test_rows_columns(self):
        m = np.zeros((40,80),dtype=np.uint8)
        m[10:30,5:8] = 10
        m[10:30,45:48] = 10
        m[1:3,1:3] = 11
        t = objects_from_mask(m,10,4)
        self.assertEqual(len(t),2)
        np.testing.assert_allclose(t.x,[19.5,19.5])
        np.testing.assert_allclose(t.y,[6,46])
        self.assertAlmostEqual(np.linalg.norm(t.iloc[0][['x','y']].astype(float)-t.iloc[1][['x','y']].astype(float)),40)

    def test_rgb_rejected(self):
        with self.assertRaises(ValueError):
            objects_from_mask(np.zeros((20,20,3)),10,4)

    def test_border(self):
        m = np.zeros((20,20))
        m[:10,:2] = 10
        self.assertTrue(objects_from_mask(m,10,4).touches_border.iloc[0])


if __name__ == '__main__':
    unittest.main()
