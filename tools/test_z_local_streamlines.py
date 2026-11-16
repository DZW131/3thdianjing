import unittest
import numpy as np
from z_local_streamlines import trace,first_hit

class Tests(unittest.TestCase):
    def segments(self):return np.array([[[10,20],[90,20]],[[10,80],[90,80]],[[10,140],[90,140]]],float)
    def fields(self,c=1):
        a=np.ones((100,180))*c;z=np.zeros_like(a);o=np.ones_like(a);return [(a,z,o)]*2
    def test_first_neighbor_not_farthest(self):
        r=trace([50,20],[0,1],self.fields(),self.segments(),0)
        self.assertEqual(r['status'],'hit');self.assertEqual(r['target'],1);self.assertAlmostEqual(r['length_px'],60)
    def test_no_support(self):
        self.assertEqual(trace([50,20],[0,1],self.fields(0),self.segments(),0)['status'],'unsupported_gap')
    def test_local_positions(self):
        for row in [30,50,70]:self.assertEqual(trace([row,20],[0,1],self.fields(),self.segments(),0)['target'],1)
    def test_ambiguity(self):
        s=self.segments();s[2]=s[1]
        self.assertEqual(first_hit(np.array([50,78]),np.array([50,81]),s,0)[0],'ambiguous_hit')
    def test_departure_avoids_z_texture(self):
        f=self.fields()[0];f[1][:,:32]=1;f[2][:,:32]=0
        self.assertEqual(trace([50,20],[0,1],[f]*2,self.segments(),0)['status'],'sharp_turn')
        self.assertEqual(trace([50,20],[0,1],[f]*2,self.segments(),0,departure_px=12)['target'],1)

if __name__=='__main__':unittest.main()
