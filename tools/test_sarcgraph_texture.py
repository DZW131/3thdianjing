import unittest
import numpy as np
import pandas as pd
from sarcgraph_texture import texture_fields, texture_audit


class TextureTests(unittest.TestCase):
    def test_stripes_and_blank(self):
        rr,cc=np.mgrid[:160,:160]
        objects=pd.DataFrame([dict(x=30.,y=80.),dict(x=130.,y=80.)])
        pairs=pd.DataFrame([dict(z1=0,z2=1)])
        # Vertical texture has little variation along rows.
        supported=texture_audit(objects,pairs,texture_fields(np.cos(cc/3)))
        conflict=texture_audit(objects,pairs,texture_fields(np.cos(rr/3)))
        blank=texture_audit(objects,pairs,texture_fields(np.zeros((160,160))))
        self.assertEqual(supported.texture_status.iloc[0],'texture_supported')
        self.assertEqual(conflict.texture_status.iloc[0],'texture_conflict')
        self.assertEqual(blank.texture_status.iloc[0],'uncertain_texture')

    def test_rotation(self):
        rr,cc=np.mgrid[:160,:160]
        objects=pd.DataFrame([dict(x=80.,y=30.),dict(x=80.,y=130.)])
        out=texture_audit(objects,pd.DataFrame([dict(z1=0,z2=1)]),texture_fields(np.cos(rr/3)))
        self.assertEqual(out.texture_status.iloc[0],'texture_supported')


if __name__=='__main__':
    unittest.main()
