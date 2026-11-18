import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from utils.jijie.scale_ocr import ScaleOCR,parse_label
from utils.jijie.scale_bar import load_scale_calibration,lookup_scale_calibration,resolve_um_per_pixel


class ScaleTests(unittest.TestCase):
    def test_label(self):
        self.assertEqual(parse_label([[None,'2 \u03bcm',.99]])[0],2)
        self.assertEqual(parse_label([[None,'0.5 um',.99]])[0],.5)
        self.assertIsNone(parse_label([[None,'2 pm',.99]]))
        self.assertIsNone(parse_label([[None,'2 um',.4]]))
        self.assertIsNone(parse_label([[None,'1 um 2 um',.99]]))

    def test_configured_unit(self):
        self.assertEqual(parse_label([[None,'2 pm',.99]],numeric_only=True)[0],2)
        self.assertEqual(parse_label([[None,'0.5',.99]],numeric_only=True)[0],.5)
        self.assertIsNone(parse_label([[None,'2 3',.99]],numeric_only=True))
        self.assertIsNone(parse_label([[None,'pm',.99]],numeric_only=True))
        self.assertIsNone(parse_label([[None,'2',.5]],numeric_only=True))

    def test_consensus_and_failure(self):
        im=np.full((400,400),255,dtype=np.uint8)
        im[330:340,220:320]=0
        calls=[]
        def engine(image,**kwargs):
            calls.append(1)
            return [[None,'2 um',.99]],None
        factor,info=ScaleOCR(engine).resolve(im)
        self.assertEqual(len(calls),2)
        self.assertAlmostEqual(factor,.02)
        self.assertAlmostEqual(100*factor**2,.04)
        self.assertEqual(info['status'],'auto_consensus')
        factor,info=ScaleOCR(lambda *a,**k:(None,None)).resolve(im)
        self.assertIsNone(factor)
        self.assertEqual(info['status'],'needs_review')

    def test_csv_is_image_local_and_status_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'scale.csv'
            p.write_text('sample_id,um_per_pixel,status\ncase__a,0.01,auto_consensus\ncase__b,0.02,needs_review\ncase__c,nan,auto_consensus\n')
            cal=load_scale_calibration(p)
            self.assertEqual(lookup_scale_calibration(cal,'case__a')[0]['um_per_pixel'],.01)
            self.assertIsNone(lookup_scale_calibration(cal,'case__other')[0])
            self.assertIsNone(lookup_scale_calibration(cal,'case__b')[0])
            self.assertIsNone(lookup_scale_calibration(cal,'case__c')[0])


if __name__=='__main__':unittest.main()
