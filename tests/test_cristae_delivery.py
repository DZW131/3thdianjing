"""Synthetic checks only; no private images or pretrained weights required."""
import csv
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from benchmark_cristae_states import metrics


class CristaeDeliveryTests(unittest.TestCase):
    def test_macro_excludes_background_and_absent_gt(self):
        cm=np.zeros((14,14),dtype=np.int64)
        cm[0,0]=10000;cm[2,2]=10;cm[2,3]=10
        result=metrics(cm)
        self.assertAlmostEqual(result['macro_present_crista_iou'],.5)
        self.assertAlmostEqual(result['macro_present_crista_dice'],2/3)
        self.assertIsNone(result['per_class'][4]['recall'])

    def test_private_bundle_packaging(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);states=tmp/'states';data=tmp/'data';out=tmp/'out'
            states.mkdir();(data/'SegmentationClass').mkdir(parents=True);(data/'JPEGImages').mkdir()
            m=np.zeros((20,20),dtype='uint8');m[5:10,6:12]=2
            Image.fromarray(m).save(data/'SegmentationClass/synthetic.png')
            Image.new('RGB',(20,20),'white').save(data/'JPEGImages/synthetic.jpg')
            row=dict(experiment='appearance_control',sample_id='synthetic',class_id=2,component_id=1,area=30)
            with (states/'oof_predictions.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
            command=[sys.executable,str(ROOT/'tools/prepare_cristae_baseline_bundle.py'),'--states',str(states),'--cardiac',str(data),'--output',str(out)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            with (out/'objects.csv').open() as handle:
                record=next(csv.DictReader(handle))
            self.assertEqual(json.loads(record['bbox']),[6,5,12,10])
            with tarfile.open(out/'private_bundle.tar.gz') as archive:
                self.assertEqual(set(archive.getnames()),{'baseline_bundle/objects.csv','baseline_bundle/object_000.png','baseline_bundle/synthetic.jpg'})
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)


if __name__=='__main__':unittest.main()
