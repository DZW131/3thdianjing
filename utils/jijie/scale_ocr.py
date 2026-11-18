"""Optional image-local scale calibration. No default physical bar length."""
import math
import re

import numpy as np
from PIL import Image, ImageOps


def parse_label(result, minimum_confidence=.85, numeric_only=False):
    if not result:
        return None
    text=' '.join(str(r[1]) for r in result)
    confidence=min(float(r[2]) for r in result)
    normalized=text.lower().replace('\u03bc','u').replace('\u00b5','u')
    pattern=(r'(?<![\d.])(\d+(?:[.,]\d+)?)(?![\d.])' if numeric_only else
             r'(?<![\d.])(\d+(?:[.,]\d+)?)\s*u\s*m\b')
    values={float(v.replace(',','.')) for v in re.findall(pattern,normalized)}
    if confidence<minimum_confidence or len(values)!=1:
        return None
    value=values.pop()
    return (value,confidence,text) if math.isfinite(value) and .01<=value<=100 else None


class ScaleOCR:
    def __init__(self, engine=None, minimum_confidence=.85, numeric_only=False):
        if engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise RuntimeError('Install requirements-scale-ocr.txt in the inference environment.') from exc
            engine=RapidOCR(intra_op_num_threads=2,inter_op_num_threads=1)
        self.engine=engine
        self.minimum_confidence=minimum_confidence
        self.numeric_only=numeric_only

    def resolve(self,image):
        from .scale_bar import detect_scale_bar_candidates
        if not isinstance(image,Image.Image):
            image=Image.fromarray(np.asarray(image))
        image=image.convert('RGB')
        w,h=image.size
        accepted=[]
        evidence=[]
        for bar in detect_scale_bar_candidates(image.convert('L'))[:3]:
            x0,y0,x1,y1=bar['bbox']
            crop=image.crop((max(0,x0-25),max(0,y0-int(h*.09)),min(w,x1+35),max(1,y0-2)))
            readings=[]
            for variant in (crop,ImageOps.autocontrast(crop.convert('L')).convert('RGB')):
                result,_=self.engine(np.asarray(variant.resize((variant.width*3,variant.height*3))),use_cls=False)
                reading=parse_label(result,self.minimum_confidence,self.numeric_only)
                if reading is None and self.numeric_only:
                    direct,_=self.engine(np.asarray(variant.resize((variant.width*3,variant.height*3))),use_cls=False,use_det=False)
                    normalized=[[None,r[0],r[1]] for r in (direct or [])]
                    reading=parse_label(normalized,self.minimum_confidence,True)
                readings.append(reading)
                evidence.append(' '.join(str(r[1]) for r in (result or [])))
            if all(readings) and readings[0][0]==readings[1][0]:
                accepted.append(dict(width=int(bar['width']),scale_bar_px=int(bar['width']),
                    scale_bar_um=readings[0][0],um_per_pixel=readings[0][0]/bar['width'],
                    ocr_confidence=min(r[1] for r in readings),bbox=list(bar['bbox']),
                    ocr_text=' | '.join(r[2] for r in readings)))
        if len(accepted)==1:
            return accepted[0]['um_per_pixel'],dict(accepted[0],status='auto_consensus',method='scale_ocr',
                unit='um',unit_source='configured' if self.numeric_only else 'ocr')
        return None,dict(status='needs_review',method='scale_ocr',reason='ambiguous_or_unreadable_scale',
            unit='um',unit_source='configured' if self.numeric_only else 'ocr',ocr_text=' | '.join(evidence))
