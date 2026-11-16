"""Fetch a bounded public paired sample without modifying project data."""
import argparse,re,requests,xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=3);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    s=requests.Session();base='https://owncloud.cesnet.cz';token='KLP6jyZzAU1OJdV'
    page=s.get(base+'/index.php/s/'+token,timeout=30);page.raise_for_status()
    rt=re.search(r'data-requesttoken="([^"]+)"',page.text)[1]
    folder='CRISTAE_images/CRISTAE_manual_annotation'
    r=s.request('PROPFIND',base+'/public.php/webdav/'+folder,auth=(token,''),headers={'Depth':'1','requesttoken':rt,'X-Requested-With':'XMLHttpRequest'},timeout=30);r.raise_for_status()
    (a.output/'listing.xml').write_text(r.text,encoding='utf8')
    names=[]
    for node in ET.fromstring(r.content).findall('{DAV:}response'):
        if node.find('.//{DAV:}getcontentlength') is not None:names.append(unquote(node.find('{DAV:}href').text.split('/')[-1]))
    print('names',names,flush=True)
    for name in sorted(names)[:a.count]:
        for sub in ['CRISTAE_manual_annotation','CRISTAE_orig_images']:
            filename=Path(name).with_suffix('.tif').name if sub=='CRISTAE_orig_images' else name
            dest=a.output/sub/filename;dest.parent.mkdir(exist_ok=True)
            if dest.exists():continue
            r=s.get(base+'/index.php/s/'+token+'/download',params={'path':'/CRISTAE_images/'+sub,'files':filename},timeout=90);r.raise_for_status();dest.write_bytes(r.content)
            print(dest,len(r.content),flush=True)

if __name__=='__main__':main()
