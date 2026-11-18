"""Inspect pickle GLOBAL names without executing author checkpoint deserialization."""
import argparse
from pathlib import Path
import pickletools
import zipfile
import json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('path',type=Path);a=ap.parse_args()
    with zipfile.ZipFile(a.path) as z:
        name=next(n for n in z.namelist() if n.endswith('/data.pkl'))
        names=sorted({arg for op,arg,_ in pickletools.genops(z.read(name)) if op.name=='GLOBAL'})
    print(json.dumps(names,indent=2))


if __name__=='__main__':main()
