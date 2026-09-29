"""Rebuild the original input workbook from official public-use ASCII files."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data2023',type=Path,required=True)
    parser.add_argument('--data2021',type=Path,required=True)
    a=parser.parse_args()
    expected={2023:'0a83cba11dc8a0cdf0043aacf0b80478f86cf93a91a8ecb81a276518e486241a',2021:'0a8605d7843fba9cb50e7c0c4094089b0ce2793a9988506971fdae3d9f2a04ae'}
    records=[]
    for year,path in [(2023,a.data2023),(2021,a.data2021)]:
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=expected[year]:raise ValueError(f'{year} ASCII SHA-256 mismatch')
        records.append({'year':year,'filename':path.name,'sha256':sha})
    ns=runpy.run_path(str(ROOT/'revision_r1/scripts/verify_2023_ascii_source.py'))
    df,widths=ns['parse_ascii'](a.data2023)
    lines=a.data2023.read_text(encoding='latin-1').splitlines()
    for q in range(27,31):df[f'QN{q}']=[ns['number'](ns['field'](line,177+q,177+q),'int') for line in lines]
    dest=ROOT/'data/raw/YRBS_2023_MH_subset.xlsx'
    dest.parent.mkdir(parents=True,exist_ok=True)
    df.to_excel(dest,index=False)
    external=ROOT/'revision_r1/data/external'/a.data2021.name
    external.parent.mkdir(parents=True,exist_ok=True)
    if a.data2021.resolve()!=external.resolve():shutil.copy2(a.data2021,external)
    out=ROOT/'revision_r1/outputs/source_hashes.json'
    out.write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Source hashes verified; 2023 input workbook reconstructed from ASCII.')

if __name__=='__main__':main()
