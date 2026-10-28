"""Reject altered public transaction evidence without changing the frozen logs."""
import argparse
import json
from pathlib import Path
import re
from review_hbm_source import review,require

def check(root):
    good=(root/'case-00-seed-0.log').read_text()
    retained=(root/'case-02-seed-0.log').read_text()
    review(good,0,0);review(retained,2,0)
    rejected=[]
    def reject(name,text,case):
        try:review(text,case,0)
        except ValueError:rejected.append(name)
        else:raise ValueError('Altered evidence accepted: '+name)
    reject('missing_verdict',good.replace('HS PASS','HS SILENCED'),0)
    reject('payload_bit',re.sub(r'(HS W[^\n]* data=)[0-9a-f]',lambda m:m[1]+('0' if m[0][-1]!='0' else '1'),good,count=1),0)
    reject('high_address',re.sub(r'(HS CMD[^\n]* addr=)[0-9a-f]+',r'\g<1>0000000000001000',good,count=1),0)
    reject('full_tag',re.sub(r'(HS RESULT[^\n]* tag=)[0-9a-f]+',r'\g<1>00000000000000000000000100000001',good,count=1),0)
    reject('missing_command','\n'.join(l for l in good.splitlines() if not l.startswith('HS CMD txn=1 ')),0)
    reject('lost_retention',retained.replace('owned=1','owned=0'),2)
    reject('operational_reset',retained.replace('cold=1','cold=0'),2)
    reject('missing_final_beat',re.sub(r'(HS W[^\n]* last=)1',r'\g<1>0',good,count=1),0)
    require(len(rejected)==8,'Incomplete negative controls')
    return dict(rejected=rejected)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('evidence',type=Path)
    print(json.dumps(check(ap.parse_args().evidence),indent=2))
