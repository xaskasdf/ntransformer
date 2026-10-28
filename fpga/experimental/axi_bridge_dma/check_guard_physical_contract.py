"""Negative controls for the PCIe pin/CDC evidence decoder; no Vivado required."""
import argparse
import copy
import csv
import json
from pathlib import Path
from build_guard_physical_contract import decode_lanes,properties,cdc_inventory


def check(root):
    out=root/'output';pins=properties(out/'package.tsv')
    rows=list(csv.DictReader((out/'serial_lanes.tsv').open(),delimiter='\t'))
    text=(out/'full_cdc.rpt').read_text()
    decoded=decode_lanes(rows,pins);warnings=cdc_inventory(text)
    controls=[]
    def reject(name,fn):
        try:fn()
        except ValueError:controls.append(name)
        else:raise ValueError('Corrupted evidence accepted: '+name)
    for name,field,value in [('duplicate_port','port',rows[1]['port']),('wrong_direction','direction','OUT'),
                             ('wrong_gt','loc','GTYE4_CHANNEL_X0Y0'),('wrong_polarity','primitive_pin','GTYRXP')]:
        changed=copy.deepcopy(rows);changed[0][field]=value
        reject(name,lambda:decode_lanes(changed,pins))
    reject('missing_lane',lambda:decode_lanes(rows[:-1],pins))
    first=next(r for r in decoded if r['port']==rows[0]['port'])['ball']
    for name,field,value in [('unbonded_pin','IS_BONDED','0'),('wrong_bank','BANK','65'),('broken_pair','DIFF_PAIR_PIN','BF41')]:
        changed=copy.deepcopy(pins);changed[first][field]=value
        reject(name,lambda:decode_lanes(rows,changed))
    reject('critical_cdc',lambda:cdc_inventory(text.replace('Warning','Critical',1)))
    reject('foreign_warning_source',lambda:cdc_inventory(text.replace('testclk_cnt_reg[0]/C','foreign_source_reg/C',1)))
    lines=text.splitlines();index=next(i for i,s in enumerate(lines) if 'CDC-15  Warning ' in s and 'testclk_cnt_reg' in s)
    reject('missing_warning',lambda:cdc_inventory('\n'.join(lines[:index]+lines[index+1:])))
    return dict(serial_ports=len(decoded),warning_paths=len(warnings),rejected=controls)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('preflight',type=Path)
    print(json.dumps(check(ap.parse_args().preflight),indent=2))
