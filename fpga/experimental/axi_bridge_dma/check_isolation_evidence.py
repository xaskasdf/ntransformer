"""Reject deliberately damaged native isolation evidence; keep each damaged input."""
import argparse
import json
from pathlib import Path
import re

from review_isolation_native import review


def replace_word(log,prefix,field,bit):
    line=next(line for line in log.splitlines() if line.startswith(prefix))
    match=re.search(r'\b'+field+r'=([0-9a-fA-F]+)',line)
    value=int(match[1],16)^(1<<bit)
    changed=line[:match.start(1)]+f'{value:0{len(match[1])}x}'+line[match.end(1):]
    return log.replace(line,changed,1)


def check(log_path,vcd_path,output):
    output.mkdir(parents=True,exist_ok=False)
    log=log_path.read_text(errors='replace');vcd=vcd_path.read_text()
    original=review(log,vcd)
    variants=[
        ('host_write_payload',replace_word(log,'BI RQ stage=0 ','data',128),vcd),
        ('completion_tag',replace_word(log,'BI RC stage=0 ','data',64),vcd),
        ('private_readback',replace_word(log,'BI PRIVATE_R stage=0 ','data',0),vcd),
        ('outbound_destination',replace_word(log,'BN CQ case=0 ','data',12),vcd),
        ('false_mapping_hold',log.replace('BI CAMPAIGN_BEGIN stage=1 owned=0 verified=0 release_b=0',
                                        'BI CAMPAIGN_BEGIN stage=1 owned=0 verified=0 release_b=1',1),vcd)]
    allocation=next(line for line in log.splitlines() if line.startswith('BI TAG stage=0 '))
    variants.append(('missing_native_tag_allocation',log.replace(allocation+'\n','',1),vcd))
    scopes=[];code=None
    for line in vcd.splitlines():
        fields=line.split()
        if fields and fields[0]=='$scope':scopes.append(fields[2])
        elif fields and fields[0]=='$upscope':scopes.pop()
        elif fields and fields[0]=='$var' and '/'.join(scopes+[fields[4]])=='board/AXI_PCIE_EP/m_axi_wdata':code=fields[3]
        if '$enddefinitions' in line:break
    if code is None:raise ValueError('VCD incoming write data signal required')
    rows=vcd.splitlines();changed=False
    for i,line in enumerate(rows):
        if line.startswith('b') and line.endswith(' '+code):
            bits,identifier=line.split()
            if not re.search('[xzXZ]',bits) and int(bits[1:],2)!=0:
                rows[i]='b'+format(int(bits[1:],2)^(1<<64),'b')+' '+identifier
                changed=True;break
    if not changed:raise ValueError('Nonzero native host write payload required')
    variants.append(('vcd_incoming_payload',log,'\n'.join(rows)+'\n'))
    results=[]
    for name,bad_log,bad_vcd in variants:
        if (bad_log,bad_vcd)==(log,vcd):raise ValueError('Mutation did not apply: '+name)
        target=output/name;target.mkdir()
        (target/'simulate.log').write_text(bad_log,encoding='utf-8')
        (target/'native.vcd').write_text(bad_vcd,encoding='utf-8')
        try:review(bad_log,bad_vcd)
        except ValueError as error:results.append(dict(name=name,rejected=True,reason=str(error)))
        else:raise ValueError('Mutation escaped: '+name)
    (output/'RESULT.json').write_text(json.dumps(dict(baseline=original['status'],controls=results),indent=2)+'\n')
    return results


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log',type=Path);ap.add_argument('vcd',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();print(json.dumps(check(a.log,a.vcd,a.output)))
