"""Preserve every original row; insert two idle rows before cold cancellation."""
import hashlib
import json
from pathlib import Path
import sys

BANKS=dict(main='pcie_request_inventory_r7.txt',ownership='pcie_rc_ownership_integration.txt',
    rq='pcie_rq_admission_integration.txt',prefix='pcie_rc_prefix_integration.txt',
    lanes='pcie_rc_lane_integration_rtl.txt',context='pcie_symbolic_value_context.txt')


def drained(lines):
    output=[];insertions=[];previous=None
    for index,line in enumerate(lines):
        row=line.split();assert len(row)==28
        if row[2]=='0' and previous is not None and previous[2]!='0':
            idle=previous.copy()
            # An offered stalled beat dropped for drain is a real protocol fault.
            # Keep this visible; never use inserted rows to edit prior outcomes.
            held=(previous[8]=='1' and int(previous[9],16)==0) or (previous[12]=='1' and previous[13]=='0')
            idle[1]='0';idle[2]='1';idle[8]='0';idle[9]='f';idle[10]='0'*512;idle[11]='0'
            idle[12]='0';idle[13]='1';idle[14]='0'*512;idle[15]='0'
            idle[17]=str(int(previous[17]=='1' or held));idle[18]='0'
            insertions.append(dict(before_original_row=index,first_output_row=len(output),count=2,
                                   expected_observing=int(idle[16]),expected_fault=int(idle[17]),held_drop=held))
            output += [' '.join(idle)]*2
        output.append(line);previous=row
    return output,insertions


if __name__=='__main__':
    root=Path(sys.argv[1]);records={}
    for bank,filename in BANKS.items():
        source=root/filename;lines=source.read_text().splitlines();output,insertions=drained(lines)
        # Remove only the recorded insertions and recover the input byte-for-byte.
        injected={i for r in insertions for i in range(r['first_output_row'],r['first_output_row']+2)}
        assert [line for i,line in enumerate(output) if i not in injected]==lines
        target=root/('pcie_symbolic_ced_drained_'+bank+'_r1.txt');text='\n'.join(output)+'\n'
        with target.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
        records[bank]=dict(source=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            path=target.name,sha256=hashlib.sha256(text.encode()).hexdigest(),original_rows=len(lines),
            rows=len(output),scenarios=len({line.split()[0] for line in lines}),insertions=insertions)
    manifest=dict(banks=records,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Every original row retained. Two idle clocks before each noncold-to-cold boundary retire prior origins. Added rows use no counter oracle; independent R7/model/S1 comparisons remain active. Expected cold/fault flags include held-offer discontinuation.')
    with (root/'pcie_symbolic_ced_drained_r1.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps({bank:dict(rows=r['rows'],original_rows=r['original_rows'],inserted=2*len(r['insertions'])) for bank,r in records.items()}))
