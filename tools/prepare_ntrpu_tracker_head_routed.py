"""Retarget the mapped public bench to the untouched cached-head netlist."""
from pathlib import Path
import hashlib
import json
import re

text=Path(__file__).with_name('prepare_ntrpu_peer_regions_routed.py').read_text()
text=text.replace('ntrpu_peer_regions_physical','ntrpu_tracker_head_physical')
text=text.replace('REGIONS_ROUTED_EXPORTED','TRACKER_HEAD_ROUTED_EXPORTED')
namespace=dict(__name__='head_routed_prepare',__file__=__file__)
exec(compile(text,__file__,'exec'),namespace)
_public_bench=namespace['public_bench']
_prepare=namespace['prepare']
FIELDS=['generation','first.logical_base','first.physical_base','first.bytes',
        'second.logical_base','second.physical_base','second.bytes','valid']
PORTS={'cfg_map'+''.join('['+part+']' for part in field.split('.')):'cfg_map.'+field for field in FIELDS}


def public_bench(root):
    bench=_public_bench(root);old='ntrpu_tracker_head_physical dut(.*,'
    assert bench.count(old)==1
    connections=',\n        '.join('.\\'+port+' ('+signal+')' for port,signal in PORTS.items())
    return bench.replace(old,'ntrpu_tracker_head_physical dut(\n        '+connections+',\n        .*,')


namespace['public_bench']=public_bench


def prepare(root,out,netlist):
    text=netlist.read_text().split('module ntrpu_tracker_head_physical',1)[1]
    header=text.split(');',1)[0]
    assert set(re.findall(r'\\(cfg_map[^\s,]+)',header))==set(PORTS)
    widths={port:int(hi)+1 for hi,port in re.findall(r'input \[(\d+):0\]\\(cfg_map[^\s;]+)\s*;',text.split('endmodule',1)[0])}
    assert widths=={port:2 if port=='cfg_map[valid]' else 64 for port in PORTS}
    meta=_prepare(root,out,netlist)
    meta.update(public_connection_map=PORTS,flattened_map_port_widths=widths)
    assert meta['generated_bench_sha256']==hashlib.sha256((out/'tb_ntrpu_peer_shared.sv').read_bytes()).hexdigest()
    (out/'COMPARISON.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8',newline='\n')
    return meta
