"""Build physical and routed-comparison fixtures for the validated compact RTL."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_ntrpu_candidate_dispatch import ports
from prepare_ntrpu_registered_dispatch import one
from prepare_ntrpu_transport_study import DIRECT, GENERATED, physical, comparison

ROOT = Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--netlist',type=Path);a=ap.parse_args();out=a.output.resolve()
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/prepare_ntrpu_transport_overlap.py'),str(out),'--variant','compact'],check=True)
    evidence=out.parent.parent/'rtl_evidence'
    prior=json.loads((evidence/'result.json').read_text())
    assert prior['variant']=='compact' and prior['status']=='EXPORTED_PENDING_REVIEW'
    assert prior['sources']==prior['sources_after']
    validated={n:digest(evidence/(n+'.sv')) for n in GENERATED}
    assert all(digest(out/(n+'.sv'))==h for n,h in validated.items()),'Physical logic differs from validated compact RTL'
    signals=ports((ROOT/'fpga/rtl/ntrpu_registered_dispatch.sv').read_text())
    (out/'ntrpu_transport_physical.sv').write_text(physical(signals))
    if a.netlist:
        net=a.netlist.read_text();top=net.split('module ntrpu_transport_physical\n',1)[1].split('\n  wire',1)[0]
        assert re.findall(r'^  (input|output) (\[[^\]]+\])?\s*(\w+);',top,re.M)==signals,'Routed ports changed'
        sources={n:((ROOT/'fpga/rtl' if n in DIRECT else out)/(n+'.sv')).read_text() for n in DIRECT+GENERATED}
        modules={m for t in sources.values() for m in re.findall(r'^module (\w+)',t,re.M)}
        mapping={n:'ref_'+n for n in modules};inverse={v:k for k,v in mapping.items()}
        assert not set(mapping.values()) & set(re.findall(r'^module (\w+)',net,re.M))
        regex=re.compile(r'\b('+'|'.join(sorted(mapping))+r')\b')
        reverse=re.compile(r'\b('+'|'.join(sorted(inverse))+r')\b')
        for n,t in sources.items():
            renamed=regex.sub(lambda m:mapping[m[0]],t)
            assert reverse.sub(lambda m:inverse[m[0]],renamed)==t
            (out/('ref_'+n+'.sv')).write_text(renamed)
        original=(out/'tb_ntrpu_candidate_transport.sv').read_text()
        (out/'tb_ntrpu_transport_netlist.sv').write_text(comparison(original,signals))
        extra=(out/'tb_ntrpu_transport_overlap_extra.sv').read_text()
        extra=one(extra,'module tb_ntrpu_transport_overlap_extra;','module tb_ntrpu_candidate_transport;')
        extra=one(comparison(extra,signals),'module tb_ntrpu_transport_netlist;','module tb_ntrpu_overlap_netlist_extra;')
        (out/'tb_ntrpu_overlap_netlist_extra.sv').write_text(extra)
        (out/'COMPARISON.json').write_text(json.dumps(dict(netlist_sha256=digest(a.netlist),latency_edges=0,
            startup_ns=150,sdf=False,force=False,reversible_renaming=True,ports=signals,extra_cases=3),indent=2)+'\n')
    (out/'COMPACT_PHYSICAL.json').write_text(json.dumps(dict(variant='compact',validated_generated_sha256=validated,
        baseline_result_sha256=digest(evidence/'result.json'),clock_period_ns=5,interface_registers_added=0),indent=2)+'\n')


if __name__=='__main__':main()
