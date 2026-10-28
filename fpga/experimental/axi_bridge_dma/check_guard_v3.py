"""Negative evidence controls for the shared-reset topology and CDC gate."""
import argparse
import copy
import json
from pathlib import Path
import re
from review_guard_v3 import cdc, constraints, topology, alias_sources
from review_guard_integration import require, rows


def check(native,rtl):
    out=native/'output/optimized';c=(out/'cdc.rpt').read_text();x=(out/'effective.xdc').read_text()
    rejected=[]
    def reject(name,fn):
        try:fn()
        except ValueError:rejected.append(name)
        else:raise ValueError('Corrupt evidence accepted: '+name)
    cdc(c);constraints(x)
    reject('critical_cdc',lambda:cdc(c.replace('Info','Critical',1)))
    reject('missing_cdc',lambda:cdc('\n'.join(l for l in c.splitlines() if not l.startswith('CDC-9'))))
    reject('duplicate_source_cdc',lambda:cdc(c+'\nCDC-11  Critical  2  Duplicate reset sources\n'))
    reject('unbounded_first_stage',lambda:constraints(x.replace('}] 4.000','}] 400.000',1)))
    reject('whole_clock_cut',lambda:constraints(x+'\nset_clock_groups -asynchronous -group mgmt_clk -group axi_aclk_out\n'))
    reject('interstage_cut',lambda:constraints(x+'\nset_false_path -from [get_pins -filter {NAME == "guarded/guard/cold_a_reg[0]/C"}] -to [get_pins -filter {NAME == "guarded/guard/cold_a_reg[1]/D"}]\n'))
    reject('lost_reset_exception',lambda:constraints('\n'.join(l for l in x.splitlines() if not ('set_false_path' in l and 'user_reset_reg/C' in l))))
    actual=rows((out/'native_reset_alias.tsv').read_text());reference=rows((out.parent/'reference_alias.tsv').read_text())
    alias_sources(actual,reference)
    changed=copy.deepcopy(reference);changed[1]['driver']='different_reset/Q'
    reject('distinct_reference_resets',lambda:alias_sources(actual,changed))
    reject('missing_reference_alias',lambda:alias_sources(actual,reference[:1]))
    changed_actual=copy.deepcopy(actual);changed_actual[0]['driver']='inverted_reset/O'
    reject('changed_reset_polarity_driver',lambda:alias_sources(changed_actual,reference))
    log=(rtl/'topology-0-1.log').read_text();vcd=(rtl/'topology-0-1/guard.vcd').read_text()
    reject('lost_wiring_rejection',lambda:topology(log.replace('V3_SHARED_RESET_CONTRACT','silenced'),vcd,1,(2,5,1)))
    log=(rtl/'topology-0-0.log').read_text();vcd=(rtl/'topology-0-0/guard.vcd').read_text()
    reject('lost_staggered_verdict',lambda:topology(log.replace('V3_TOPOLOGY_PASS','silenced'),vcd,0,(2,5,1)))
    return dict(rejected=rejected)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('native',type=Path);ap.add_argument('rtl',type=Path)
    a=ap.parse_args();print(json.dumps(check(a.native,a.rtl),indent=2))
