"""Reject corrupted copies of actual V2 synthesis evidence; leave originals intact."""
import argparse
import json
from pathlib import Path
import re
from check_guard_cdc_reviewer import replace_row
from review_guard_v2_cdc import load, review


def check(baseline,scoped):
    positive=review(baseline,scoped)
    mutations={
        'missing_async_reg':('cells.tsv',lambda t:replace_row(t,'cell','armed_a_reg[0]','async_reg','0')),
        'wrong_clock':('pins.tsv',lambda t:replace_row(t,'pin','armed_a_reg[0]/C','clocks','mgmt_clk')),
        'broken_chain':('pins.tsv',lambda t:replace_row(t,'pin','availability[1].release_a_reg[1]/D','drivers','fault_reg/Q')),
        'combined_async_inputs':('cells.tsv',lambda t:replace_row(t,'cell','availability[0].release_a[2]_i_1','ref','LUT2')),
        'wrong_input_source':('pins.tsv',lambda t:replace_row(t,'pin','availability[0].release_m[2]_i_1/I0','input_ports','perst_n')),
        'reset_from_early_stage':('pins.tsv',lambda t:replace_row(t,'pin','armed_a[2]_i_1/I0','drivers','cold_a_reg[1]/Q')),
        'critical_cdc':('cdc.rpt',lambda t:t+'\nCDC-10  Critical  1  Combinational logic\n'),
        'missing_cdc_detail':('cdc.rpt',lambda t:re.sub(r'^.*CDC-9.*availability\[2\].*\n','',t,flags=re.M)),
        'broad_clock_cut':('effective.xdc',lambda t:t+'\nset_clock_groups -asynchronous -group mgmt_clk -group axi_clk\n'),
        'relaxed_bound':('effective.xdc',lambda t:t.replace('}] 4.000','}] 400.000')),
        'missing_control_exception':('effective.xdc',lambda t:re.sub(r'^set_false_path.*availability\[2\].*\n','',t,flags=re.M)),
        'changed_same_clock_path':('all_paths.tsv',lambda t:replace_row(t,'endpoint','cold_a_reg[1]/D','slack','99.000')),
        'negative_interstage_hold':('stage_paths.tsv',lambda t:replace_row(t,'endpoint','cold_a_reg[1]/D','after_slack','-0.001')),
        'hidden_io_gap':('check_timing.rpt',lambda t:t.replace('no_input_delay (5)','no_input_delay (0)')),
        'waiver_added':('waivers.rpt',lambda t:t.replace('No waivers found.','One waiver found.')),
        'ignored_constraint':('exceptions_ignored.rpt',lambda t:t.replace('No ignored timing exceptions found.','One ignored exception.')),
    }
    rejected={}
    for name,(file,mutate) in mutations.items():
        changed=dict(scoped);changed[file]=mutate(scoped[file])
        if changed[file]==scoped[file]:raise ValueError('No-op mutation: '+name)
        try:review(baseline,changed)
        except ValueError as error:rejected[name]=str(error)
        else:raise ValueError('Corrupted evidence accepted: '+name)
    return dict(status='GUARD_V2_CDC_NEGATIVE_CONTROLS_PASS',positive=positive,rejected_controls=rejected)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('reports',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();result=check(load(a.reports/'baseline'),load(a.reports/'scoped'))
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
