"""Negative controls using copies of actual OOC report text, without modifying evidence."""
import argparse
import copy
import csv
import io
import json
from pathlib import Path
import re

from review_guard_cdc import load, review, rows


def replace_row(text, key, value, field, replacement):
    data=rows(text)
    selected=[row for row in data if row[key]==value]
    if not selected: raise ValueError('Mutation target missing')
    selected[0][field]=replacement
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=data[0].keys(),delimiter='\t',lineterminator='\n')
    writer.writeheader();writer.writerows(data)
    return out.getvalue()


def check(baseline, scoped):
    result=review(baseline,scoped)
    mutations={
        'missing_async_reg': ('cells.tsv', lambda t:replace_row(t,'cell','cold_a_reg[0]','async_reg','0')),
        'wrong_clock': ('pins.tsv', lambda t:replace_row(t,'pin','echo_a_reg[0]/C','clocks','mgmt_clk')),
        'broken_chain': ('pins.tsv', lambda t:replace_row(t,'pin','echo_a_reg[1]/D','drivers','fault_reg/Q')),
        'unexpected_reset_driver': ('pins.tsv', lambda t:replace_row(t,'pin','cold_a[2]_i_1/I0','input_ports','perst_n')),
        'hidden_cdc10': ('cdc.rpt', lambda t:re.sub(r'^CDC-10\s+Critical.*\n','',t,flags=re.M)),
        'broad_clock_exception': ('effective.xdc', lambda t:t+'\nset_clock_groups -asynchronous -group mgmt_clk -group axi_clk\n'),
        'relaxed_data_bound': ('effective.xdc', lambda t:t.replace('}] 4.000','}] 400.000')),
        'changed_internal_timing': ('paths.tsv', lambda t:replace_row(t,'endpoint','cold_a_reg[1]/D','slack','99.000')),
        'hidden_external_gap': ('check_timing.rpt', lambda t:t.replace('no_input_delay (5)','no_input_delay (0)')),
        'waiver_added': ('waivers.rpt', lambda t:t.replace('No waivers found.','One CDC waiver found.')),
        'ignored_exception': ('exceptions_ignored.rpt', lambda t:t.replace('No ignored timing exceptions found.','One ignored timing exception found.')),
    }
    rejected={}
    for name,(file,mutate) in mutations.items():
        changed=copy.copy(scoped);changed[file]=mutate(scoped[file])
        if changed[file]==scoped[file]:raise ValueError('No-op mutation: '+name)
        try:review(baseline,changed)
        except ValueError as exc:rejected[name]=str(exc)
        else:raise ValueError('Accepted corrupted evidence: '+name)
    return dict(status='CDC_REVIEWER_NEGATIVE_CONTROLS_PASS',native_diagnostic=result,
                rejected_controls=rejected,physical_cdc_approval=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline',type=Path);ap.add_argument('scoped',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();result=check(load(a.baseline),load(a.scoped))
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
