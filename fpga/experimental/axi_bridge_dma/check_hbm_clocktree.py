"""Alter copies of evidence to exercise fail-closed physical review checks."""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import shutil
from review_hbm_clocktree import review,sha

def change_field(text,column,value,predicate=lambda row:True):
    reader=csv.DictReader(io.StringIO(text),delimiter='\t');fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        if predicate(row):row[column]=value;break
    else:raise ValueError('No matching negative-control row')
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=fields,delimiter='\t',lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return out.getvalue()

def check(root,out):
    if out.exists():raise ValueError('New negative-control output required')
    baseline=review(root);out.mkdir(parents=True);copy=out/'fixture';shutil.copytree(root,copy)
    original_manifest=(copy/'FILES.json').read_bytes();manifest=json.loads(original_manifest)
    mutations=[
        ('missing_cdc_endpoint','output/audit/paths.tsv',lambda t:'\n'.join(t.splitlines()[:-1])+'\n'),
        ('relaxed_max_delay','output/audit/paths.tsv',lambda t:t.replace('\t4.000\t','\t8.000\t',1) if '\t4.000\t' in t else t.replace('\t4.0\t','\t8.0\t',1)),
        ('wrong_clock_period','output/audit/clocks.tsv',lambda t:t.replace('pcie_clk\t4.0','pcie_clk\t8.0',1)),
        ('missing_clock_buffer','output/audit/clocks.tsv',lambda t:t.replace('clock_probe_pcie_clk','missing_buffer',1)),
        ('missing_sync_stage','output/audit/interstage.tsv',lambda t:'\n'.join(t.splitlines()[:-1])+'\n'),
        ('capture_bypass','output/audit/capture_topology.tsv',lambda t:t.replace('\t1\tRAMD32','\t2\tRAMD32',1)),
        ('missing_reset','output/audit/canonical_resets.tsv',lambda t:'\n'.join(t.splitlines()[:-1])+'\n'),
        ('clock_without_definition','output/audit/check_timing.rpt',lambda t:t.replace('checking no_clock (0)','checking no_clock (1)')),
        ('critical_cdc','output/audit/cdc.rpt',lambda t:t+'\nCDC-1 Critical 1 injected unsynchronized path\n'),
        ('ignored_constraint','output/audit/exceptions_ignored.rpt',lambda t:t.replace('No ignored timing exceptions found.','Injected overridden max delay')),
        ('missing_gray_bound','output/audit/bus_skew.rpt',lambda t:t.replace('2.000','3.000',1)),
    ]
    if baseline['mode']!='preflight':
        mutations.extend([
            ('cdc_delay_violation','output/audit/paths.tsv',lambda t:change_field(t,'slack_ns','-0.001')),
            ('synchronizer_setup_violation','output/audit/interstage.tsv',lambda t:change_field(t,'slack_ns','-0.001',lambda r:r['type']=='max')),
            ('synchronizer_hold_violation','output/audit/interstage.tsv',lambda t:change_field(t,'slack_ns','-0.001',lambda r:r['type']=='min')),
            ('unrouted_clock','output/audit/clocks.tsv',lambda t:change_field(t,'route_status','UNROUTED')),
            ('internal_route_gap','output/audit/route_gaps.tsv',lambda t:t+'unexpected_internal_net\tUNROUTED\t\tunexpected_cell/Q\n'),
        ])
        if baseline['reset_distribution_buffers']:
            mutations.extend([
                ('wrong_reset_buffer_source','output/audit/control_buffers.tsv',lambda t:change_field(t,'input_driver','wrong_register/Q')),
                ('reset_buffer_disabled','output/audit/control_buffers.tsv',lambda t:change_field(t,'ce_ref','GND')),
                ('reset_buffer_inverted','output/audit/control_buffers.tsv',lambda t:change_field(t,'i_inverted','1')),
                ('reset_buffer_synchronized_enable','output/audit/control_buffers.tsv',lambda t:change_field(t,'ce_type','SYNC')),
                ('reset_buffer_clocks_logic','output/audit/control_buffers.tsv',lambda t:change_field(t,'clock_loads','1')),
                ('unproved_physical_reset','output/audit/canonical_resets.tsv',lambda t:change_field(t,'physical_driver','unknown_buffer/O')),
            ])
    records=[]
    for name,path,mutate in mutations:
        p=copy/path;old=p.read_bytes();text=p.read_text();changed=mutate(text)
        if changed==text:raise ValueError('Mutation not applied: '+name)
        p.write_text(changed);modified=dict(manifest);modified[path]=sha(p)
        (copy/'FILES.json').write_text(json.dumps(modified,indent=2)+'\n')
        case=out/name;case.mkdir();(case/p.name).write_bytes(p.read_bytes())
        try:review(copy)
        except ValueError as exc:records.append(dict(name=name,path=path,sha256=sha(p),rejected=True,reason=str(exc)))
        else:raise ValueError('Invalid evidence accepted: '+name)
        finally:p.write_bytes(old);(copy/'FILES.json').write_bytes(original_manifest)
    if review(copy)!=baseline:raise ValueError('Fixture restore failed')
    result=dict(status='HBM_CLOCKTREE_NEGATIVES_REJECTED',count=len(records),checks=records)
    (out/'REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    print(json.dumps(check(a.root,a.output),indent=2))
