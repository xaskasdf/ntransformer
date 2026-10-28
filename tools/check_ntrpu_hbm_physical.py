"""Independently review physical checkpoints and a bounded native boot window."""
import argparse, csv, hashlib, json, re
from pathlib import Path
from check_ntrpu_hbm_audit import base, cdc, metrics, need, report_checks

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def physical(out, r):
    need(r['generated']==r['generated_after'], 'Generated sources changed')
    need(r['input_sha256']==r['input_after'], 'Input checkpoint changed')
    suite=out/'suite';stage=r['stage'];text=(out/'console.log').read_text()
    checkpoint='suite/physical_'+stage+'.dcp'
    label_error=stage=='synth' and 'NTRPU_HBM_PHYSICAL_COMPLETE ip' in text
    if label_error:
        need('Design State      : Synthesized' in (suite/'timing.rpt').read_text(),'Mislabeled checkpoint state')
        checkpoint='suite/physical_ip.dcp'
    else:need('NTRPU_HBM_PHYSICAL_COMPLETE '+stage in text, 'Missing physical completion')
    need(not re.search(r'^\s*ERROR:',text,re.M), 'Tool error')
    critical=re.findall(r'^\s*CRITICAL WARNING: (.+)',text,re.M)
    # A completed, failing timing run remains useful evidence, never a closure pass.
    need(all(x.startswith('[Timing 38-282]') for x in critical), 'Unreviewed critical diagnostic')
    m=metrics(suite/'metrics.txt')
    need(m['LATCHES']==m['BLACKBOXES']=='0','Incomplete netlist')
    need(float(m['PERIOD_transport_clk'])==5 and float(m['PERIOD_memory_clk'])==2.222,'Clock targets changed')
    buffers=list(csv.reader((suite/'buffers.tsv').read_text().splitlines(),delimiter='\t'))
    clocks=[b for b in buffers if b[0] in ('source_clock','memory_clock')]
    control=[b for b in buffers if b not in clocks]
    need(len(clocks)==2,'Clock buffers missing')
    expected={'source_clock':'transport_clk','memory_clock':'memory_clk'}
    need(all(b[1]=='BUFGCE' and b[4]==expected[b[0]] for b in clocks),'Clock buffer connectivity')
    need(all(b[4]=='' for b in control),'Unexpected added clock buffer')
    if stage!='synth':need(all(b[2].startswith('BUFG') for b in buffers) and len({b[2] for b in buffers})==len(buffers),'Clock/control buffer placement')
    counts=cdc(suite/'cdc.rpt')
    expected_cdc={'CDC-3':19 if stage=='synth' else 15,'CDC-9':12}
    need(counts==expected_cdc,'Unreviewed CDC topology')
    regs=[x.split('\t') for x in (suite/'async_registers.tsv').read_text().splitlines()]
    startup=[x for x in regs if re.search(r'(^|/)startup/',x[0])]
    need(len(startup)==21 and all(x[2]=='1' for x in startup),'Startup synchronizers changed')
    need(len(regs)==(313 if stage=='synth' else 291),'Unreviewed synchronizer inventory')
    cdc_text=(suite/'cdc.rpt').read_text()
    for endpoint in ['fault_meta_reg[0]/D','fault_meta_reg[1]/D','ready_meta_reg[2]/D']:
        need(any(endpoint in line and 'CDC-3' in line and 'Max Delay Datapath Only' in line for line in cdc_text.splitlines()),'Custom status crossing changed')
    checks=report_checks(suite/'check_timing.rpt')
    need(all(checks[k]==0 for k in ('no_clock','unconstrained_internal_endpoints','loops','latch_loops','multiple_clock')),'Internal timing coverage')
    drc=(suite/'drc.rpt').read_text()
    drc_rows=re.findall(r'^\|\s*([A-Z]+-\d+)\s*\|\s*(Critical Warning|Warning|Error)\s*\|[^|]+\|\s*(\d+)\s*\|',drc,re.M)
    allowed={'PDCN-1569':('Warning',12)} if stage!='synth' else {}
    if stage=='place':allowed['RTSTAT-13']=('Critical Warning',1)
    if stage=='route':allowed['RTSTAT-10']=('Warning',1)
    need(all(allowed.get(rule)==(level,int(n)) for rule,level,n in drc_rows),'Unreviewed DRC violation')
    drc_count=int(re.search(r'Checks found:\s+(\d+)',drc)[1])
    need(sum(int(n) for _,_,n in drc_rows)==drc_count,'DRC inventory incomplete')
    interaction=(suite/'clock_interaction.rpt').read_text()
    cross=next(l for l in interaction.splitlines() if re.match(r'memory_clk\s+transport_clk\s',l))
    need('unsafe' not in cross and 'Max Delay Datapath Only' in cross,'Status constraints lost')
    external=(suite/'external_status.rpt').read_text()
    need('MaxDelay Path' in external and all(f'init_complete_async[{i}]' in external for i in range(2)),'External status constraint lost')
    line=next(l for l in (suite/'utilization.rpt').read_text().splitlines() if re.match(r'\| ntrpu_hbm_physical\s+\|',l))
    fields=[x.strip() for x in line.split('|')[1:-1]]
    resources=dict(zip(('lut','logic_lut','lutram','srl','ff','ramb36','ramb18','uram','dsp'),map(int,fields[2:])))
    resources['bram36_equivalent']=resources['ramb36']+resources['ramb18']/2
    slacks={k:float(v) for k,v in m.items() if k.startswith(('setup_','hold_'))}
    memory=(suite/'memory_clk_setup.rpt').read_text()
    route=None
    if stage=='route':
        text=(suite/'route_status.rpt').read_text()
        route={k:int(v) for k,v in re.findall(r'# of (.+?)\.+\s*:\s*(\d+)',text)}
        need(route,'Missing route counts')
        need(route['routable nets']==route['fully routed nets']>0 and route['nets with routing errors']==0,'Routing errors')
        need(all(v==0 for k,v in route.items() if any(x in k.lower() for x in ('unrouted','partially','conflict','routing errors'))),'Incomplete routing')
    return dict(status='PHYSICAL_'+stage.upper()+'_REVIEWED_TIMING_'+('NONNEGATIVE' if all(v>=0 for v in slacks.values()) else 'OPEN'),
                evidence_validated=True,stage=stage,clock_slack_ns=slacks,resources=resources,
                buffers=buffers,control_buffers=control,cdc=counts,startup_async_registers=len(startup),total_async_registers=len(regs),coverage=checks,
                route_counts=route,critical_warnings=critical,drc_findings=drc_rows,drc_clean=drc_count==0,
                worst_memory_path=dict(source=re.search(r'Source:\s+(.+)',memory)[1].strip(),destination=re.search(r'Destination:\s+(.+)',memory)[1].strip()),
                checkpoint_sha256=sha(out/checkpoint),checkpoint_relative_path=checkpoint,
                stage_label_error=label_error,
                placed=stage in ('place','route'),routed=stage=='route',
                qualified_board_timing=False,native_hbm_in_design=False,external_io_budget_defined=False,
                post_optimization_functional_simulation=False)

def boot(out,r):
    need(r['donor_unchanged'],'Snapshot changed')
    need(sha(out/'evidence/run.tcl')==r['generated_tcl_sha256'],'Observation Tcl changed')
    text=(out/'console.log').read_text()
    need(not re.search(r'^\s*(Fatal:|ERROR:)',text,re.M),'Simulator error')
    rows=list(csv.DictReader((out/'evidence/progress.tsv').read_text().splitlines(),delimiter='\t'))
    def ns(t):
        m=re.fullmatch(r'([\d.]+)(fs|ps|ns|us|ms|s)',t);need(m,'Time format')
        return float(m[1])*{'fs':1e-6,'ps':1e-3,'ns':1,'us':1000,'ms':1e6,'s':1e9}[m[2]]
    times=[ns(x['simulation_time']) for x in rows]
    need(times[0]==0 and all(b>=a and b-a<=1000.001 for a,b in zip(times,times[1:])),'Sampling cadence')
    need(times[-1]>21000,'No extension beyond earlier observation')
    active=[x for x in rows if ns(x['simulation_time'])>=5000]
    need(all(x['mem_rst_n']==x['apb_rst_n']=='1' for x in active),'Reset release')
    need(all(x['valid_window_fail_'+str(s)]=='0' for x in active for s in (0,1)),'Valid window failure sampled')
    complete='BOOT PASS ' in text
    return dict(status='NATIVE_HBM_BOOT_PASS' if complete else 'NATIVE_HBM_EXTENDED_PROGRESS_ONLY',
                evidence_validated=True,native_boot_validated=complete,native_transport_traffic=False,
                simulated_ns=times[-1],samples=len(rows),last_flags=[rows[-1]['apb_complete_'+str(s)] for s in (0,1)],
                rom_addresses={str(s):list(dict.fromkeys(x['xpm_addra_'+str(s)] for x in active)) for s in (0,1)},
                final_sample=rows[-1],sampling_is_diagnostic=True)

def audit(out,r):
    need(r['input_sha256']==r['input_after']==r['donor_after'],'Routed checkpoint changed')
    prior=json.loads((out/'PRIOR_REVIEW.json').read_text())
    need(prior['routed'] and prior['checkpoint_sha256']==r['input_sha256'],'Wrong routed checkpoint')
    text=(out/'console.log').read_text()
    need('NTRPU_HBM_ROUTE_AUDIT_COMPLETE' in text,'Missing audit completion')
    need(not re.search(r'^\s*(ERROR:|CRITICAL WARNING:)',text,re.M),'Audit tool diagnostic')
    skew=(out/'bus_skew.rpt').read_text()
    rows=[tuple(map(float,row)) for row in re.findall(r'^\s+(?:Slow|Fast)\s+([.0-9]+)\s+([.0-9]+)\s+([-0-9.]+)\s*$',skew,re.M)]
    need(len(rows)==12 and all(req==2.222 and value>=0 and slack>=0 and abs(req-value-slack)<0.002 for req,value,slack in rows),'Bus skew requirements not met')
    timing=(out/'timing.rpt').read_text();section=timing.split('Design Timing Summary',1)[1]
    summary=next(line.split() for line in section.splitlines() if re.match(r'^\s+-?\d+\.\d+\s+-?\d+\.\d+\s+\d+\s+\d+',line))
    need(len(summary)==12,'Timing summary shape')
    need(all(float(summary[i])>=0 for i in (0,4,8)) and all(float(summary[i])==0 for i in (1,2,5,6,9,10)),'Timing or pulse-width failure')
    need('All user specified timing constraints are met.' in timing,'Incomplete timing closure within supplied constraints')
    buffers=list(csv.reader((out/'global_buffers.tsv').read_text().splitlines(),delimiter='\t'))
    need(len(buffers)==3 and {row[5] for row in buffers}=={'','memory_clk','transport_clk'},'Global buffer topology')
    route={k:int(v) for k,v in re.findall(r'# of (.+?)\.+\s*:\s*(\d+)',(out/'route_status.rpt').read_text())}
    need(route['fully routed nets']==route['routable nets']==30716 and route['nets with routing errors']==0,'Route audit mismatch')
    return dict(status='ROUTED_INTERNAL_TIMING_AND_BUS_SKEW_PASS',evidence_validated=True,
                skew_constraints=len(rows),skew_requirement_ns=2.222,max_skew_ns=max(x[1] for x in rows),minimum_skew_slack_ns=min(x[2] for x in rows),
                worst_setup_ns=float(summary[0]),worst_hold_ns=float(summary[4]),worst_pulse_width_slack_ns=float(summary[8]),
                setup_endpoints=int(summary[3]),hold_endpoints=int(summary[7]),pulse_width_endpoints=int(summary[11]),
                global_buffers=buffers,checkpoint_sha256=r['input_sha256'],
                qualified_board_timing=False,external_io_budget_defined=False,post_optimization_functional_simulation=False,
                drc_clean=prior['drc_clean'],remaining_drc_findings=prior['drc_findings'])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--review',type=Path,required=True)
    a=ap.parse_args();r=base(a.output)
    if r['experiment']=='hbm_boot_window':review=boot(a.output,r)
    elif r['experiment']=='hbm_route_audit':review=audit(a.output,r)
    else:review=physical(a.output,r)
    review.update(seconds=r['seconds'],checker_sha256=sha(Path(__file__)))
    a.review.write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review))
if __name__=='__main__':main()
