"""Review native progress and synthesis/CDC evidence without timing overclaims."""
import argparse,csv,hashlib,json,re,statistics
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,why):
    if not ok:raise ValueError(why)
def metrics(p):return dict(line.split(None,1) for line in p.read_text().splitlines())
def cdc(p):
    text=p.read_text()
    rows=re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s',text,re.M)
    need(rows and all(level=='Info' for _,level,_ in rows),'Non-informational CDC finding')
    return {name:int(count) for name,_,count in rows}
def report_checks(p):return dict((n,int(v)) for n,v in re.findall(r'checking (\w+) \((\d+)\)',p.read_text()))
def base(out):
    record=json.loads((out/'result.json').read_text());files=json.loads((out/'FILES.json').read_text())
    need(all(sha(out/n)==h for n,h in files.items()),'Evidence hashes')
    need(record['sources']==record['sources_after'],'Changed source inputs')
    need(record['status']=='EXPORTED_PENDING_REVIEW','Tool failure/timeout')
    return record

def synthesis(out,r):
    need(r['generated']==r['generated_after'],'Generated logic changed')
    text=(out/'console.log').read_text();need('NTRPU_HBM_CDC_AUDIT_COMPLETE' in text,'Missing audit completion')
    need(not re.search(r'^\s*(ERROR:|CRITICAL WARNING:)',text,re.M),'Unreviewed tool diagnostic')
    suite=out/'suite';m=metrics(suite/'metrics.txt')
    need(m['LATCHES']==m['BLACKBOXES']=='0','Incomplete/nonsequential synthesis')
    need(float(m['PERIOD_transport_clk'])==5 and float(m['PERIOD_memory_clk'])==2.222,'Clock targets')
    counts=cdc(suite/'cdc.rpt');need(counts==cdc(suite/'cdc_without_waivers.rpt'),'Hidden CDC waiver')
    need(counts=={'CDC-3':19,'CDC-9':12},'Unexpected CDC topology')
    registers=[l.split('\t') for l in (suite/'async_registers.tsv').read_text().splitlines()]
    startup=[x for x in registers if x[0].startswith('startup/')]
    need(len(startup)==21 and all(x[2]=='1' for x in startup),'Startup synchronizers missing')
    checks=report_checks(suite/'check_timing.rpt')
    need(all(checks[k]==0 for k in ('no_clock','unconstrained_internal_endpoints','loops','latch_loops','multiple_clock')),'Internal timing coverage')
    need(re.search(r'Checks found:\s+0',(suite/'drc.rpt').read_text()),'DRC violation')
    line=next(l for l in (suite/'utilization.rpt').read_text().splitlines() if re.match(r'\| ntrpu_hbm_transport_async\s+\|',l))
    fields=[x.strip() for x in line.split('|')[1:-1]]
    resources=dict(zip(('lut','logic_lut','lutram','srl','ff','ramb36','ramb18','uram','dsp'),map(int,fields[2:])))
    resources['bram36_equivalent']=resources['ramb36']+resources['ramb18']/2
    path=(suite/'memory_clk_setup.rpt').read_text()
    return dict(status='CDC_STRUCTURE_AUDITED_TIMING_OPEN',cdc=counts,resources=resources,
                clock_slack_ns={k:float(v) for k,v in m.items() if k.startswith(('setup_','hold_'))},
                worst_memory_path=dict(source=re.search(r'Source:\s+(.+)',path)[1].strip(),destination=re.search(r'Destination:\s+(.+)',path)[1].strip()),
                startup_async_registers=len(startup),external_input_delays_missing=checks['no_input_delay'],
                external_output_delays_missing=checks['no_output_delay'],checkpoint_sha256=sha(suite/'async_synth.dcp'),
                virtual_apb_clock_is_assumption=True,placed=False,routed=False,qualified_timing=False)

def constraints(out,r,prior):
    need(prior is not None,'Prior synthesis required')
    old=base(prior);m=metrics(out/'metrics.txt');original=metrics(prior/'suite/metrics.txt')
    need(r['prior_result_sha256']==sha(prior/'result.json'),'Wrong prior result')
    need(r['checkpoint_sha256']==r['input_after']==sha(prior/'suite/async_synth.dcp'),'Changed checkpoint')
    need(all(v==original[k] for k,v in m.items()),'Functional timing altered by status constraints')
    text=(out/'console.log').read_text()
    need('NTRPU_HBM_CDC_CONSTRAINTS_COMPLETE' in text,'Missing constraint completion')
    need(not re.search(r'^\s*(ERROR:|CRITICAL WARNING:)',text,re.M),'Rejected/ignored constraint')
    endpoints=[l.split('\t') for l in (out/'status_endpoints.tsv').read_text().splitlines()]
    expected={f'startup/{kind}_meta_reg[{i}]' for kind,n in [('ready',3),('fault',2)] for i in range(n)}
    need({l[0] for l in endpoints}==expected and all(l[1]=='1' and l[2]==l[0]+'/D' and float(l[3])==5 for l in endpoints),'Constraint endpoint inventory')
    interactions=(out/'clock_interaction.rpt').read_text()
    cross=next(l for l in interactions.splitlines() if re.match(r'memory_clk\s+transport_clk\s',l))
    need('unsafe' not in cross and 'Max Delay Datapath Only' in cross,'Unconstrained asynchronous status path')
    external=(out/'external_status.rpt').read_text()
    need(all(f'init_complete_async[{i}]' in external and f'startup/ready_meta_reg[{i}]/D' in external for i in range(2)),'External status path coverage')
    need('MaxDelay Path' in external,'External status constraint did not apply')
    counts=cdc(out/'cdc.rpt');need(counts=={'CDC-3':19,'CDC-9':12},'CDC topology changed')
    return dict(status='TARGETED_STATUS_CONSTRAINTS_AUDITED_TIMING_OPEN',cdc=counts,first_stage_endpoints=5,
                same_clock_slacks_preserved=True,clock_slack_ns={k:float(v) for k,v in m.items()},
                checkpoint_sha256=sha(out/'async_constrained.dcp'),qualified_timing=False,placed=False,routed=False)

def progress(out,r):
    need(r['donor_unchanged'],'Native donor changed')
    text=(out/'console.log').read_text();p=out/'evidence/progress.tsv'
    if not p.exists():
        need('APB_INTERNAL_VISIBILITY_UNAVAILABLE' in text,'Missing observation log')
        return dict(status='APB_SCOPE_DIAGNOSED_WITHOUT_SIMULATION',native_boot_validated=False,
                    next_scope='/tb_ntrpu_hbm_boot/dut/memory/inst/TWO_STACK.u_hbm_top')
    rows=list(csv.DictReader(p.read_text().splitlines(),delimiter='\t'))
    def ns(s):
        m=re.fullmatch(r'([\d.]+)(fs|ps|ns|us|ms|s)',s);need(m,'Time unit')
        return float(m[1])*{'fs':1e-6,'ps':1e-3,'ns':1,'us':1000,'ms':1e6,'s':1e9}[m[2]]
    times=[ns(x['simulation_time']) for x in rows]
    need(times[0]==0 and all(b-a==1000 for a,b in zip(times,times[1:])),'Observation cadence')
    need(len(rows)>=6 and all(x['mem_rst_n']=='1' and x['apb_rst_n']=='1' for x in rows if ns(x['simulation_time'])>=5000),'Reset release missing')
    need(not re.search(r'^\s*(Fatal:|ERROR:)',text,re.M),'Native simulator failure')
    available=[k for k in rows[0] if all(x[k]!='UNAVAILABLE' for x in rows)]
    internal=[k for k in available if k.startswith(('xpm_','gen_','init_seq_','intrnl_','valid_window_','apb_poll_'))]
    observations={k:dict(distinct=sorted({x[k] for x in rows[1:]}),last=rows[-1][k]) for k in internal}
    # APB PRDATA/PSLVERR can be unspecified outside a valid transfer. Record
    # unknown samples rather than interpreting idle bus values as failures.
    unknown={key:sum(bool(re.search('[xz]',x[key],re.I)) for x in rows if ns(x['simulation_time'])>=5000) for key in internal}
    deltas=[(int(b['wall_ms'])-int(a['wall_ms']))/1000 for a,b in zip(rows[5:],rows[6:])]
    return dict(status='NATIVE_PROGRESS_OBSERVED_BOOT_UNCONFIRMED',samples=len(rows),last_simulated_us=times[-1]/1000,
                last_flags=[rows[-1]['apb_complete_0'],rows[-1]['apb_complete_1']],
                post_reset_seconds_per_us=dict(min=min(deltas),median=statistics.median(deltas),max=max(deltas)),
                internal_signals_observable=len(internal),internal=observations,unknown_post_reset_samples=unknown,native_boot_validated='BOOT PASS ' in text,
                hard_hbm_traffic_validated=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--prior',type=Path);ap.add_argument('--review',type=Path,required=True)
    a=ap.parse_args();r=base(a.output)
    if r['experiment']=='hbm_cdc_audit':result=synthesis(a.output,r)
    elif r['experiment']=='hbm_cdc_constraints':result=constraints(a.output,r,a.prior)
    else:result=progress(a.output,r)
    result.update(seconds=r['seconds'],checker_sha256=sha(Path(__file__)))
    a.review.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
